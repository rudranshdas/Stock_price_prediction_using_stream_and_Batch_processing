from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
import tensorflow as tf
import pickle
import time
import os
from pyspark.sql.types import *

# ================== WINDOWS CONFIGURATION ==================
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] = f"{os.environ['PATH']};C:\\hadoop\\bin"
os.environ['SPARK_DIST_CLASSPATH'] = os.path.join(os.environ['HADOOP_HOME'], 'bin')

# Verify winutils
try:
    os.listdir('C:\\hadoop\\bin')
    print("WinUtils verified")
except Exception as e:
    print(f"WinUtils error: {e}")
    exit()

# ================== SPARK CONFIGURATION ==================
spark = SparkSession.builder \
    .appName("KafkaStockConsumer") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,mysql:mysql-connector-java:8.0.33") \
    .config("spark.driver.extraClassPath", "mysql-connector-j-8.0.33.jar") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

# ================== MYSQL CONFIGURATION ==================
MYSQL_CONFIG = {
    "url": "jdbc:mysql://localhost:3306/stock_prediction",
    "driver": "com.mysql.cj.jdbc.Driver",
    "dbtable": "stock_ticks_raw",
    "user": "root",
    "password": ""
}

# ================== MODEL LOADING ==================
model = tf.keras.models.load_model('lstm_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Metrics tracking
stream_times = []

def predict_next_price(prices):
    scaled_prices = np.array([(p - scaler['min']) / (scaler['max'] - scaler['min']) for p in prices[-5:]])
    seq = scaled_prices.reshape(1, 5, 1)
    prediction = model.predict(seq, verbose=0)[0][0]
    return prediction * (scaler['max'] - scaler['min']) + scaler['min']

def process_batch(batch_df, batch_id):
    start_time = time.perf_counter()
    
    if not batch_df.isEmpty():
        # Store raw data to MySQL
        batch_df.write \
            .format("jdbc") \
            .option("url", MYSQL_CONFIG["url"]) \
            .option("driver", MYSQL_CONFIG["driver"]) \
            .option("dbtable", MYSQL_CONFIG["dbtable"]) \
            .option("user", MYSQL_CONFIG["user"]) \
            .option("password", MYSQL_CONFIG["password"]) \
            .mode("append") \
            .save()
        
        # Process predictions
        batch_pd = batch_df.orderBy("timestamp").toPandas()
        prices = batch_pd['price'].tolist()
        
        if len(prices) >= 5:
            predicted = predict_next_price(prices)
            processing_time = time.perf_counter() - start_time
            stream_times.append(processing_time)
            print(f"Processed {len(prices)} records | Predicted: {predicted:.2f} | Time: {processing_time:.4f}s")

# ================== KAFKA CONFIGURATION ==================
schema = StructType([
    StructField("symbol", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("price", DoubleType()),
    StructField("volume", LongType())
])

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "stock_ticks") \
    .load()

# Parse and process
parsed_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

query = parsed_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .start()

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\nStream Processing Report:")
    print(f"Total messages: {len(stream_times)}")
    print(f"Average time: {np.mean(stream_times):.4f}s")
    print(f"Fastest: {np.min(stream_times):.4f}s | Slowest: {np.max(stream_times):.4f}s")
    query.stop()
finally:
    spark.stop()