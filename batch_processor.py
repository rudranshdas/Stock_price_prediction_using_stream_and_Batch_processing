from pyspark.sql import SparkSession
import numpy as np
import tensorflow as tf
import pickle
import time
from datetime import datetime, timedelta
import os
import pandas as pd

# ================== WINDOWS CONFIGURATION ==================
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] = f"{os.environ['PATH']};C:\\hadoop\\bin"
os.environ['SPARK_DIST_CLASSPATH'] = os.path.join(os.environ['HADOOP_HOME'], 'bin')

# Verify winutils is accessible
try:
    os.listdir('C:\\hadoop\\bin')
    print("WinUtils configuration verified successfully")
except Exception as e:
    print(f"WinUtils access error: {e}")
    exit()

# ================== SPARK CONFIGURATION ==================
spark = SparkSession.builder \
    .appName("BatchStockProcessor") \
    .master("local[*]") \
    .config("spark.jars", "<path to /mysql-connector-j-9.3.0.jar>") \
    .config("spark.driver.extraClassPath", "<path to /mysql-connector-j-9.3.0.jar>") \
    .config("spark.executor.extraClassPath", "<path to /mysql-connector-j-9.3.0.jar>") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Metrics tracking
batch_times = []
batch_stats = {
    'successful_runs': 0,
    'failed_runs': 0,
    'data_points_processed': 0,
    'last_run_time': None
}

# Load the LSTM model once
print("Loading LSTM model...")
try:
    model = tf.keras.models.load_model('lstm_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def print_batch_stats():
    print("\n=== Batch Processing Statistics ===")
    print(f"Last run time: {batch_stats['last_run_time']}")
    print(f"Successful runs: {batch_stats['successful_runs']}")
    print(f"Failed runs: {batch_stats['failed_runs']}")
    print(f"Total data points processed: {batch_stats['data_points_processed']}")
    if batch_times:
        print(f"\nPerformance Metrics:")
        print(f"  Average processing time: {np.mean(batch_times):.2f}s")
        print(f"  Minimum processing time: {np.min(batch_times):.2f}s")
        print(f"  Maximum processing time: {np.max(batch_times):.2f}s")
        print(f"  Total processing time: {np.sum(batch_times):.2f}s")
    print("="*40 + "\n")

def fetch_data():
    print("\nFetching data from database...")
    start_time = time.time()
    
    df = spark.read \
        .format("jdbc") \
        .option("url", "jdbc:mysql://localhost:3306/stock_prediction") \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .option("dbtable", "(SELECT * FROM stock_ticks_raw ORDER BY timestamp DESC LIMIT 10000) as tmp") \
        .option("user", "root") \
        .option("password", "<your password>") \
        .load()
    
    
    count = df.count()
    duration = time.time() - start_time
    print(f"Fetched {count} records in {duration:.2f} seconds")
    return df


def make_predictions(prices):
    sequence_length = 60
    sequences = []

    # Use latest 60 prices for prediction (sliding window)
    for i in range(len(prices) - sequence_length):
        window = prices[i:i+sequence_length]
        sequences.append(window)

    sequences = np.array(sequences).reshape(-1, sequence_length, 1)
    
    print(f"Running predictions on {sequences.shape[0]} sequences...")
    preds = model.predict(sequences, verbose=0)
    return preds.flatten()

def retrain_model():
    try:
        if model is None:
            raise RuntimeError("Model not loaded. Cannot proceed.")

        start_time = time.time()
        batch_stats['last_run_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        df = fetch_data()
        
        print("Converting Spark DataFrame to Pandas and sorting by timestamp...")
        pdf = df.limit(10000).toPandas().sort_values('timestamp')
        prices = pdf['price'].astype(float).values
        record_count = len(prices)
        batch_stats['data_points_processed'] += record_count
        
        print(f"Processing {record_count} data points...")

        if len(prices) < 100:
            print(f"Warning: Insufficient data ({len(prices)} records), minimum 100 required")
            batch_stats['failed_runs'] += 1
            return False

        chunk_size = 5000
        print(f"Processing data in chunks of {chunk_size}...")

        for i in range(0, len(prices), chunk_size):
            chunk = prices[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1} (records {i} to {min(i+chunk_size, len(prices))})")
            # Placeholder: Simulate processing

        # Predict stock price movement using latest available data
        if len(prices) >= 60:
            predictions = make_predictions(prices)
            print("Latest predictions (last 5):", predictions[-5:])
        else:
            print("Not enough data to generate prediction window.")

        duration = time.time() - start_time
        batch_times.append(duration)
        batch_stats['successful_runs'] += 1
        
        print(f"Batch processing completed successfully in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"\nError during retraining: {str(e)}")
        batch_stats['failed_runs'] += 1
        return False


if __name__ == "__main__":
    try:
        print("Batch processor started. Press Ctrl+C to stop.")
        while True:
            print(f"\n{'='*40}")
            print(f"Starting new batch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if retrain_model():
                print_batch_stats()
                print("Next retraining in 1 hour...")
                time.sleep(60)
            else:
                print_batch_stats()
                print("Retrying in 5 minutes...")
                time.sleep(300)
                
    except KeyboardInterrupt:
        print("\n\n=== Final Batch Processing Report ===")
        print_batch_stats()
        print("Shutting down batch processor...")
    finally:
        spark.stop()
        print("Spark session closed")
