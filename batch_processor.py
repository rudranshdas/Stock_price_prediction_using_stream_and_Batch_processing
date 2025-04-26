from pyspark.sql import SparkSession
import numpy as np
import tensorflow as tf
import pickle
import time
from datetime import datetime, timedelta
import os

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
    .config("spark.jars", "file:///C:/Users/Nithin%20ramakrishnan/dbt_3_proj/mysql-connector-j-9.3.0.jar") \
    .config("spark.driver.extraClassPath", "file:///C:/Users/Nithin%20ramakrishnan/dbt_3_proj/mysql-connector-j-9.3.0.jar") \
    .config("spark.executor.extraClassPath", "file:///C:/Users/Nithin%20ramakrishnan/dbt_3_proj/mysql-connector-j-9.3.0.jar") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ================== MODEL CONFIGURATION ==================
MODEL_PATH = 'lstm_model.keras'
SCALER_PATH = 'scaler.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Successfully loaded model and scaler")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    spark.stop()
    exit()

# Metrics tracking
batch_times = []
prediction_history = []

def predict_next_price(prices):
    """Predict next price using the last 5 prices (aligned with streaming approach)"""
    scaled_prices = np.array([(float(p) - scaler['min']) / (scaler['max'] - scaler['min']) for p in prices[-5:]])
    seq = scaled_prices.reshape(1, 5, 1)
    prediction = model.predict(seq, verbose=0)[0][0]
    return prediction * (scaler['max'] - scaler['min']) + scaler['min']

def fetch_data():
    """Fetch data with LIMIT to prevent memory issues"""
    print("\nFetching data from database...")
    return spark.read \
        .format("jdbc") \
        .option("url", "jdbc:mysql://localhost:3306/stock_prediction") \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .option("dbtable", "(SELECT symbol, timestamp, price FROM stock_ticks_raw ORDER BY timestamp DESC LIMIT 1000) as tmp") \
        .option("user", "root") \
        .option("password", "") \
        .load()

def process_predictions(pdf):
    """Process predictions on the batch data"""
    prices = pdf['price'].values
    predictions = []
    
    # We need at least 5 prices to make a prediction
    if len(prices) >= 5:
        # Make predictions for each 5-price window
        for i in range(5, len(prices)+1):
            window = prices[i-5:i]
            predicted_price = predict_next_price(window)
            predictions.append({
                'timestamp': pdf.iloc[i-1]['timestamp'],
                'actual_price': prices[i-1],
                'predicted_price': predicted_price
            })
            
            # Store prediction for reporting
            prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': predicted_price,
                'actual': prices[i-1] if i < len(prices) else None
            })
    
    return predictions

def retrain_model():
    try:
        start_time = time.time()
        
        # 1. Fetch data
        df = fetch_data()
        pdf = df.toPandas()
        pdf['price'] = pdf['price'].astype(float)  # Ensure float prices
        if len(pdf) < 5:
            print(f"Insufficient data ({len(pdf)} records), need at least 5")
            return False
            
        # 2. Process predictions
        predictions = process_predictions(pdf)
        
        if not predictions:
            print("No predictions made - insufficient data windows")
            return False
            
        # 3. Calculate metrics
        last_pred = predictions[-1]
        duration = time.time() - start_time
        batch_times.append(duration)
        
        # 4. Print results
        print("\n=== Prediction Results ===")
        print(f"Last actual price: {last_pred['actual_price']:.2f}")
        print(f"Last predicted price: {last_pred['predicted_price']:.2f}")
        
        if len(predictions) > 1:
            errors = [abs(p['actual_price'] - p['predicted_price']) for p in predictions[:-1]]
            print(f"Mean Absolute Error: {np.mean(errors):.4f}")
            print(f"Predictions made: {len(predictions)}")
        
        print(f"Processing time: {duration:.2f}s")
        return True
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        print("Batch prediction processor started. Press Ctrl+C to stop.")
        while True:
            print(f"\n{'='*40}")
            print(f"Starting new batch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if retrain_model():
                print("\nNext batch in 1 hour...")
            else:
                print("\nRetrying in 5 minutes...")
                time.sleep(30)  # 5 minutes for retry
                continue
                
            time.sleep(60)  # Run hourly (3600 seconds)
            
    except KeyboardInterrupt:
        print("\n\n=== Final Prediction Report ===")
        print(f"Total batches processed: {len(batch_times)}")
        if batch_times:
            print(f"Average processing time: {np.mean(batch_times):.2f}s")
            print(f"Total predictions made: {len(prediction_history)}")
            
        if prediction_history:
            last_actual = [p['actual'] for p in prediction_history if p['actual'] is not None]
            if last_actual:
                errors = [abs(p['actual'] - p['prediction']) for p in prediction_history if p['actual'] is not None]
                print(f"Overall MAE: {np.mean(errors):.4f}")
                
        print("\nShutting down processor...")
    finally:
        spark.stop()
        print("Spark session closed")