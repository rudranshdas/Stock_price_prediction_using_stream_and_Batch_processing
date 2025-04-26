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
    os.listdir('C:\\hadoop\\bin')  # This should show winutils.exe
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

# Metrics tracking
batch_times = []
batch_stats = {
    'successful_runs': 0,
    'failed_runs': 0,
    'data_points_processed': 0,
    'last_run_time': None
}

def print_batch_stats():
    """Print detailed statistics about the batch processing"""
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
    """Fetch data with LIMIT to prevent memory issues"""
    print("\nFetching data from database...")
    start_time = time.time()
    
    df = spark.read \
        .format("jdbc") \
        .option("url", "jdbc:mysql://localhost:3306/stock_prediction") \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .option("dbtable", "(SELECT * FROM stock_ticks_raw ORDER BY timestamp DESC LIMIT 10000) as tmp") \
        .option("user", "root") \
        .option("password", "") \
        .load()
    
    count = df.count()
    duration = time.time() - start_time
    print(f"Fetched {count} records in {duration:.2f} seconds")
    return df

def retrain_model():
    try:
        start_time = time.time()
        batch_stats['last_run_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Fetch data
        df = fetch_data()
        
        # Convert to Pandas in batches
        print("Converting Spark DataFrame to Pandas...")
        pdf = df.limit(10000).toPandas()  # Explicit limit
        prices = pdf['price'].values
        record_count = len(prices)
        batch_stats['data_points_processed'] += record_count
        
        print(f"Processing {record_count} data points...")
        
        if len(prices) < 100:
            print(f"Warning: Insufficient data ({len(prices)} records), minimum 100 required")
            batch_stats['failed_runs'] += 1
            return False
            
        # Process in smaller chunks
        chunk_size = 5000
        print(f"Processing data in chunks of {chunk_size}...")
        
        for i in range(0, len(prices), chunk_size):
            chunk = prices[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1} (records {i} to {min(i+chunk_size, len(prices))})")
            # Your training logic here
            # Example: time.sleep(0.1)  # Simulate processing
        
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
            else:
                print_batch_stats()
                print("Retrying in 5 minutes...")
                time.sleep(300)  # 5 minutes for retry
                continue
                
            time.sleep(60)  # Run hourly (3600 seconds)
            
    except KeyboardInterrupt:
        print("\n\n=== Final Batch Processing Report ===")
        print_batch_stats()
        print("Shutting down batch processor...")
    finally:
        spark.stop()
        print("Spark session closed")