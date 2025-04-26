from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

symbol = "AAPL"
base_price = 180.0
volatility = 0.5

def generate_mock_tick():
    global base_price
    price_change = (random.random() - 0.5) * volatility * base_price / 100
    base_price += price_change
    
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "price": round(base_price, 2),
        "volume": random.randint(100000, 500000)
    }

if __name__ == "__main__":
    try:
        print(f"Producing mock data for {symbol}. Press Ctrl+C to stop.")
        while True:
            tick_data = generate_mock_tick()
            producer.send('stock_ticks', value=tick_data)
            time.sleep(0.1)  # Faster data generation for testing
    except KeyboardInterrupt:
        producer.close()