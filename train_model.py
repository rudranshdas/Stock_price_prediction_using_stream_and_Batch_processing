import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta  # Added timedelta import

# Configuration
SYMBOL = "AAPL"
SEQ_LENGTH = 5
MODEL_PATH = "lstm_model.keras"
SCALER_PATH = "scaler.pkl"

def generate_historical_data(days=30):
    """Generate synthetic training data"""
    np.random.seed(42)
    base = 180.0
    prices = [base + (np.random.random() - 0.5) * 2 * i/100 for i in range(1000)]
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(len(prices))]
    return np.array(prices), timestamps

def create_sequences(data, seq_length):
    """Convert time series into LSTM sequences"""
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_and_save_model():
    # 1. Load data
    prices, _ = generate_historical_data()
    
    # 2. Scale data
    price_min, price_max = np.min(prices), np.max(prices)
    scaled_prices = (prices - price_min) / (price_max - price_min)
    
    # 3. Prepare sequences
    X, y = create_sequences(scaled_prices, SEQ_LENGTH)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # 4. Build model
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # 5. Train
    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # 6. Save artifacts
    model.save(MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'min': price_min, 'max': price_max}, f)
    
    # 7. Plot training
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("training_history.png")
    print(f"Model and scaler saved. Training graph generated.")

if __name__ == "__main__":
    train_and_save_model()