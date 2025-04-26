import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual metrics)
stream_times = np.random.normal(0.03, 0.005, 100)  # 30ms ± 5ms
batch_times = np.random.normal(45, 5, 5)           # 45s ± 5s

plt.figure(figsize=(12, 6))

# Stream Processing
plt.subplot(1, 2, 1)
plt.plot(stream_times * 1000, 'b.')  # Convert to milliseconds
plt.title("Stream Processing Latency")
plt.ylabel("Milliseconds")
plt.grid(True)

# Batch Processing
plt.subplot(1, 2, 2)
plt.bar(range(len(batch_times)), batch_times, color='orange')
plt.title("Batch Processing Duration")
plt.ylabel("Seconds")
plt.grid(True)

plt.tight_layout()
plt.savefig("processing_times.png")
print("Visualization saved to processing_times.png")