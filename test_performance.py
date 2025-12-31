# save as: test_performance.py
import time
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import psutil
import os

def performance_test():
    """Test system performance"""
    print("="*60)
    print("NIDS PERFORMANCE TEST")
    print("="*60)
    
    # 1. Model loading speed
    print("\n1. Model Loading Performance:")
    start = time.time()
    model = joblib.load('models/basic_nids_model.pkl')
    load_time = time.time() - start
    print(f"   Load time: {load_time*1000:.1f} ms")
    print(f"   Model size: {os.path.getsize('models/basic_nids_model.pkl')/1024:.1f} KB")
    
    # 2. Prediction speed
    print("\n2. Prediction Performance:")
    n_samples = 1000
    n_features = model.n_features_in_
    
    # Generate test data
    X_test = np.random.rand(n_samples, n_features)
    
    # Single prediction
    start = time.time()
    single_pred = model.predict(X_test[:1])
    single_time = time.time() - start
    print(f"   Single prediction: {single_time*1000:.2f} ms")
    
    # Batch prediction
    start = time.time()
    batch_pred = model.predict(X_test)
    batch_time = time.time() - start
    print(f"   1000 predictions: {batch_time*1000:.2f} ms")
    print(f"   Average per prediction: {batch_time/n_samples*1000:.2f} ms")
    print(f"   Throughput: {n_samples/batch_time:.0f} predictions/sec")
    
    # 3. Memory usage
    print("\n3. Memory Usage:")
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"   Current memory: {memory_mb:.1f} MB")
    
    # 4. CPU usage during prediction
    print("\n4. CPU Usage Test:")
    cpu_start = psutil.cpu_percent(interval=0.1)
    
    # Heavy prediction test
    start = time.time()
    for i in range(100):
        model.predict(X_test[i:i+1])
    predict_time = time.time() - start
    
    cpu_end = psutil.cpu_percent(interval=0.1)
    print(f"   100 sequential predictions: {predict_time*1000:.0f} ms")
    print(f"   CPU usage: {cpu_end:.1f}%")
    
    # 5. Real-time simulation
    print("\n5. Real-time Simulation:")
    packets_per_second = 100
    test_duration = 5  # seconds
    
    print(f"   Simulating {packets_per_second} packets/sec for {test_duration}s")
    
    processed = 0
    start_time = time.time()
    
    while time.time() - start_time < test_duration:
        # Generate packet
        packet = np.random.rand(1, n_features)
        
        # Predict
        model.predict(packet)
        
        processed += 1
        time.sleep(1/packets_per_second)
    
    actual_pps = processed / test_duration
    print(f"   Processed: {processed} packets")
    print(f"   Actual rate: {actual_pps:.1f} packets/sec")
    
    # Performance rating
    print("\n" + "="*60)
    print("PERFORMANCE RATING")
    print("="*60)
    
    ratings = []
    
    # Load time rating
    if load_time < 0.1:
        ratings.append(("Model Load", "Excellent", "✓"))
    elif load_time < 0.5:
        ratings.append(("Model Load", "Good", "✓"))
    else:
        ratings.append(("Model Load", "Slow", "⚠"))
    
    # Prediction speed rating
    avg_pred_time = batch_time / n_samples
    if avg_pred_time < 0.001:
        ratings.append(("Prediction Speed", "Excellent", "✓"))
    elif avg_pred_time < 0.01:
        ratings.append(("Prediction Speed", "Good", "✓"))
    else:
        ratings.append(("Prediction Speed", "Slow", "⚠"))
    
    # Throughput rating
    throughput = n_samples / batch_time
    if throughput > 1000:
        ratings.append(("Throughput", "Excellent", "✓"))
    elif throughput > 100:
        ratings.append(("Throughput", "Good", "✓"))
    else:
        ratings.append(("Throughput", "Low", "⚠"))
    
    # Memory rating
    if memory_mb < 100:
        ratings.append(("Memory Usage", "Excellent", "✓"))
    elif memory_mb < 500:
        ratings.append(("Memory Usage", "Good", "✓"))
    else:
        ratings.append(("Memory Usage", "High", "⚠"))
    
    for category, rating, symbol in ratings:
        print(f"   {symbol} {category}: {rating}")
    
    print(f"\nOverall Performance: {'GOOD' if '⚠' not in [r[2] for r in ratings] else 'NEEDS IMPROVEMENT'}")

if __name__ == "__main__":
    performance_test()