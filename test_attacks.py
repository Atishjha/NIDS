# save as: test_attacks.py
import numpy as np
import joblib
import time
from collections import defaultdict

def simulate_attack_test():
    """Test detection of different attack types"""
    print("="*60)
    print("ATTACK DETECTION TEST SUITE")
    print("="*60)
    
    # Load model
    model = joblib.load('models/basic_nids_model.pkl')
    n_features = model.n_features_in_
    
    attack_types = {
        'DoS': {
            'description': 'Denial of Service - High traffic volume',
            'features': lambda: np.random.rand(n_features) * 10000,
            'expected': 'ATTACK'
        },
        'Port Scan': {
            'description': 'Scanning multiple ports',
            'features': lambda: np.concatenate([
                np.random.rand(10) * 5000,  # High count features
                np.random.rand(n_features-10) * 10
            ]),
            'expected': 'ATTACK'
        },
        'Data Exfiltration': {
            'description': 'Large data transfer',
            'features': lambda: np.concatenate([
                np.random.rand(5) * 100000,  # Very high byte features
                np.random.rand(n_features-5) * 100
            ]),
            'expected': 'ATTACK'
        },
        'Normal Web': {
            'description': 'Normal web browsing',
            'features': lambda: np.random.rand(n_features) * 100,
            'expected': 'NORMAL'
        },
        'Normal Email': {
            'description': 'Email traffic',
            'features': lambda: np.random.rand(n_features) * 50,
            'expected': 'NORMAL'
        }
    }
    
    print("\nTesting attack detection...")
    print(f"Samples per test: 100")
    print(f"Confidence threshold: 70%")
    print()
    
    results = {}
    
    for attack_name, config in attack_types.items():
        print(f"\nTesting: {attack_name}")
        print(f"  Description: {config['description']}")
        
        correct = 0
        total = 100
        confidences = []
        detection_times = []
        
        for i in range(total):
            # Generate features
            features = config['features']().reshape(1, -1)
            
            # Predict
            start = time.time()
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            detection_time = time.time() - start
            
            confidence = probability[1] if prediction == 1 else probability[0]
            confidences.append(confidence)
            detection_times.append(detection_time)
            
            # Check if correct
            predicted_label = 'ATTACK' if prediction == 1 else 'NORMAL'
            if predicted_label == config['expected']:
                correct += 1
        
        accuracy = correct / total
        avg_confidence = np.mean(confidences)
        avg_time = np.mean(detection_times) * 1000  # ms
        
        results[attack_name] = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_time_ms': avg_time,
            'expected': config['expected']
        }
        
        # Print results
        symbol = "✓" if accuracy >= 0.9 else "⚠" if accuracy >= 0.7 else "✗"
        print(f"  {symbol} Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
        print(f"    Avg confidence: {avg_confidence*100:.1f}%")
        print(f"    Avg detection time: {avg_time:.1f} ms")
        
        if accuracy < 0.7:
            print(f"    ⚠ WARNING: Poor detection for {attack_name}")
    
    # Summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    
    attack_detection = [r for name, r in results.items() if r['expected'] == 'ATTACK']
    normal_detection = [r for name, r in results.items() if r['expected'] == 'NORMAL']
    
    if attack_detection:
        avg_attack_acc = np.mean([r['accuracy'] for r in attack_detection])
        print(f"Attack Detection Rate: {avg_attack_acc*100:.1f}%")
    
    if normal_detection:
        avg_normal_acc = np.mean([r['accuracy'] for r in normal_detection])
        print(f"Normal Traffic Accuracy: {avg_normal_acc*100:.1f}%")
    
    # Performance rating
    print("\n" + "="*60)
    print("DETECTION PERFORMANCE RATING")
    print("="*60)
    
    if attack_detection and normal_detection:
        overall_acc = (avg_attack_acc + avg_normal_acc) / 2
        
        if overall_acc >= 0.95:
            print("🎉 EXCELLENT: Overall accuracy > 95%")
            print("   Your NIDS detects attacks very well!")
        elif overall_acc >= 0.90:
            print("👍 GOOD: Overall accuracy 90-95%")
            print("   Your NIDS performs well in attack detection.")
        elif overall_acc >= 0.80:
            print("⚠ FAIR: Overall accuracy 80-90%")
            print("   Detection could be improved.")
        else:
            print("❌ POOR: Overall accuracy < 80%")
            print("   Attack detection needs significant improvement.")
    
    # Save detailed results
    import json
    import os
    os.makedirs('test_results', exist_ok=True)
    
    with open('test_results/attack_detection.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_attack_detection_chart(results)

def create_attack_detection_chart(results):
    """Create visualization of attack detection results"""
    import matplotlib.pyplot as plt
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]
    colors = ['red' if results[name]['expected'] == 'ATTACK' else 'green' for name in names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, accuracies, color=colors, alpha=0.7)
    
    plt.xlabel('Traffic Type')
    plt.ylabel('Detection Accuracy (%)')
    plt.title('NIDS Attack Detection Performance')
    plt.ylim([0, 105])
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{accuracy:.1f}%', ha='center', va='bottom')
    
    # Add legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Attack Traffic')
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Normal Traffic')
    plt.legend(handles=[red_patch, green_patch])
    
    plt.tight_layout()
    plt.savefig('test_results/attack_detection_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Chart saved: test_results/attack_detection_chart.png")

if __name__ == "__main__":
    simulate_attack_test()