# save as: test_accuracy.py
import numpy as np
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy_test():
    """Test model accuracy with synthetic data"""
    print("="*60)
    print("NIDS ACCURACY VALIDATION TEST")
    print("="*60)
    
    # Load model
    model = joblib.load('models/basic_nids_model.pkl')
    n_features = model.n_features_in_
    
    # Generate synthetic test data
    n_samples = 10000
    print(f"\nGenerating {n_samples:,} test samples...")
    
    # Create balanced dataset (50% normal, 50% attack)
    X_test = []
    y_test = []
    
    for i in range(n_samples):
        if i % 2 == 0:  # Normal traffic
            features = np.random.rand(n_features) * 10  # Low values
            label = 0
        else:  # Attack traffic
            features = np.random.rand(n_features) * 1000  # High values
            # Make some features very high (attack patterns)
            for j in range(5):
                features[j] = np.random.rand() * 10000
            label = 1
        
        X_test.append(features)
        y_test.append(label)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"   Normal samples: {sum(y_test == 0):,}")
    print(f"   Attack samples: {sum(y_test == 1):,}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of attack
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC-ROC:     {auc:.4f} ({auc*100:.2f}%)")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"               Normal  Attack")
    print(f"Actual Normal  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Attack  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Confusion Matrix Heatmap
    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax = axes[0, 1]
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence Distribution
    ax = axes[1, 0]
    normal_conf = y_pred_proba[y_test == 0]
    attack_conf = y_pred_proba[y_test == 1]
    
    ax.hist(normal_conf, bins=30, alpha=0.5, label='Normal', color='green')
    ax.hist(attack_conf, bins=30, alpha=0.5, label='Attack', color='red')
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Bar Chart
    ax = axes[1, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [accuracy, precision, recall, f1, auc]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('test_results/accuracy_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    import os
    os.makedirs('test_results', exist_ok=True)
    
    results = {
        'test_samples': n_samples,
        'normal_samples': int(sum(y_test == 0)),
        'attack_samples': int(sum(y_test == 1)),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_score': float(auc),
        'confusion_matrix': cm.tolist(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('test_results/accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Test results saved to 'test_results/' directory")
    
    # Performance rating
    print("\n" + "="*60)
    print("PERFORMANCE RATING")
    print("="*60)
    
    if accuracy >= 0.95:
        print("🎉 EXCELLENT: Model accuracy > 95%")
        print("   Your NIDS is performing at production level!")
    elif accuracy >= 0.90:
        print("👍 GOOD: Model accuracy 90-95%")
        print("   Your NIDS is performing well.")
    elif accuracy >= 0.80:
        print("⚠️  FAIR: Model accuracy 80-90%")
        print("   Consider retraining with more data.")
    else:
        print("❌ POOR: Model accuracy < 80%")
        print("   Model needs improvement. Check training data.")

if __name__ == "__main__":
    import time
    accuracy_test()