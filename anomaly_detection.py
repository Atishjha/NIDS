# anomaly_detection.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from pyod.models.auto_encoder import AutoEncoder

class AnomalyDetector:
    """Anomaly detection for unknown attacks"""
    
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=0.1,
                kernel='rbf',
                gamma='scale'
            )
        }
        
    def train_on_normal(self, normal_data):
        """Train on normal traffic only"""
        for name, model in self.models.items():
            model.fit(normal_data)
            print(f"Trained {name} on {len(normal_data)} normal samples")
    
    def detect_anomalies(self, data):
        """Detect anomalies in new data"""
        results = {}
        
        for name, model in self.models.items():
            predictions = model.predict(data)
            anomaly_scores = model.decision_function(data)
            
            results[name] = {
                'predictions': predictions,  # -1 for anomaly, 1 for normal
                'scores': anomaly_scores,
                'anomaly_count': np.sum(predictions == -1)
            }
            
        return results