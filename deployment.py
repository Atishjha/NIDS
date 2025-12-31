import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')
from config import Config

class RealTimeNIDS:
    """Real-time Network Intrusion Detection System"""
    
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        """Initialize real-time NIDS"""
        self.config = Config()
        self.model = joblib.load(model_path)
        
        # Load preprocessing artifacts
        artifacts = joblib.load('models/preprocessing_artifacts.pkl')
        self.label_encoders = artifacts['label_encoders']
        self.selected_features = artifacts['selected_features']
        
        # Load scaler
        self.scaler = joblib.load('models/scaler.pkl')
        
        # Statistics
        self.total_packets = 0
        self.attack_packets = 0
        self.attack_history = []
        
    def preprocess_single(self, packet_features: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single packet's features"""
        # Convert to DataFrame
        df = pd.DataFrame([packet_features])
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                if df[col].iloc[0] not in encoder.classes_:
                    df[col] = 'UNKNOWN'
                df[col] = encoder.transform([df[col].iloc[0]])[0]
        
        # Scale numerical features
        numerical_cols = [col for col in self.config.NUMERICAL_FEATURES 
                         if col in df.columns]
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Select features
        if self.selected_features:
            df = df[self.selected_features]
        
        return df.values
    
    def predict(self, packet_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if a packet is malicious"""
        self.total_packets += 1
        
        # Preprocess
        X = self.preprocess_single(packet_features)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Get class name
        class_name = self.config.UNIFIED_LABELS.get(prediction, 'Unknown')
        
        result = {
            'is_attack': prediction != 0,
            'predicted_class': prediction,
            'class_name': class_name,
            'probability': probability[prediction] if probability is not None else 1.0,
            'all_probabilities': probability.tolist() if probability is not None else []
        }
        
        # Update statistics
        if result['is_attack']:
            self.attack_packets += 1
            self.attack_history.append({
                'timestamp': pd.Timestamp.now(),
                'attack_type': class_name,
                'probability': result['probability'],
                'features': packet_features
            })
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        attack_rate = (self.attack_packets / self.total_packets * 100) if self.total_packets > 0 else 0
        
        return {
            'total_packets': self.total_packets,
            'attack_packets': self.attack_packets,
            'attack_rate_percentage': attack_rate,
            'normal_packets': self.total_packets - self.attack_packets,
            'recent_attacks': self.attack_history[-10:] if self.attack_history else []
        }
    
    def reset_statistics(self):
        """Reset statistics"""
        self.total_packets = 0
        self.attack_packets = 0
        self.attack_history = []
        
    def save_attack_log(self, filename: str = 'attack_log.csv'):
        """Save attack history to file"""
        if self.attack_history:
            df = pd.DataFrame(self.attack_history)
            df.to_csv(f"{self.config.RESULTS_DIR}/{filename}", index=False)
            print(f"Attack log saved with {len(df)} entries")

# Example usage for real-time monitoring
def monitor_live_traffic():
    """Example of real-time monitoring"""
    nids = RealTimeNIDS()
    
    # Simulate live packets (replace with actual packet capture)
    sample_packets = [
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF', 'src_bytes': 100},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'ftp', 'flag': 'SF', 'src_bytes': 5000},
        # Add more packet features...
    ]
    
    for i, packet in enumerate(sample_packets):
        result = nids.predict(packet)
        
        if result['is_attack']:
            print(f"🚨 ALERT: Attack detected! Type: {result['class_name']}, Confidence: {result['probability']:.2%}")
        else:
            print(f"✅ Normal traffic")
    
    # Print statistics
    stats = nids.get_statistics()
    print(f"\nStatistics:")
    print(f"Total packets: {stats['total_packets']}")
    print(f"Attack packets: {stats['attack_packets']}")
    print(f"Attack rate: {stats['attack_rate_percentage']:.2f}%")