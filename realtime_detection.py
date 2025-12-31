import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime

class RealTimeNIDS:
    """Real-time Network Intrusion Detection"""
    
    def __init__(self, model_path='models/best_nids_model.pkl',
                 encoder_path='models/label_encoder.pkl',
                 threshold=0.8):
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.threshold = threshold
        self.stats = {
            'total_packets': 0,
            'normal_packets': 0,
            'attack_packets': 0,
            'attack_types': {},
            'start_time': datetime.now()
        }
        self.recent_alerts = []
        
    def preprocess_packet(self, packet_features):
        """Preprocess a single packet"""
        # Convert to DataFrame
        df = pd.DataFrame([packet_features])
        
        # Handle missing values
        df = df.fillna(0)
        
        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.fillna(0)
        
        return df
    
    def predict(self, packet_features):
        """Predict on a single packet"""
        self.stats['total_packets'] += 1
        
        # Preprocess
        X = self.preprocess_packet(packet_features)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        confidence = np.max(probability)
        
        # Decode label
        attack_type = self.label_encoder.inverse_transform([prediction])[0]
        
        # Update statistics
        if attack_type == 'Normal':
            self.stats['normal_packets'] += 1
            is_attack = False
        else:
            self.stats['attack_packets'] += 1
            is_attack = True
            
            # Update attack type counts
            if attack_type in self.stats['attack_types']:
                self.stats['attack_types'][attack_type] += 1
            else:
                self.stats['attack_types'][attack_type] = 1
            
            # Add to recent alerts if confidence is high
            if confidence >= self.threshold:
                alert = {
                    'timestamp': datetime.now(),
                    'attack_type': attack_type,
                    'confidence': confidence,
                    'features': packet_features
                }
                self.recent_alerts.append(alert)
                # Keep only last 100 alerts
                if len(self.recent_alerts) > 100:
                    self.recent_alerts.pop(0)
                
                # Print alert
                print(f"\n🚨 ALERT: {attack_type} detected!")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   Time: {alert['timestamp']}")
        
        return {
            'is_attack': is_attack,
            'attack_type': attack_type,
            'confidence': confidence,
            'all_probabilities': dict(zip(self.label_encoder.classes_, probability))
        }
    
    def get_statistics(self):
        """Get current statistics"""
        total_time = (datetime.now() - self.stats['start_time']).total_seconds()
        
        stats_summary = {
            'total_packets': self.stats['total_packets'],
            'normal_packets': self.stats['normal_packets'],
            'attack_packets': self.stats['attack_packets'],
            'attack_rate': (self.stats['attack_packets'] / self.stats['total_packets'] * 100) 
                          if self.stats['total_packets'] > 0 else 0,
            'packets_per_second': self.stats['total_packets'] / total_time if total_time > 0 else 0,
            'attack_types': self.stats['attack_types'],
            'recent_alerts': self.recent_alerts[-10:],  # Last 10 alerts
            'monitoring_duration': total_time
        }
        
        return stats_summary
    
    def print_dashboard(self):
        """Print real-time dashboard"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("REAL-TIME NIDS DASHBOARD")
        print("="*60)
        print(f"Monitoring Duration: {stats['monitoring_duration']:.1f} seconds")
        print(f"Total Packets: {stats['total_packets']:,}")
        print(f"Normal Packets: {stats['normal_packets']:,} ({stats['attack_rate']:.1f}% attack rate)")
        print(f"Attack Packets: {stats['attack_packets']:,}")
        print(f"Processing Speed: {stats['packets_per_second']:.1f} packets/sec")
        
        if stats['attack_types']:
            print(f"\nAttack Type Distribution:")
            for attack_type, count in stats['attack_types'].items():
                percentage = (count / stats['attack_packets'] * 100) if stats['attack_packets'] > 0 else 0
                print(f"  {attack_type:20s}: {count:6d} ({percentage:5.1f}%)")
        
        if stats['recent_alerts']:
            print(f"\nRecent Alerts (last 10):")
            for alert in stats['recent_alerts']:
                print(f"  {alert['timestamp'].strftime('%H:%M:%S')} - {alert['attack_type']} "
                      f"(Confidence: {alert['confidence']:.2%})")
        
        print("="*60)

# Example usage
def simulate_realtime_traffic():
    """Simulate real-time traffic for testing"""
    nids = RealTimeNIDS()
    
    print("Starting real-time NIDS simulation...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Simulate packets
        packet_count = 0
        while True:
            # Generate random packet features (simulation)
            packet_features = {
                'duration': np.random.uniform(0, 100),
                'src_bytes': np.random.randint(0, 10000),
                'dst_bytes': np.random.randint(0, 10000),
                'count': np.random.randint(1, 100),
                'srv_count': np.random.randint(1, 50),
                'same_srv_rate': np.random.uniform(0, 1),
                'diff_srv_rate': np.random.uniform(0, 1),
                'dst_host_count': np.random.randint(1, 100),
                'dst_host_srv_count': np.random.randint(1, 50),
                'dst_host_same_srv_rate': np.random.uniform(0, 1)
            }
            
            # Occasionally add attack-like features
            if np.random.random() < 0.1:  # 10% chance of attack
                packet_features['src_bytes'] = np.random.randint(10000, 100000)
                packet_features['count'] = np.random.randint(100, 1000)
            
            # Predict
            result = nids.predict(packet_features)
            
            packet_count += 1
            
            # Print dashboard every 100 packets
            if packet_count % 100 == 0:
                nids.print_dashboard()
            
            time.sleep(0.01)  # Simulate packet interval
            
    except KeyboardInterrupt:
        print("\n\nStopping real-time NIDS...")
        nids.print_dashboard()

if __name__ == "__main__":
    # Run simulation
    simulate_realtime_traffic()