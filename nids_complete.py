import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def create_directories():
    """Create necessary directories"""
    for dir_name in ['models', 'results', 'plots']:
        os.makedirs(dir_name, exist_ok=True)

def train_basic_model():
    """Train basic binary classification model (Normal vs Attack)"""
    print("="*70)
    print("TRAINING BASIC NIDS MODEL")
    print("Binary Classification: Normal vs Attack")
    print("="*70)
    
    start_time = time.time()
    
    datasets = []
    
    # 1. NSL-KDD Train
    print("\n[1] Processing NSL-KDD Train...")
    try:
        df = pd.read_parquet('KDDTrain.parquet')
        # Find label column
        label_col = None
        for col in df.columns:
            if col.lower() in ['label', 'target', 'class']:
                label_col = col
                break
        
        if label_col:
            df['is_attack'] = ~df[label_col].astype(str).str.contains('normal', case=False)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            # Use first 30 features for speed
            selected_cols = numeric_cols[:30] + ['is_attack']
            datasets.append(df[selected_cols])
            print(f"   Added {len(df)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. NSL-KDD Test
    print("\n[2] Processing NSL-KDD Test...")
    try:
        df = pd.read_parquet('KDDTest.parquet')
        label_col = None
        for col in df.columns:
            if col.lower() in ['label', 'target', 'class']:
                label_col = col
                break
        
        if label_col:
            df['is_attack'] = ~df[label_col].astype(str).str.contains('normal', case=False)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            selected_cols = numeric_cols[:30] + ['is_attack']
            datasets.append(df[selected_cols])
            print(f"   Added {len(df)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. UNSW-NB15 Train
    print("\n[3] Processing UNSW-NB15 Train...")
    try:
        df = pd.read_parquet('UNSW_NB15_training-set.parquet')
        if 'label' in df.columns:
            df['is_attack'] = df['label']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            selected_cols = numeric_cols[:30] + ['is_attack']
            datasets.append(df[selected_cols])
            print(f"   Added {len(df)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. UNSW-NB15 Test
    print("\n[4] Processing UNSW-NB15 Test...")
    try:
        df = pd.read_parquet('UNSW_NB15_testing-set.parquet')
        if 'label' in df.columns:
            df['is_attack'] = df['label']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            selected_cols = numeric_cols[:30] + ['is_attack']
            datasets.append(df[selected_cols])
            print(f"   Added {len(df)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. CIC-IDS2017 (sample)
    print("\n[5] Processing CIC-IDS2017...")
    try:
        df = pd.read_csv('cicids2017_cleaned.csv', nrows=50000)
        label_col = None
        for col in df.columns:
            if 'label' in col.lower() or 'attack' in col.lower():
                label_col = col
                break
        
        if label_col:
            df['is_attack'] = ~df[label_col].astype(str).str.contains('benign', case=False)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            selected_cols = numeric_cols[:30] + ['is_attack']
            datasets.append(df[selected_cols])
            print(f"   Added {len(df)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    if not datasets:
        print("\n✗ No datasets loaded!")
        return None, None
    
    # Combine datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    combined_df = combined_df.fillna(0)
    
    print(f"\nCombined dataset: {len(combined_df):,} samples")
    print(f"Attack distribution:")
    attack_count = combined_df['is_attack'].sum()
    normal_count = len(combined_df) - attack_count
    print(f"  Normal: {normal_count:,} ({normal_count/len(combined_df)*100:.1f}%)")
    print(f"  Attack: {attack_count:,} ({attack_count/len(combined_df)*100:.1f}%)")
    
    # Prepare for training
    X = combined_df.drop('is_attack', axis=1)
    y = combined_df['is_attack'].astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Train model
    print("\nTraining RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"               Normal  Attack")
    print(f"Actual Normal  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Attack  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Save model
    joblib.dump(model, 'models/basic_nids_model.pkl')
    print(f"\nModel saved to: models/basic_nids_model.pkl")
    
    # Save feature names for real-time detection
    feature_info = {
        'feature_names': X.columns.tolist(),
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    
    return model, X.columns.tolist()

class RealTimeNIDS:
    """Real-time Network Intrusion Detection System"""
    
    def __init__(self, model_path='models/basic_nids_model.pkl'):
        try:
            self.model = joblib.load(model_path)
            
            # Load feature info
            with open('models/feature_info.json', 'r') as f:
                feature_info = json.load(f)
                self.feature_names = feature_info['feature_names']
            
            print("✓ Real-time NIDS initialized!")
            print(f"✓ Model loaded with {len(self.feature_names)} features")
            
        except FileNotFoundError:
            print("✗ Model not found. Please train the model first.")
            print("Running training...")
            self.model, self.feature_names = train_basic_model()
            if self.model is None:
                raise Exception("Failed to train model")
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'normal_packets': 0,
            'attack_packets': 0,
            'start_time': time.time(),
            'alerts': []
        }
    
    def preprocess_packet(self, packet_data):
        """Preprocess a single packet"""
        # Convert to DataFrame with correct feature order
        df = pd.DataFrame([packet_data])
        
        # Ensure all expected features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    
    def predict_packet(self, packet_data):
        """Predict if packet is malicious"""
        self.stats['total_packets'] += 1
        
        try:
            # Preprocess
            X = self.preprocess_packet(packet_data)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            is_attack = bool(prediction)
            confidence = probability[1] if is_attack else probability[0]
            
            # Create result message
            if is_attack:
                result_message = f"🚨 ATTACK DETECTED! (Confidence: {confidence:.2%})"
            else:
                result_message = f"✅ Normal traffic (Confidence: {confidence:.2%})"
            
            # Update statistics
            if is_attack:
                self.stats['attack_packets'] += 1
                alert = {
                    'timestamp': time.time(),
                    'is_attack': True,
                    'confidence': confidence,
                    'message': result_message,
                    'src_bytes': packet_data.get('src_bytes', 0),
                    'dst_bytes': packet_data.get('dst_bytes', 0),
                    'duration': packet_data.get('duration', 0)
                }
                self.stats['alerts'].append(alert)
                
                # Keep only last 100 alerts
                if len(self.stats['alerts']) > 100:
                    self.stats['alerts'].pop(0)
                
                return {
                    'is_attack': True,
                    'confidence': confidence,
                    'message': result_message
                }
            else:
                self.stats['normal_packets'] += 1
                return {
                    'is_attack': False,
                    'confidence': confidence,
                    'message': result_message
                }
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'is_attack': False,
                'confidence': 0,
                'message': "⚠️  Error in prediction"
            }
    
    def get_statistics(self):
        """Get current statistics"""
        current_time = time.time()
        elapsed = current_time - self.stats['start_time']
        
        stats = {
            'total_packets': self.stats['total_packets'],
            'normal_packets': self.stats['normal_packets'],
            'attack_packets': self.stats['attack_packets'],
            'attack_rate': (self.stats['attack_packets'] / self.stats['total_packets'] * 100 
                          if self.stats['total_packets'] > 0 else 0),
            'packets_per_second': self.stats['total_packets'] / elapsed if elapsed > 0 else 0,
            'elapsed_time': elapsed,
            'recent_alerts': self.stats['alerts'][-10:]  # Last 10 alerts
        }
        
        return stats
    
    def print_dashboard(self, clear_screen=True):
        """Print real-time dashboard"""
        if clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        
        stats = self.get_statistics()
        
        print("="*70)
        print("REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
        print("="*70)
        print(f"Monitoring Duration: {stats['elapsed_time']:.1f} seconds")
        print(f"Total Packets: {stats['total_packets']:,}")
        print(f"Normal Packets: {stats['normal_packets']:,}")
        print(f"Attack Packets: {stats['attack_packets']:,}")
        print(f"Attack Rate: {stats['attack_rate']:.2f}%")
        print(f"Processing Speed: {stats['packets_per_second']:.1f} packets/sec")
        print("-"*70)
        
        if stats['recent_alerts']:
            print("RECENT ALERTS:")
            for alert in stats['recent_alerts']:
                time_str = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
                print(f"  [{time_str}] {alert['message']}")
                if 'src_bytes' in alert:
                    print(f"       Bytes: {alert['src_bytes']:,}/{alert['dst_bytes']:,}")
                    print(f"       Duration: {alert['duration']:.1f}s")
        else:
            print("No recent alerts")
        
        print("="*70)
        print("Press Ctrl+C to stop monitoring")

def simulate_network_traffic(nids, duration=60, packets_per_second=10):
    """Simulate network traffic for testing"""
    print(f"\nSimulating network traffic for {duration} seconds...")
    print(f"Rate: {packets_per_second} packets/second")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    packet_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Generate simulated packet data
            packet_data = {}
            
            # Use actual feature names from the model
            for feature in nids.feature_names[:20]:  # Use first 20 features
                if 'byte' in feature.lower():
                    packet_data[feature] = np.random.randint(100, 1000)
                elif 'count' in feature.lower():
                    packet_data[feature] = np.random.randint(1, 10)
                elif 'rate' in feature.lower():
                    packet_data[feature] = np.random.uniform(0, 0.1)
                elif 'duration' in feature.lower():
                    packet_data[feature] = np.random.uniform(0, 10)
                else:
                    packet_data[feature] = np.random.uniform(0, 1)
            
            # Occasionally simulate an attack (10% chance)
            if np.random.random() < 0.1:
                # Attack pattern: high bytes, high count
                for feature in nids.feature_names:
                    if 'byte' in feature.lower():
                        packet_data[feature] = np.random.randint(10000, 1000000)
                    elif 'count' in feature.lower():
                        packet_data[feature] = np.random.randint(100, 1000)
                    elif 'duration' in feature.lower():
                        packet_data[feature] = np.random.uniform(100, 1000)
            
            # Predict
            result = nids.predict_packet(packet_data)
            
            # Print attack alerts immediately
            if result['is_attack']:
                print(f"\n{result['message']}")
                # Show some feature values
                byte_features = [f for f in nids.feature_names if 'byte' in f.lower()]
                if byte_features:
                    print(f"  Feature values: {byte_features[0]}={packet_data.get(byte_features[0], 0):,}")
            
            packet_count += 1
            
            # Print dashboard every 50 packets
            if packet_count % 50 == 0:
                nids.print_dashboard(clear_screen=False)
                print(f"\nProcessed {packet_count} packets...")
            
            # Control packet rate
            time.sleep(1.0 / packets_per_second)
            
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    
    # Final statistics
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    stats = nids.get_statistics()
    print(f"Total packets simulated: {packet_count}")
    print(f"Normal packets: {stats['normal_packets']}")
    print(f"Attack packets: {stats['attack_packets']}")
    print(f"Attack rate: {stats['attack_rate']:.2f}%")
    print(f"Processing speed: {stats['packets_per_second']:.1f} packets/sec")
    
    return stats

def main():
    """Main function"""
    create_directories()
    
    print("="*70)
    print("NETWORK INTRUSION DETECTION SYSTEM")
    print("Complete Pipeline: Training + Real-time Detection")
    print("="*70)
    
    # Check if model exists
    model_exists = os.path.exists('models/basic_nids_model.pkl')
    feature_info_exists = os.path.exists('models/feature_info.json')
    
    if not model_exists or not feature_info_exists:
        print("No trained model found. Training new model...")
        model, features = train_basic_model()
        if model is None:
            print("Failed to train model. Exiting.")
            return
    else:
        print("Using existing trained model...")
        try:
            model = joblib.load('models/basic_nids_model.pkl')
            with open('models/feature_info.json', 'r') as f:
                feature_info = json.load(f)
                features = feature_info['feature_names']
            print(f"✓ Model loaded with {len(features)} features")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            model, features = train_basic_model()
            if model is None:
                print("Failed to train model. Exiting.")
                return
    
    # Initialize real-time NIDS
    print("\n" + "="*70)
    print("INITIALIZING REAL-TIME DETECTION")
    print("="*70)
    
    nids = RealTimeNIDS()
    
    # Start simulation
    print("\nStarting network traffic simulation...")
    
    while True:
        print("\nOptions:")
        print("1. Start traffic simulation (30 seconds)")
        print("2. Start traffic simulation (custom duration)")
        print("3. View current statistics")
        print("4. Test with single packet")
        print("5. Retrain model")
        print("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if choice == '1':
            simulate_network_traffic(nids, duration=30, packets_per_second=20)
        
        elif choice == '2':
            try:
                duration = int(input("Enter duration in seconds: "))
                if duration > 0:
                    simulate_network_traffic(nids, duration=duration, packets_per_second=20)
                else:
                    print("Duration must be positive")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or cancelled")
        
        elif choice == '3':
            stats = nids.get_statistics()
            print(f"\nCurrent Statistics:")
            print(f"  Total Packets: {stats['total_packets']:,}")
            print(f"  Normal Packets: {stats['normal_packets']:,}")
            print(f"  Attack Packets: {stats['attack_packets']:,}")
            print(f"  Attack Rate: {stats['attack_rate']:.2f}%")
            print(f"  Processing Speed: {stats['packets_per_second']:.1f} packets/sec")
            
            if stats['recent_alerts']:
                print(f"\nRecent Alerts (last {len(stats['recent_alerts'])}):")
                for alert in stats['recent_alerts'][-5:]:
                    time_str = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
                    print(f"  [{time_str}] {alert.get('message', 'Alert')}")
            else:
                print("\nNo recent alerts")
        
        elif choice == '4':
            # Create a test packet
            print("\nTesting with simulated packet...")
            test_packet = {}
            
            # Ask for packet type
            print("\nPacket type:")
            print("1. Normal packet")
            print("2. Attack packet (high traffic)")
            print("3. Custom values")
            
            try:
                packet_type = input("Enter choice (1-3): ").strip()
            except KeyboardInterrupt:
                print("\nCancelled")
                continue
            
            # Generate packet based on type
            if packet_type == '1':
                # Normal packet
                for feature in features[:10]:
                    if 'byte' in feature.lower():
                        test_packet[feature] = np.random.randint(100, 1000)
                    elif 'count' in feature.lower():
                        test_packet[feature] = np.random.randint(1, 10)
                    elif 'duration' in feature.lower():
                        test_packet[feature] = np.random.uniform(0.1, 5)
                    else:
                        test_packet[feature] = np.random.uniform(0, 1)
                print("Generated normal packet")
            
            elif packet_type == '2':
                # Attack packet
                for feature in features[:10]:
                    if 'byte' in feature.lower():
                        test_packet[feature] = np.random.randint(10000, 1000000)
                    elif 'count' in feature.lower():
                        test_packet[feature] = np.random.randint(100, 1000)
                    elif 'duration' in feature.lower():
                        test_packet[feature] = np.random.uniform(50, 500)
                    else:
                        test_packet[feature] = np.random.uniform(0, 1)
                print("Generated attack packet")
            
            elif packet_type == '3':
                # Custom values
                print(f"\nEnter values for features (or press Enter for default):")
                for feature in features[:5]:  # First 5 features
                    try:
                        value = input(f"  {feature}: ").strip()
                        if value:
                            test_packet[feature] = float(value)
                        else:
                            # Default based on feature name
                            if 'byte' in feature.lower():
                                test_packet[feature] = 1000
                            elif 'count' in feature.lower():
                                test_packet[feature] = 10
                            elif 'duration' in feature.lower():
                                test_packet[feature] = 5.0
                            else:
                                test_packet[feature] = 0.5
                    except (ValueError, KeyboardInterrupt):
                        print(f"Invalid value for {feature}, using default")
                        test_packet[feature] = 1000 if 'byte' in feature.lower() else 10
            
            else:
                print("Invalid choice, using normal packet")
                for feature in features[:5]:
                    test_packet[feature] = 100 if 'byte' in feature.lower() else 1
            
            # Fill remaining features with zeros
            for feature in features:
                if feature not in test_packet:
                    test_packet[feature] = 0
            
            # Predict
            result = nids.predict_packet(test_packet)
            print(f"\nTest Result: {result['message']}")
            
            # Show some feature values
            print("\nPacket features (first 5):")
            for i, feature in enumerate(features[:5]):
                print(f"  {feature}: {test_packet.get(feature, 0)}")
        
        elif choice == '5':
            print("\nRetraining model...")
            model, features = train_basic_model()
            if model is not None:
                # Reinitialize NIDS with new model
                nids = RealTimeNIDS()
                print("Model retrained and NIDS reinitialized!")
            else:
                print("Failed to retrain model")
        
        elif choice == '6':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()