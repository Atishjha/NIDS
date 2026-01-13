# working_realtime_detection.py - COMPLETELY FIXED VERSION
import joblib
import json
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class RealTimeNIDS:
    def __init__(self):
        print("🚀 Real-Time NIDS Initialization")
        print("=" * 50)
        
        # Load model
        model_path = 'models/nids_model.pkl'
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("Please run train_model.py first!")
            exit(1)
        
        try:
            self.model = joblib.load(model_path)
            print(f"✓ Model: {type(self.model).__name__}")
            
            # Check if model is actually trained
            if hasattr(self.model, 'classes_'):
                print(f"✓ Model classes: {self.model.classes_}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
        
        # Load feature names from feature_info.json
        self.feature_order = self.load_feature_names()
        print(f"✓ Loaded {len(self.feature_order)} features")
        
        # Verify with model
        self.verify_features()
        
        # Load label encoder
        self.label_encoder = self.load_label_encoder()
        
        # Load scaler if exists
        self.scaler = self.load_scaler()
        
        # Statistics
        self.packet_count = 0
        self.attack_count = 0
        
        # CRITICAL: Get actual class order from model
        self.attack_class_index = 0  # Default
        if hasattr(self.model, 'classes_'):
            classes = list(self.model.classes_)
            if 'Attack' in classes:
                self.attack_class_index = classes.index('Attack')
                print(f"✓ Attack class index: {self.attack_class_index}")
        
        print(f"\n✅ System ready!")
        print("=" * 50 + "\n")
    
    def load_feature_names(self):
        """Load feature names from feature_info.json"""
        if not os.path.exists('models/feature_info.json'):
            print("❌ feature_info.json not found!")
            return []
        
        with open('models/feature_info.json', 'r') as f:
            data = json.load(f)
        
        # Try different possible keys
        feature_names = []
        for key in ['feature_names', 'features', 'columns', 'X_columns']:
            if key in data:
                feature_names = data[key]
                print(f"Using '{key}' for feature names")
                break
        
        if not feature_names and hasattr(self.model, 'feature_names_in_'):
            feature_names = list(self.model.feature_names_in_)
            print("Using model's feature_names_in_")
        
        return feature_names
    
    def verify_features(self):
        """Verify features match model expectations"""
        if hasattr(self.model, 'feature_names_in_'):
            model_features = list(self.model.feature_names_in_)
            print(f"Model expects {len(model_features)} features")
            
            if len(self.feature_order) != len(model_features):
                print(f"⚠️ Warning: Feature count mismatch!")
                print(f"  Config has {len(self.feature_order)} features")
                print(f"  Model expects {len(model_features)} features")
                
                # Use model's features as they are correct
                self.feature_order = model_features
                print("  Using model's feature names")
                
        elif hasattr(self.model, 'n_features_in_'):
            print(f"Model expects {self.model.n_features_in_} features")
            if len(self.feature_order) != self.model.n_features_in_:
                print(f"⚠️ Warning: Feature count mismatch!")
                print(f"  Config has {len(self.feature_order)} features")
                print(f"  Model expects {self.model.n_features_in_} features")
    
    def load_label_encoder(self):
        """Load or create label encoder"""
        encoder_path = 'models/label_encoder.pkl'
        
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            print(f"✓ Label encoder loaded")
            classes = list(encoder.classes_)
            print(f"  Classes: {classes}")
            
            # Verify Attack is in classes
            if 'Attack' not in classes:
                print("⚠️ Warning: 'Attack' not in encoder classes!")
                print(f"  Available: {classes}")
            
            return encoder
        else:
            print(f"⚠️ {encoder_path} not found. Creating basic encoder.")
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            encoder.fit(['Normal', 'Attack'])  # Note: Normal first!
            
            # Save it
            joblib.dump(encoder, encoder_path)
            print(f"✓ Created and saved basic label encoder")
            print(f"  Classes: {list(encoder.classes_)}")
            return encoder
    
    def load_scaler(self):
        """Load scaler if available"""
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded")
            return scaler
        else:
            print("⚠️ Scaler not found. Will use raw features.")
            return None
    
    def create_realistic_packet(self, is_attack=False, attack_type=None):
        """Create realistic network packet"""
        packet = {feature: 0.0 for feature in self.feature_order}
        
        # Get baseline values for normal traffic
        if not is_attack:
            # NORMAL TRAFFIC - small values
            for feature in self.feature_order:
                if any(word in feature.lower() for word in ['byte', 'bytes', 'length']):
                    packet[feature] = random.randint(64, 1500)
                elif any(word in feature.lower() for word in ['duration', 'time']):
                    packet[feature] = random.uniform(0.01, 2.0)
                elif any(word in feature.lower() for word in ['rate', 'load']):
                    packet[feature] = random.uniform(0.1, 100.0)
                elif any(word in feature.lower() for word in ['count', 'num']):
                    packet[feature] = random.randint(1, 10)
                elif any(word in feature.lower() for word in ['port']):
                    packet[feature] = random.randint(1024, 65535)
                else:
                    packet[feature] = random.uniform(0, 10)
        else:
            # ATTACK TRAFFIC - based on actual attack patterns
            attack_type = attack_type or random.choice(['ddos', 'portscan', 'bruteforce'])
            
            # Start with base normal values
            for feature in self.feature_order:
                packet[feature] = random.uniform(0, 10)
            
            # Apply attack-specific patterns based on real CIC-IDS2017/CIC-IDS2018 patterns
            if attack_type == 'ddos':
                # DDoS: High volume, high rates
                for feature in self.feature_order:
                    feature_lower = feature.lower()
                    if 'src_bytes' in feature_lower or 'sbytes' in feature_lower:
                        packet[feature] = random.randint(100000, 1000000)
                    elif 'dst_bytes' in feature_lower or 'dbytes' in feature_lower:
                        packet[feature] = random.randint(100000, 1000000)
                    elif 'dload' in feature_lower:
                        packet[feature] = random.randint(500000, 5000000)
                    elif 'sload' in feature_lower:
                        packet[feature] = random.randint(500000, 5000000)
                    elif 'rate' in feature_lower:
                        packet[feature] = random.uniform(1000, 10000)
                    elif 'count' in feature_lower:
                        packet[feature] = random.randint(1000, 10000)
            
            elif attack_type == 'portscan':
                # Port scan: Many connections, short duration
                for feature in self.feature_order:
                    feature_lower = feature.lower()
                    if 'duration' in feature_lower:
                        packet[feature] = random.uniform(0.001, 0.01)
                    elif 'count' in feature_lower:
                        packet[feature] = random.randint(100, 1000)
                    elif 'same_srv_rate' in feature_lower:
                        packet[feature] = 0.0
                    elif 'dst_host_same_srv_rate' in feature_lower:
                        packet[feature] = 0.0
            
            elif attack_type == 'bruteforce':
                # Brute force: Many failed logins
                for feature in self.feature_order:
                    feature_lower = feature.lower()
                    if 'wrong_fragment' in feature_lower:
                        packet[feature] = 1.0
                    elif 'num_failed_logins' in feature_lower:
                        packet[feature] = random.randint(10, 100)
                    elif 'logged_in' in feature_lower:
                        packet[feature] = 0.0
                    elif 'root_shell' in feature_lower:
                        packet[feature] = 1.0
                    elif 'su_attempted' in feature_lower:
                        packet[feature] = 1.0
        
        return packet
    
    def extract_features(self, packet):
        """Extract features in correct order"""
        features = []
        
        for feature_name in self.feature_order:
            # Get value from packet, default to 0 if not found
            value = packet.get(feature_name, 0)
            
            # Convert to float
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, packet):
        """Make prediction on packet"""
        try:
            # Extract features
            X = self.extract_features(packet)
            
            # Apply scaler if available
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                
                # DEBUG: Check probabilities
                print(f"DEBUG: Raw probabilities = {proba}")
                print(f"DEBUG: Model classes = {self.model.classes_}")
                
                # FIXED: Determine which index corresponds to Attack
                if hasattr(self.model, 'classes_'):
                    classes = list(self.model.classes_)
                    if 'Attack' in classes:
                        attack_index = list(classes).index('Attack')
                        attack_prob = proba[attack_index]
                        normal_index = list(classes).index('Normal') if 'Normal' in classes else 1 - attack_index
                        normal_prob = proba[normal_index]
                    else:
                        # Fallback: assume first class is Attack
                        attack_prob = proba[0]
                        normal_prob = proba[1] if len(proba) > 1 else 1 - attack_prob
                else:
                    # Fallback: assume Attack is class 0
                    attack_prob = proba[0]
                    normal_prob = proba[1] if len(proba) > 1 else 1 - attack_prob
                
                print(f"DEBUG: attack_prob = {attack_prob:.2%}")
                print(f"DEBUG: normal_prob = {normal_prob:.2%}")
                
                # Apply threshold - CRITICAL FIX: Use dynamic threshold
                THRESHOLD = 0.55  # Lower threshold for better detection
                
                if attack_prob > THRESHOLD:
                    label = 'Attack'
                    confidence = attack_prob
                else:
                    label = 'Normal'
                    confidence = normal_prob
                
                print(f"DEBUG: Decision = {label} (threshold={THRESHOLD})")
                
                return label, confidence, attack_prob
            else:
                # Fallback for models without predict_proba
                prediction = self.model.predict(X)[0]
                label = self.label_encoder.inverse_transform([prediction])[0]
                attack_prob = 1.0 if label == 'Attack' else 0.0
                return label, 1.0, attack_prob
                
        except Exception as e:
            print(f"❌ Prediction error: {str(e)[:100]}...")
            return 'Error', 0.0, 0.0
    
    def create_super_attack(self):
        """Create an OBVIOUS attack packet for testing"""
        print("💥 CREATING SUPER OBVIOUS ATTACK PACKET")
        
        # Start with all zeros
        packet = {feature: 0.0 for feature in self.feature_order}
        
        # Set EXTREME values based on typical attack features
        # Using actual important features from CIC datasets
        extreme_values = {
            'src_bytes': 10000000,
            'dst_bytes': 10000000,
            'dload': 5000000.0,
            'sload': 5000000.0,
            'rate': 100000.0,
            'dbytes': 10000000,
            'sbytes': 10000000,
            'count': 10000,
            'same_srv_rate': 0.0,
            'dst_host_same_srv_rate': 0.0,
            'dst_host_diff_srv_rate': 1.0,
            'dst_host_same_src_port_rate': 0.0,
            'dst_host_serror_rate': 1.0,
            'dst_host_srv_serror_rate': 1.0,
            'wrong_fragment': 1.0,
            'urgent': 1.0,
            'hot': 1.0,
            'num_failed_logins': 50,
            'logged_in': 0.0,
            'root_shell': 1.0,
            'su_attempted': 1.0,
            'num_root': 100,
            'num_file_creations': 100,
            'num_shells': 10,
            'num_access_files': 100,
            'num_outbound_cmds': 100,
            'is_host_login': 1.0,
            'is_guest_login': 1.0
        }
        
        # Apply extreme values
        for feature, value in extreme_values.items():
            # Check if feature exists in our feature list (case-insensitive)
            for f in self.feature_order:
                if f.lower() == feature.lower():
                    packet[f] = value
                    break
        
        print(f"  src_bytes: {packet.get('src_bytes', 0):,} (HUGE!)")
        print(f"  dst_bytes: {packet.get('dst_bytes', 0):,} (HUGE!)")
        print(f"  dload: {packet.get('dload', 0):,.0f} (EXTREME!)")
        
        return packet
    
    def log_packet(self, src_ip, dst_ip, protocol, label, confidence, attack_prob, expected):
        """Log and display packet"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        status = "🚨 ATTACK" if label == 'Attack' else "✅ Normal"
        expected_str = f"({expected})"
        
        # Color coding based on correctness
        if (expected == 'Attack' and label == 'Attack') or (expected == 'Normal' and label == 'Normal'):
            correct = "✓"
        else:
            correct = "✗"
        
        print(f"{timestamp} | Pkt #{self.packet_count:04d} {correct} | "
              f"{src_ip:15} → {dst_ip:15} | {protocol:6} | "
              f"{status:10} {expected_str:9} | "
              f"Conf: {confidence:.1%} | AttackProb: {attack_prob:.1%}")
        
        # Update statistics
        self.packet_count += 1
        if label == 'Attack':
            self.attack_count += 1
        
        return {
            'timestamp': timestamp,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'prediction': label,
            'confidence': confidence,
            'attack_probability': attack_prob,
            'expected': expected
        }
    
    def show_stats(self):
        """Show current statistics"""
        if self.packet_count > 0:
            attack_rate = (self.attack_count / self.packet_count) * 100
            print(f"\n📊 Stats: {self.packet_count} packets | "
                  f"{self.attack_count} attacks ({attack_rate:.1f}%)")
    
    def simulate_traffic(self, duration=60, attack_probability=0.3):
        """Simulate network traffic"""
        print(f"🎬 Starting Simulation")
        print(f"   Duration: {duration} seconds")
        print(f"   Attack Probability: {attack_probability:.0%}")
        print("=" * 70)
        
        # Network configuration
        src_ips = ['192.168.1.' + str(i) for i in range(1, 50)]
        dst_ips = ['10.0.0.' + str(i) for i in range(1, 20)] + ['8.8.8.8', '1.1.1.1']
        protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS']
        
        attack_types = ['ddos', 'portscan', 'bruteforce']
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Decide if this is an attack
                is_attack = random.random() < attack_probability
                attack_type = random.choice(attack_types) if is_attack else None
                
                # Create packet
                packet = self.create_realistic_packet(is_attack, attack_type)
                
                # Generate network info
                src_ip = random.choice(src_ips)
                dst_ip = random.choice(dst_ips)
                protocol = random.choice(protocols)
                
                # Make prediction
                label, confidence, attack_prob = self.predict(packet)
                
                # Log
                expected = 'Attack' if is_attack else 'Normal'
                self.log_packet(src_ip, dst_ip, protocol, label, confidence, attack_prob, expected)
                
                # Show stats every 10 packets
                if self.packet_count % 10 == 0:
                    self.show_stats()
                
                time.sleep(0.2)  # Simulate real-time delay
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Simulation stopped by user")
        
        finally:
            self.final_report()
    
    def final_report(self):
        """Show final report"""
        print("\n" + "=" * 70)
        print("📈 FINAL REPORT")
        print("=" * 70)
        
        if self.packet_count > 0:
            attack_rate = (self.attack_count / self.packet_count) * 100
            
            print(f"Total Packets Processed: {self.packet_count}")
            print(f"Attack Detections: {self.attack_count} ({attack_rate:.1f}%)")
            
            # Analysis
            if attack_rate < 5:
                print("\n⚠️  ANALYSIS: Low Attack Detection Rate")
                print("  Try lowering threshold to 0.10 in predict() method")
            elif attack_rate > 70:
                print("\n⚠️  ANALYSIS: High False Positive Rate")
                print("  Try increasing threshold to 0.30 in predict() method")
            else:
                print("\n✅ ANALYSIS: Reasonable detection rate")
            
        print("=" * 70)


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("       ATTACK DETECTION DIAGNOSTIC (FIXED VERSION)")
    print("=" * 70)
    
    nids = RealTimeNIDS()
    
    # Test 1: Super obvious attack
    print("\n" + "=" * 70)
    print("💥 TEST 1: Super Obvious Attack")
    print("=" * 70)
    
    super_attack = nids.create_super_attack()
    label, conf, attack_prob = nids.predict(super_attack)
    
    if label == 'Attack':
        print(f"✅ SUCCESS! Model detected attack with {attack_prob:.1%} probability")
    else:
        print(f"❌ FAILURE! Model said '{label}' with only {attack_prob:.1%} attack probability")
    
    # Test 2: Regular attack patterns
    print("\n" + "=" * 70)
    print("💥 TEST 2: Regular Attack Patterns")
    print("=" * 70)
    
    attack_types = ['ddos', 'portscan', 'bruteforce']
    results = []
    
    for attack_type in attack_types:
        print(f"\n{attack_type.upper()}:")
        attack_packet = nids.create_realistic_packet(is_attack=True, attack_type=attack_type)
        label, conf, attack_prob = nids.predict(attack_packet)
        
        results.append((attack_type, label, attack_prob))
        
        if label == 'Attack':
            print(f"  ✅ Detected! Attack probability: {attack_prob:.1%}")
        else:
            print(f"  ❌ Missed! Only {attack_prob:.1%} attack probability")
    
    # Test 3: Normal traffic
    print("\n" + "=" * 70)
    print("✅ TEST 3: Normal Traffic (should be Normal)")
    print("=" * 70)
    
    normal_packet = nids.create_realistic_packet(is_attack=False)
    label, conf, attack_prob = nids.predict(normal_packet)
    
    if label == 'Normal':
        print(f"✅ CORRECT! Normal traffic correctly identified")
        print(f"   Attack probability: {attack_prob:.1%} (should be low)")
    else:
        print(f"❌ FALSE POSITIVE! Normal traffic labeled as {label}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    super_success = "✅" if label == 'Attack' else "❌"
    attack_detections = sum(1 for _, label, _ in results if label == 'Attack')
    
    print(f"Super Attack Test: {super_success}")
    print(f"Regular Attacks Detected: {attack_detections}/3")
    print(f"Normal Traffic: {'✅ Correct' if label == 'Normal' else '❌ Wrong'}")
    
    if attack_detections == 0:
        print("\n🚨 URGENT ISSUE: Model detects NO attacks!")
        print("\n🔧 QUICK FIXES:")
        print("1. Lower threshold to 0.10 in predict() method")
        print("2. Check if model was trained properly")
        print("3. Verify feature_info.json matches training data")
        print("4. Consider retraining with more attack samples")
    
    elif attack_detections < 2:
        print("\n⚠️  WARNING: Model detects few attacks")
        print("Try lowering threshold to 0.10 or 0.15")
        
    else:
        print("\n✅ GOOD: Model detects attacks!")
        
        # Ask user if they want to run simulation
        response = input("\nRun full simulation? (y/n): ")
        if response.lower() == 'y':
            print("\n" + "=" * 70)
            print("       STARTING FULL SIMULATION")
            print("=" * 70)
            nids.simulate_traffic(duration=60, attack_probability=0.3)
