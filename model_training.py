# train_model.py - FIXED VERSION (JSON serializable)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')
import time
import os
import json
from collections import Counter

def create_synthetic_dataset():
    """Create a proper synthetic dataset with clear attack patterns"""
    print("📊 Creating synthetic dataset with clear attack patterns...")
    
    n_samples = 50000  # 50K samples
    n_features = 90
    
    # Feature names based on CIC-IDS2017 dataset
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    # Add more features to reach 90
    for i in range(len(feature_names), n_features):
        feature_names.append(f'feature_{i}')
    
    # Create DataFrame
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
    
    # Create clear patterns for attacks vs normal
    y = []
    attack_indices = []  # Store which samples are attacks
    
    print("🔍 Creating labels with clear patterns...")
    for i in range(n_samples):
        is_attack = np.random.rand() < 0.3  # 30% attacks
        
        if is_attack:
            # ATTACK PATTERNS - make specific features extreme
            # DDoS-like attacks
            if np.random.rand() < 0.3:  # 30% of attacks are DDoS
                X.loc[i, 'src_bytes'] = np.random.uniform(100000, 1000000)
                X.loc[i, 'dst_bytes'] = np.random.uniform(100000, 1000000)
                X.loc[i, 'count'] = np.random.uniform(100, 1000)
                X.loc[i, 'srv_count'] = np.random.uniform(100, 1000)
                X.loc[i, 'duration'] = np.random.uniform(0.001, 0.01)  # Very short
                attack_type = 'ddos'
            
            # Port scan attacks
            elif np.random.rand() < 0.5:  # 50% of attacks are port scans
                X.loc[i, 'same_srv_rate'] = 0.0
                X.loc[i, 'diff_srv_rate'] = 1.0
                X.loc[i, 'dst_host_same_srv_rate'] = 0.0
                X.loc[i, 'dst_host_diff_srv_rate'] = 1.0
                X.loc[i, 'duration'] = np.random.uniform(0.001, 0.01)
                attack_type = 'portscan'
            
            # Brute force attacks
            else:  # 20% of attacks are brute force
                X.loc[i, 'num_failed_logins'] = np.random.randint(5, 100)
                X.loc[i, 'num_root'] = 1 if np.random.rand() < 0.3 else 0
                X.loc[i, 'root_shell'] = 1 if np.random.rand() < 0.2 else 0
                X.loc[i, 'su_attempted'] = 1
                attack_type = 'bruteforce'
            
            attack_indices.append(i)
            y.append('Attack')
            
        else:
            # NORMAL TRAFFIC - reasonable values
            X.loc[i, 'src_bytes'] = np.random.uniform(64, 1500)
            X.loc[i, 'dst_bytes'] = np.random.uniform(64, 1500)
            X.loc[i, 'duration'] = np.random.uniform(0.1, 10.0)
            X.loc[i, 'count'] = np.random.uniform(1, 10)
            X.loc[i, 'srv_count'] = np.random.uniform(1, 10)
            X.loc[i, 'same_srv_rate'] = np.random.uniform(0.7, 1.0)
            y.append('Normal')
    
    y = np.array(y)
    
    print(f"✅ Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")
    
    # FIX: Convert numpy types to Python native types for JSON serialization
    class_distribution = {}
    for cls, cnt in zip(unique, counts):
        class_distribution[str(cls)] = int(cnt)  # Convert to Python int
    
    # Save feature names
    os.makedirs('models', exist_ok=True)
    with open('models/feature_info.json', 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'n_features': int(n_features),  # Convert to Python int
            'class_distribution': class_distribution,
            'attack_indices_count': int(len(attack_indices))  # Convert to Python int
        }, f, indent=2)
    
    return X, y, feature_names

def convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj

def train_model():
    """Train a properly working RandomForest model"""
    print("=" * 70)
    print("🤖 NIDS MODEL TRAINING (FIXED VERSION)")
    print("=" * 70)
    
    # Create or load dataset
    data_path = 'data/processed_dataset.csv'
    
    if os.path.exists(data_path):
        print("📁 Loading existing dataset...")
        df = pd.read_csv(data_path)
        X = df.drop('label', axis=1)
        y = df['label']
        feature_names = X.columns.tolist()
    else:
        print("📊 Creating synthetic dataset...")
        X, y, feature_names = create_synthetic_dataset()
        
        # Save dataset
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✅ Dataset saved to {data_path}")
    
    # Encode labels
    print("\n🔤 Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classes: {label_encoder.classes_}")
    class_mapping = {str(cls): int(idx) for cls, idx in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    print(f"Class mapping: {class_mapping}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")
    
    # Split data
    print("\n✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.3, 
        stratify=y_encoded,
        random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    print("\n📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest with proper settings
    print("\n🌳 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,          # More trees for better accuracy
        max_depth=20,              # Deeper trees to learn complex patterns
        min_samples_split=5,       # Prevent overfitting
        min_samples_leaf=2,        # Prevent overfitting
        max_features='sqrt',       # Use sqrt features for diversity
        bootstrap=True,            # Use bootstrap samples
        oob_score=True,            # Calculate out-of-bag score
        random_state=42,
        n_jobs=-1,                 # Use all CPU cores
        verbose=1                  # Show progress
    )
    
    # Train model
    start_time = time.time()
    print("Training started... (this may take a few minutes)")
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"OOB Score: {model.oob_score_:.4f}")
    
    # Make predictions
    print("\n📊 Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate model
    print("\n📈 Evaluation Results:")
    print("=" * 50)
    
    # Decode labels for reporting
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    # Confusion matrix
    print("\n🎯 Confusion Matrix:")
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    cm_df = pd.DataFrame(cm, 
                         index=[f'True {cls}' for cls in label_encoder.classes_], 
                         columns=[f'Pred {cls}' for cls in label_encoder.classes_])
    print(cm_df)
    
    # Calculate probabilities for ROC-AUC
    print("\n📊 ROC-AUC Score:")
    try:
        if len(label_encoder.classes_) == 2:
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        print(f"AUC Score: {auc_score:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC score: {e}")
    
    # Feature importance
    print("\n🔝 Top 20 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string())
    
    # Cross-validation
    print("\n🔄 Cross-Validation:")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test a few samples to verify model works
    print("\n🧪 Model Verification Test:")
    print("=" * 50)
    
    # Create test samples
    test_samples = []
    
    # Sample 1: Normal traffic
    normal_sample = np.zeros(len(feature_names))
    if 'src_bytes' in feature_names:
        normal_sample[feature_names.index('src_bytes')] = 1000
    if 'dst_bytes' in feature_names:
        normal_sample[feature_names.index('dst_bytes')] = 1000
    if 'duration' in feature_names:
        normal_sample[feature_names.index('duration')] = 1.5
    if 'count' in feature_names:
        normal_sample[feature_names.index('count')] = 5
    test_samples.append(('Normal Sample', normal_sample))
    
    # Sample 2: DDoS attack
    ddos_sample = np.zeros(len(feature_names))
    if 'src_bytes' in feature_names:
        ddos_sample[feature_names.index('src_bytes')] = 1000000
    if 'dst_bytes' in feature_names:
        ddos_sample[feature_names.index('dst_bytes')] = 1000000
    if 'duration' in feature_names:
        ddos_sample[feature_names.index('duration')] = 0.001
    if 'count' in feature_names:
        ddos_sample[feature_names.index('count')] = 1000
    test_samples.append(('DDoS Attack', ddos_sample))
    
    # Sample 3: Port scan
    portscan_sample = np.zeros(len(feature_names))
    if 'same_srv_rate' in feature_names:
        portscan_sample[feature_names.index('same_srv_rate')] = 0.0
    if 'diff_srv_rate' in feature_names:
        portscan_sample[feature_names.index('diff_srv_rate')] = 1.0
    if 'duration' in feature_names:
        portscan_sample[feature_names.index('duration')] = 0.001
    if 'dst_host_same_srv_rate' in feature_names:
        portscan_sample[feature_names.index('dst_host_same_srv_rate')] = 0.0
    test_samples.append(('Port Scan', portscan_sample))
    
    for name, sample in test_samples:
        sample_scaled = scaler.transform([sample])
        proba = model.predict_proba(sample_scaled)[0]
        pred_class = model.predict(sample_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        
        print(f"\n{name}:")
        print(f"  Predicted: {pred_label}")
        print(f"  Probabilities: {proba}")
        print(f"  Attack probability: {proba[1 if 'Attack' in label_encoder.classes_ and pred_label == 'Attack' else 0]:.2%}")
    
    # Save everything
    print("\n💾 Saving model and artifacts...")
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/nids_model.pkl')
    print("✓ Model saved as models/nids_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✓ Scaler saved as models/scaler.pkl")
    
    # Save label encoder
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    print("✓ Label encoder saved as models/label_encoder.pkl")
    
    # Update feature info with importance
    # Convert numpy types to Python types for JSON
    feature_importance_dict = {}
    for feature, importance in zip(feature_names, model.feature_importances_):
        feature_importance_dict[feature] = float(importance)  # Convert to Python float
    
    feature_info = {
        'feature_names': feature_names,
        'n_features': int(len(feature_names)),
        'feature_importance': feature_importance_dict,
        'classes': label_encoder.classes_.tolist(),
        'class_mapping': class_mapping,
        'model_type': 'RandomForestClassifier',
        'model_params': convert_to_python_types(model.get_params())
    }
    
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print("✓ Feature info saved as models/feature_info.json")
    
    # Save training results
    results = {
        'training_time': float(training_time),
        'oob_score': float(model.oob_score_),
        'test_accuracy': float(model.score(X_test_scaled, y_test)),
        'cv_scores': convert_to_python_types(cv_scores),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'classification_report': convert_to_python_types(classification_report(y_test_decoded, y_pred_decoded, output_dict=True)),
        'confusion_matrix': convert_to_python_types(cm)
    }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Training results saved as models/training_results.json")
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📋 Model Summary:")
    print(f"  • Model type: RandomForestClassifier")
    print(f"  • Training time: {training_time:.2f}s")
    print(f"  • OOB Score: {model.oob_score_:.4f}")
    print(f"  • Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")
    print(f"  • CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  • Number of features: {len(feature_names)}")
    print(f"  • Classes: {list(label_encoder.classes_)}")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    # Clean up old models first
    print("🧹 Cleaning up old models...")
    model_files = ['nids_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'feature_info.json', 'training_results.json']
    for file in model_files:
        filepath = os.path.join('models', file)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed: {filepath}")
    
    # Train the model
    try:
        model, scaler, label_encoder = train_model()
        
        # Quick verification
        print("\n🔍 Quick Verification:")
        print("=" * 50)
        
        # Load and test the model
        loaded_model = joblib.load('models/nids_model.pkl')
        loaded_scaler = joblib.load('models/scaler.pkl')
        loaded_encoder = joblib.load('models/label_encoder.pkl')
        
        # Create a simple test
        with open('models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        # Test 1: Normal traffic
        normal_features = np.zeros(len(feature_info['feature_names']))
        if 'src_bytes' in feature_info['feature_names']:
            normal_features[feature_info['feature_names'].index('src_bytes')] = 1000
        if 'dst_bytes' in feature_info['feature_names']:
            normal_features[feature_info['feature_names'].index('dst_bytes')] = 1000
        
        # Test 2: Attack traffic
        attack_features = np.zeros(len(feature_info['feature_names']))
        if 'src_bytes' in feature_info['feature_names']:
            attack_features[feature_info['feature_names'].index('src_bytes')] = 1000000
        if 'dst_bytes' in feature_info['feature_names']:
            attack_features[feature_info['feature_names'].index('dst_bytes')] = 1000000
        
        for name, features in [('Normal', normal_features), ('Attack', attack_features)]:
            features_scaled = loaded_scaler.transform([features])
            proba = loaded_model.predict_proba(features_scaled)[0]
            pred = loaded_model.predict(features_scaled)[0]
            pred_label = loaded_encoder.inverse_transform([pred])[0]
            
            print(f"\n{name} traffic:")
            print(f"  Predicted: {pred_label}")
            if 'Attack' in loaded_encoder.classes_:
                attack_idx = list(loaded_encoder.classes_).index('Attack')
                print(f"  Attack probability: {proba[attack_idx]:.2%}")
                print(f"  Normal probability: {proba[1 - attack_idx]:.2%}")
        
        print("\n✅ Model is ready for real-time detection!")
        print("Run: python realtime_detection.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Please check the error above and try again.")
