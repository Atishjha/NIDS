import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def advanced_nids():
    """Advanced NIDS with multi-class classification"""
    print("="*70)
    print("ADVANCED NETWORK INTRUSION DETECTION SYSTEM")
    print("Multi-Class Classification")
    print("="*70)
    
    start_time = time.time()
    
    # Create directories
    for dir_name in ['models', 'results', 'logs', 'plots']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Dictionary to store all datasets
    all_datasets = {}
    
    # 1. Process NSL-KDD
    print("\n[1] Processing NSL-KDD datasets...")
    try:
        kdd_train = pd.read_parquet('KDDTrain.parquet')
        kdd_test = pd.read_parquet('KDDTest.parquet')
        
        # Find label column
        label_col = None
        for col in kdd_train.columns:
            if col.lower() in ['label', 'target', 'class']:
                label_col = col
                break
        
        if label_col:
            # Map to attack categories
            def map_nsl_kdd_label(label):
                label = str(label).lower()
                
                # Normal traffic
                if 'normal' in label:
                    return 'Normal'
                
                # DoS attacks
                dos_attacks = ['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land', 
                              'apache2', 'udpstorm', 'processtable', 'worm']
                if any(attack in label for attack in dos_attacks):
                    return 'DoS'
                
                # Probe attacks
                probe_attacks = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
                if any(attack in label for attack in probe_attacks):
                    return 'Probe'
                
                # R2L attacks
                r2l_attacks = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 
                              'warezmaster', 'warezclient', 'spy', 'named', 'snmpguess']
                if any(attack in label for attack in r2l_attacks):
                    return 'R2L'
                
                # U2R attacks
                u2r_attacks = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 
                              'sqlattack', 'xterm', 'ps', 'httptunnel']
                if any(attack in label for attack in u2r_attacks):
                    return 'U2R'
                
                # Default to Other
                return 'Other'
            
            kdd_train['attack_type'] = kdd_train[label_col].apply(map_nsl_kdd_label)
            kdd_test['attack_type'] = kdd_test[label_col].apply(map_nsl_kdd_label)
            
            # Select numeric features
            numeric_cols = kdd_train.select_dtypes(include=[np.number]).columns.tolist()
            if 'attack_type' not in kdd_train.columns:
                numeric_cols = numeric_cols[:50]  # Limit features
            
            kdd_train = kdd_train[numeric_cols + ['attack_type']]
            kdd_test = kdd_test[numeric_cols + ['attack_type']]
            
            all_datasets['nsl_kdd_train'] = kdd_train
            all_datasets['nsl_kdd_test'] = kdd_test
            
            print(f"   Processed: {len(kdd_train):,} training samples")
            print(f"   Attack distribution: {kdd_train['attack_type'].value_counts().to_dict()}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Process UNSW-NB15
    print("\n[2] Processing UNSW-NB15 datasets...")
    try:
        unsw_train = pd.read_parquet('UNSW_NB15_training-set.parquet')
        unsw_test = pd.read_parquet('UNSW_NB15_testing-set.parquet')
        
        # UNSW already has attack categories
        if 'attack_cat' in unsw_train.columns:
            # Rename categories for consistency
            attack_mapping = {
                'Normal': 'Normal',
                'Reconnaissance': 'Probe',
                'Exploits': 'Exploit',
                'Generic': 'Malware',
                'Shellcode': 'Malware',
                'Worms': 'Malware',
                'Backdoor': 'Backdoor',
                'Analysis': 'Analysis',
                'Fuzzers': 'Fuzzer'
            }
            
            unsw_train['attack_type'] = unsw_train['attack_cat'].map(attack_mapping)
            unsw_test['attack_type'] = unsw_test['attack_cat'].map(attack_mapping)
            
            # Fill NaN values (categories not in mapping)
            unsw_train['attack_type'] = unsw_train['attack_type'].fillna('Other')
            unsw_test['attack_type'] = unsw_test['attack_type'].fillna('Other')
            
            # Select numeric features
            numeric_cols = unsw_train.select_dtypes(include=[np.number]).columns.tolist()
            # Remove label columns
            for col in ['label', 'id']:
                if col in numeric_cols:
                    numeric_cols.remove(col)
            
            numeric_cols = numeric_cols[:50]  # Limit features
            
            unsw_train = unsw_train[numeric_cols + ['attack_type']]
            unsw_test = unsw_test[numeric_cols + ['attack_type']]
            
            all_datasets['unsw_train'] = unsw_train
            all_datasets['unsw_test'] = unsw_test
            
            print(f"   Processed: {len(unsw_train):,} training samples")
            print(f"   Attack distribution: {unsw_train['attack_type'].value_counts().to_dict()}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Process CIC-IDS2017
    print("\n[3] Processing CIC-IDS2017...")
    try:
        # Load subset for speed
        cicids = pd.read_csv('cicids2017_cleaned.csv', nrows=200000)
        
        # Find label column
        label_col = None
        for col in cicids.columns:
            if 'label' in col.lower() or 'attack' in col.lower():
                label_col = col
                break
        
        if label_col:
            # Map CIC-IDS labels to categories
            def map_cicids_label(label):
                label = str(label).lower()
                
                if 'benign' in label:
                    return 'Normal'
                
                # DoS/DDoS
                if any(x in label for x in ['dos', 'ddos', 'hulk', 'slowloris', 'goldeneye']):
                    return 'DoS'
                
                # Probe/Scanning
                if any(x in label for x in ['portscan', 'scan']):
                    return 'Probe'
                
                # Brute Force
                if any(x in label for x in ['patator', 'brute']):
                    return 'BruteForce'
                
                # Web attacks
                if any(x in label for x in ['xss', 'sql', 'web']):
                    return 'WebAttack'
                
                # Botnet
                if 'bot' in label:
                    return 'Botnet'
                
                # Infiltration
                if 'infiltration' in label:
                    return 'Infiltration'
                
                return 'Other'
            
            cicids['attack_type'] = cicids[label_col].apply(map_cicids_label)
            
            # Select numeric features
            numeric_cols = cicids.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != label_col]
            numeric_cols = numeric_cols[:50]  # Limit features
            
            cicids = cicids[numeric_cols + ['attack_type']]
            all_datasets['cicids2017'] = cicids
            
            print(f"   Processed: {len(cicids):,} samples")
            print(f"   Attack distribution: {cicids['attack_type'].value_counts().to_dict()}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Combine all datasets
    if not all_datasets:
        print("\n✗ No datasets were processed successfully!")
        return
    
    print(f"\n{'='*70}")
    print("COMBINING ALL DATASETS")
    print('='*70)
    
    # Align features across datasets
    all_features = set()
    for df in all_datasets.values():
        all_features.update(df.columns)
    
    all_features = list(all_features)
    if 'attack_type' in all_features:
        all_features.remove('attack_type')
    
    # Create combined dataset with common features
    combined_data = []
    for name, df in all_datasets.items():
        df_copy = df.copy()
        # Add missing columns with zeros
        for feature in all_features:
            if feature not in df_copy.columns and feature != 'attack_type':
                df_copy[feature] = 0
        
        # Select only the common features + attack_type
        selected_cols = [col for col in all_features if col in df_copy.columns] + ['attack_type']
        df_copy = df_copy[selected_cols]
        df_copy['source'] = name
        
        combined_data.append(df_copy)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Handle missing values
    combined_df = combined_df.fillna(0)
    
    print(f"Total samples: {len(combined_df):,}")
    print(f"Total features: {combined_df.shape[1] - 2}")  # Excluding attack_type and source
    
    # Attack type distribution
    print(f"\nAttack Type Distribution:")
    attack_counts = combined_df['attack_type'].value_counts()
    for attack_type, count in attack_counts.items():
        percentage = count / len(combined_df) * 100
        print(f"  {attack_type:15s}: {count:8,} ({percentage:5.1f}%)")
    
    # Source distribution
    print(f"\nDataset Source Distribution:")
    source_counts = combined_df['source'].value_counts()
    for source, count in source_counts.items():
        percentage = count / len(combined_df) * 100
        print(f"  {source:20s}: {count:8,} ({percentage:5.1f}%)")
    
    # Encode attack types
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    combined_df['attack_type_encoded'] = le.fit_transform(combined_df['attack_type'])
    
    # Save label mapping
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    with open('models/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Prepare for training
    X = combined_df.drop(['attack_type', 'attack_type_encoded', 'source'], axis=1)
    y = combined_df['attack_type_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n{'='*70}")
    print("TRAINING MODELS")
    print('='*70)
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    
    # Train multiple models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_train = time.time()
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_train
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        
        # For multi-class AUC
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc_score = 0
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'training_time': train_time,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC Score: {auc_score:.4f}")
        print(f"  Training time: {train_time:.1f}s")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    print('='*70)
    
    # Detailed evaluation of best model
    best_result = results[best_model_name]
    y_pred = best_model.predict(X_test)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion matrix visualization
    plot_confusion_matrix(y_test, y_pred, le.classes_, best_model_name)
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, X.columns, best_model_name)
    
    # Save models and results
    joblib.dump(best_model, 'models/best_nids_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Save all models
    for name, result in results.items():
        joblib.dump(result['model'], f'models/{name.lower()}_model.pkl')
    
    # Save comprehensive results
    results_summary = {
        'best_model': best_model_name,
        'best_accuracy': float(results[best_model_name]['accuracy']),
        'dataset_stats': {
            'total_samples': len(combined_df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'attack_types': len(le.classes_)
        },
        'model_performance': {
            name: {
                'accuracy': float(result['accuracy']),
                'auc_score': float(result['auc_score']),
                'training_time': result['training_time']
            }
            for name, result in results.items()
        }
    }
    
    with open('results/model_performance.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED IN {total_time:.1f} SECONDS")
    print(f"Results saved to 'results/' directory")
    print(f"Models saved to 'models/' directory")
    print('='*70)
    
    return results

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_normalized_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.title(f'Top 20 Feature Importances - {model_name}', fontsize=14)
    plt.bar(range(20), importances[indices[:20]], align='center')
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'plots/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save feature importance to CSV
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(f'results/feature_importance_{model_name}.csv', index=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

def predict_new_data(model_path='models/best_nids_model.pkl', 
                     encoder_path='models/label_encoder.pkl',
                     data_path=None):
    """Predict on new data"""
    print("\n" + "="*70)
    print("PREDICTING ON NEW DATA")
    print("="*70)
    
    # Load model and encoder
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    
    if data_path and os.path.exists(data_path):
        # Load new data
        if data_path.endswith('.parquet'):
            new_data = pd.read_parquet(data_path)
        else:
            new_data = pd.read_csv(data_path)
        
        print(f"Loaded new data: {new_data.shape}")
        
        # Preprocess (simplified)
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns.tolist()
        new_data = new_data[numeric_cols].fillna(0)
        
        # Align features with training
        # This is a simplified version - in production, you'd need proper feature alignment
        
        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)
        
        # Create results
        results = pd.DataFrame({
            'prediction': predictions,
            'predicted_class': le.inverse_transform(predictions),
            'confidence': np.max(probabilities, axis=1)
        })
        
        # Add probabilities for each class
        for i, class_name in enumerate(le.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]
        
        # Save predictions
        results.to_csv('results/new_data_predictions.csv', index=False)
        
        print(f"\nPredictions saved to: results/new_data_predictions.csv")
        print(f"\nPrediction Summary:")
        print(results['predicted_class'].value_counts())
        
        return results
    else:
        print("No data path provided or file not found.")
        print("Usage: predict_new_data(data_path='path/to/your/data.csv')")

if __name__ == "__main__":
    # Run advanced NIDS
    results = advanced_nids()
    
    # Example of how to use prediction function
    # Uncomment to predict on new data
    # predict_new_data(data_path='path/to/new_data.csv')