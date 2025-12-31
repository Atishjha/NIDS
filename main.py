import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Simple implementation without complex imports
def run_simple_nids():
    """Simple NIDS implementation"""
    print("="*70)
    print("SIMPLE NETWORK INTRUSION DETECTION SYSTEM")
    print("="*70)
    
    start_time = time.time()
    
    # Create directories
    for dir_name in ['models', 'results', 'logs']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Load and process each dataset
    datasets = []
    
    # 1. Load NSL-KDD Train
    print("\n[1] Loading NSL-KDD Train...")
    try:
        kdd_train = pd.read_parquet('KDDTrain.parquet')
        print(f"   Loaded: {kdd_train.shape}")
        
        # Find label column
        label_col = None
        for col in kdd_train.columns:
            if col.lower() in ['label', 'target', 'class']:
                label_col = col
                break
        
        if label_col:
            # Create binary label
            kdd_train['is_attack'] = ~kdd_train[label_col].astype(str).str.contains('normal', case=False)
            # Select numeric columns
            numeric_cols = kdd_train.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            # Take first 30 numeric columns
            selected_cols = numeric_cols[:30] + ['is_attack']
            kdd_train = kdd_train[selected_cols]
            datasets.append(kdd_train)
            print(f"   Processed: {len(kdd_train)} samples")
        else:
            print("   Warning: No label column found")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Load NSL-KDD Test
    print("\n[2] Loading NSL-KDD Test...")
    try:
        kdd_test = pd.read_parquet('KDDTest.parquet')
        print(f"   Loaded: {kdd_test.shape}")
        
        label_col = None
        for col in kdd_test.columns:
            if col.lower() in ['label', 'target', 'class']:
                label_col = col
                break
        
        if label_col:
            kdd_test['is_attack'] = ~kdd_test[label_col].astype(str).str.contains('normal', case=False)
            numeric_cols = kdd_test.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            selected_cols = numeric_cols[:30] + ['is_attack']
            kdd_test = kdd_test[selected_cols]
            datasets.append(kdd_test)
            print(f"   Processed: {len(kdd_test)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Load UNSW-NB15 Train
    print("\n[3] Loading UNSW-NB15 Train...")
    try:
        unsw_train = pd.read_parquet('UNSW_NB15_training-set.parquet')
        print(f"   Loaded: {unsw_train.shape}")
        
        # UNSW has 'label' column (0=normal, 1=attack)
        if 'label' in unsw_train.columns:
            unsw_train['is_attack'] = unsw_train['label']
            numeric_cols = unsw_train.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            selected_cols = numeric_cols[:30] + ['is_attack']
            unsw_train = unsw_train[selected_cols]
            datasets.append(unsw_train)
            print(f"   Processed: {len(unsw_train)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Load UNSW-NB15 Test
    print("\n[4] Loading UNSW-NB15 Test...")
    try:
        unsw_test = pd.read_parquet('UNSW_NB15_testing-set.parquet')
        print(f"   Loaded: {unsw_test.shape}")
        
        if 'label' in unsw_test.columns:
            unsw_test['is_attack'] = unsw_test['label']
            numeric_cols = unsw_test.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            selected_cols = numeric_cols[:30] + ['is_attack']
            unsw_test = unsw_test[selected_cols]
            datasets.append(unsw_test)
            print(f"   Processed: {len(unsw_test)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. Load CIC-IDS2017 (first 100k rows for speed)
    print("\n[5] Loading CIC-IDS2017...")
    try:
        cicids = pd.read_csv('cicids2017_cleaned.csv', nrows=100000)
        print(f"   Loaded: {cicids.shape}")
        
        # Find label column
        label_col = None
        for col in cicids.columns:
            if 'label' in col.lower() or 'attack' in col.lower():
                label_col = col
                break
        
        if label_col:
            cicids['is_attack'] = ~cicids[label_col].astype(str).str.contains('benign', case=False)
            numeric_cols = cicids.select_dtypes(include=[np.number]).columns.tolist()
            if 'is_attack' in numeric_cols:
                numeric_cols.remove('is_attack')
            selected_cols = numeric_cols[:30] + ['is_attack']
            cicids = cicids[selected_cols]
            datasets.append(cicids)
            print(f"   Processed: {len(cicids)} samples")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Combine all datasets
    if not datasets:
        print("\n✗ No datasets were loaded successfully!")
        return
    
    print(f"\n{'='*70}")
    print("COMBINING DATASETS")
    print('='*70)
    
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"Total samples: {len(combined_df):,}")
    print(f"Total features: {combined_df.shape[1] - 1}")
    
    # Check label distribution
    print(f"\nLabel distribution:")
    attack_count = combined_df['is_attack'].sum()
    normal_count = len(combined_df) - attack_count
    print(f"  Normal: {normal_count:,} ({normal_count/len(combined_df)*100:.1f}%)")
    print(f"  Attack: {attack_count:,} ({attack_count/len(combined_df)*100:.1f}%)")
    
    # Handle missing values
    combined_df = combined_df.fillna(0)
    
    # Prepare for training
    X = combined_df.drop('is_attack', axis=1)
    y = combined_df['is_attack'].astype(int)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n{'='*70}")
    print("TRAINING MODEL")
    print('='*70)
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    
    # Train RandomForest
    print("\nTraining RandomForest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"               Normal  Attack")
    print(f"Actual Normal  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Attack  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Feature importance
    print(f"\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    joblib.dump(model, 'models/nids_model.pkl')
    print(f"\nModel saved to: models/nids_model.pkl")
    
    # Save results
    results = {
        'accuracy': model.score(X_test, y_test),
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict()
    }
    
    import json
    with open('results/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED IN {total_time:.1f} SECONDS")
    print('='*70)

if __name__ == "__main__":
    run_simple_nids()