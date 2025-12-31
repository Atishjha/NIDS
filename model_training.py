import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')
from config import Config
import time

class ModelTrainer:
    """Train multiple ML models for intrusion detection"""
    
    def __init__(self):
        self.config = Config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for training"""
        # Separate features and labels
        X = df.drop(['unified_label', 'source_dataset'], axis=1, errors='ignore')
        y = df['unified_label']
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.MODEL_CONFIG.validation_size,
            stratify=y,
            random_state=self.config.MODEL_CONFIG.random_state
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Feature count: {X_train.shape[1]}")
        
        return X_train, X_val, y_train, y_val
    
    def train_models(self, X_train, X_val, y_train, y_val):
        """Train multiple models"""
        models_to_train = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.config.MODEL_CONFIG.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.MODEL_CONFIG.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.MODEL_CONFIG.random_state,
                n_jobs=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=200,
                depth=8,
                learning_rate=0.05,
                random_seed=self.config.MODEL_CONFIG.random_state,
                verbose=False,
                task_type='CPU'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.config.MODEL_CONFIG.random_state
            ),
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                alpha=0.001,
                random_state=self.config.MODEL_CONFIG.random_state,
                early_stopping=True,
                verbose=False
            )
        }
        
        for name, model in models_to_train.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            start_time = time.time()
            
            try:
                # Handle class imbalance
                if name not in ['CatBoost', 'XGBoost', 'LightGBM']:
                    # For sklearn models, use class weights
                    from sklearn.utils.class_weight import compute_class_weight
                    classes = np.unique(y_train)
                    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                    class_weight_dict = dict(zip(classes, class_weights))
                    
                    if hasattr(model, 'class_weight'):
                        model.class_weight = class_weight_dict
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                
                # Predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                report = classification_report(y_val, y_pred, output_dict=True)
                auc_score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else 0
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'train_score': train_score,
                    'val_score': val_score,
                    'classification_report': report,
                    'auc_score': auc_score,
                    'training_time': time.time() - start_time,
                    'model': model
                }
                
                print(f"Train Accuracy: {train_score:.4f}")
                print(f"Validation Accuracy: {val_score:.4f}")
                print(f"AUC Score: {auc_score:.4f}")
                print(f"Training Time: {time.time() - start_time:.2f}s")
                
                # Update best model
                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name} with validation score: {self.best_score:.4f}")
    
    def cross_validate(self, X, y, model, cv=5):
        """Perform cross-validation"""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.config.MODEL_CONFIG.random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        evaluation_results = {}
        
        for name, model_info in self.results.items():
            model = model_info['model']
            
            print(f"\n{'='*50}")
            print(f"Evaluating {name} on test set...")
            
            # Test predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            test_accuracy = model.score(X_test, y_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else 0
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            evaluation_results[name] = {
                'test_accuracy': test_accuracy,
                'classification_report': report,
                'auc_score': auc_score,
                'confusion_matrix': cm
            }
            
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test AUC: {auc_score:.4f}")
            print(f"Classification Report:")
            print(classification_report(y_test, y_pred))
        
        return evaluation_results
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            joblib.dump(model, f"{self.config.MODELS_DIR}/{name}_model.pkl")
            print(f"Saved {name} model")
        
        # Save best model separately
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{self.config.MODELS_DIR}/best_model.pkl")
            print(f"Saved best model ({self.best_model_name})")
        
        # Save results
        joblib.dump(self.results, f"{self.config.RESULTS_DIR}/training_results.pkl")
    
    def load_model(self, model_name: str):
        """Load a saved model"""
        return joblib.load(f"{self.config.MODELS_DIR}/{model_name}_model.pkl")