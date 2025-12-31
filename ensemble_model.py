import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from config import Config

class EnsembleModel:
    """Create ensemble models for improved performance"""
    
    def __init__(self):
        self.config = Config
        self.ensemble_models = {}
        
    def create_voting_ensemble(self, models_dict: dict, voting_type: str = 'soft'):
        """Create voting ensemble classifier"""
        estimators = [(name, model) for name, model in models_dict.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            n_jobs=-1
        )
        
        return voting_clf
    
    def create_stacking_ensemble(self, models_dict: dict):
        """Create stacking ensemble classifier"""
        estimators = [(name, model) for name, model in models_dict.items()]
        
        # Use logistic regression as final estimator
        final_estimator = LogisticRegression(
            max_iter=1000,
            random_state=self.config.MODEL_CONFIG.random_state,
            class_weight='balanced'
        )
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )
        
        return stacking_clf
    
    def create_weighted_ensemble(self, models_dict: dict, weights: list = None):
        """Create weighted ensemble prediction"""
        if weights is None:
            # Equal weights by default
            weights = [1/len(models_dict)] * len(models_dict)
        
        def weighted_predict(X):
            predictions = []
            for (name, model), weight in zip(models_dict.items(), weights):
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X) * weight
                else:
                    # Convert class predictions to probability-like format
                    pred = np.zeros((X.shape[0], len(np.unique(model.classes_))))
                    class_pred = model.predict(X)
                    for i, cls in enumerate(model.classes_):
                        pred[:, i] = (class_pred == cls).astype(float) * weight
                predictions.append(pred)
            
            # Average predictions
            avg_pred = np.mean(predictions, axis=0)
            return np.argmax(avg_pred, axis=1)
        
        return weighted_predict
    
    def train_ensemble(self, X_train, y_train, base_models: dict):
        """Train ensemble models"""
        print("Training ensemble models...")
        
        # Voting Ensemble
        print("\n1. Training Voting Ensemble...")
        voting_clf = self.create_voting_ensemble(base_models)
        voting_clf.fit(X_train, y_train)
        self.ensemble_models['Voting'] = voting_clf
        
        # Stacking Ensemble
        print("\n2. Training Stacking Ensemble...")
        stacking_clf = self.create_stacking_ensemble(base_models)
        stacking_clf.fit(X_train, y_train)
        self.ensemble_models['Stacking'] = stacking_clf
        
        # Store base models for weighted ensemble
        self.base_models = base_models
        
        print("\nEnsemble models trained successfully!")
        
        return self.ensemble_models
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble models"""
        results = {}
        
        for name, model in self.ensemble_models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name} Ensemble...")
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = (y_pred == y_test).mean()
            
            # Classification report
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # AUC score
            from sklearn.metrics import roc_auc_score
            try:
                y_pred_proba = model.predict_proba(X_test)
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                auc_score = 0
            
            results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'classification_report': report
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
        
        # Evaluate weighted ensemble
        print(f"\n{'='*50}")
        print("Evaluating Weighted Ensemble...")
        
        weighted_predict = self.create_weighted_ensemble(self.base_models)
        y_pred_weighted = weighted_predict(X_test)
        accuracy_weighted = (y_pred_weighted == y_test).mean()
        
        results['Weighted'] = {
            'accuracy': accuracy_weighted,
            'auc_score': 0,  # Can't calculate AUC without probabilities
            'classification_report': classification_report(y_test, y_pred_weighted, output_dict=True)
        }
        
        print(f"Accuracy: {accuracy_weighted:.4f}")
        
        return results
    
    def save_ensemble_models(self):
        """Save ensemble models"""
        for name, model in self.ensemble_models.items():
            joblib.dump(model, f"{self.config.MODELS_DIR}/ensemble_{name}.pkl")
            print(f"Saved {name} ensemble model")
        
        # Save base models reference
        joblib.dump(self.base_models, f"{self.config.MODELS_DIR}/base_models.pkl")