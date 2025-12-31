import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import joblib
from typing import Dict, Any
from config import Config

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.config = Config
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def evaluate_model(self, model, X_test, y_test, model_name: str = "Model"):
        """Comprehensive model evaluation"""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS FOR {model_name.upper()}")
        print('='*60)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nBasic Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.config.UNIFIED_LABELS.values()))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        # ROC Curve for multi-class
        if y_pred_proba is not None:
            self.plot_roc_curve(y_test, y_pred_proba, model_name)
        
        # Precision-Recall curve
        self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, X_test.columns, model_name)
        
        # Save evaluation metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        joblib.dump(metrics, f"{self.config.RESULTS_DIR}/metrics_{model_name}.pkl")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        labels = list(self.config.UNIFIED_LABELS.values())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.config.RESULTS_DIR}/confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print confusion matrix as percentage
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized Confusion Matrix (Rows sum to 1):")
        print(pd.DataFrame(cm_normalized, index=labels, columns=labels).round(3))
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name: str):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(len(self.config.UNIFIED_LABELS)))
        n_classes = y_true_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, 
                    label=f'{self.config.UNIFIED_LABELS[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.config.RESULTS_DIR}/roc_curve_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print AUC scores
        print("\nAUC Scores per class:")
        for i in range(n_classes):
            print(f"{self.config.UNIFIED_LABELS[i]}: {roc_auc[i]:.4f}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name: str):
        """Plot precision-recall curves"""
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(len(self.config.UNIFIED_LABELS)))
        n_classes = y_true_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, lw=2, 
                    label=f'{self.config.UNIFIED_LABELS[i]}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.config.RESULTS_DIR}/precision_recall_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name: str):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances - {model_name}')
            plt.bar(range(min(20, len(feature_names))), 
                   importances[indices[:20]], 
                   align='center')
            plt.xticks(range(min(20, len(feature_names))), 
                      [feature_names[i] for i in indices[:20]], 
                      rotation=90)
            plt.tight_layout()
            plt.savefig(f"{self.config.RESULTS_DIR}/feature_importance_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print top features
            print("\nTop 20 Feature Importances:")
            for i in range(min(20, len(feature_names))):
                print(f"{i+1:2d}. {feature_names[indices[i]]:30s} {importances[indices[i]]:.4f}")
    
    def compare_models(self, models_results: Dict[str, Dict]):
        """Compare multiple models"""
        comparison_data = []
        
        for model_name, results in models_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['test_accuracy'],
                'Precision': results['classification_report']['weighted avg']['precision'],
                'Recall': results['classification_report']['weighted avg']['recall'],
                'F1-Score': results['classification_report']['weighted avg']['f1-score'],
                'AUC': results.get('auc_score', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.config.RESULTS_DIR}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def analyze_false_predictions(self, model, X_test, y_test, model_name: str):
        """Analyze false predictions"""
        y_pred = model.predict(X_test)
        
        false_positives = X_test[(y_test == 0) & (y_pred != 0)].copy()
        false_negatives = X_test[(y_test != 0) & (y_pred == 0)].copy()
        
        if not false_positives.empty:
            false_positives['predicted_label'] = y_pred[(y_test == 0) & (y_pred != 0)]
            false_positives['true_label'] = y_test[(y_test == 0) & (y_pred != 0)]
            print(f"\nFalse Positives Analysis ({len(false_positives)} instances):")
            print("Most common predicted attack types:")
            print(false_positives['predicted_label'].value_counts().head())
        
        if not false_negatives.empty:
            false_negatives['predicted_label'] = y_pred[(y_test != 0) & (y_pred == 0)]
            false_negatives['true_label'] = y_test[(y_test != 0) & (y_pred == 0)]
            print(f"\nFalse Negatives Analysis ({len(false_negatives)} instances):")
            print("Most common missed attack types:")
            print(false_negatives['true_label'].value_counts().head())
        
        # Save false predictions for further analysis
        if not false_positives.empty:
            false_positives.to_csv(f"{self.config.RESULTS_DIR}/false_positives_{model_name}.csv", index=False)
        if not false_negatives.empty:
            false_negatives.to_csv(f"{self.config.RESULTS_DIR}/false_negatives_{model_name}.csv", index=False)