import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from config import Config

class DataPreprocessor:
    """Preprocess the unified dataset"""
    
    def __init__(self):
        self.config = Config
        self.label_encoders = {}
        self.scaler = None
        self.imputer_numeric = None
        self.imputer_categorical = None
        self.selected_features = None
        
    def preprocess(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df = df.copy()
        
        print(f"Preprocessing: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Store labels and dataset source
        if 'unified_label' in df.columns:
            labels = df['unified_label']
            df = df.drop('unified_label', axis=1)
        
        dataset_source = None
        if 'source_dataset' in df.columns:
            dataset_source = df['source_dataset']
            df = df.drop('source_dataset', axis=1)
        
        # Handle missing values
        df = self._handle_missing_values(df, is_training)
        
        # Encode categorical features
        df = self._encode_categorical_features(df, is_training)
        
        # Scale features
        df = self._scale_features(df, is_training)
        
        # Add back labels and source
        if 'unified_label' in locals():
            df['unified_label'] = labels.values
        
        if dataset_source is not None:
            df['source_dataset'] = dataset_source.values
        
        print(f"After preprocessing: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric missing values
        missing_numeric = [col for col in numeric_cols if df[col].isnull().any()]
        if missing_numeric:
            print(f"Numeric columns with missing values: {len(missing_numeric)}")
            
            if is_training:
                self.imputer_numeric = SimpleImputer(strategy='median')
                df[missing_numeric] = self.imputer_numeric.fit_transform(df[missing_numeric])
                joblib.dump(self.imputer_numeric, f"{self.config.MODELS_DIR}/imputer_numeric.pkl")
            else:
                if self.imputer_numeric is None:
                    self.imputer_numeric = joblib.load(f"{self.config.MODELS_DIR}/imputer_numeric.pkl")
                df[missing_numeric] = self.imputer_numeric.transform(df[missing_numeric])
        
        # Handle categorical missing values
        missing_categorical = [col for col in categorical_cols if df[col].isnull().any()]
        if missing_categorical:
            print(f"Categorical columns with missing values: {len(missing_categorical)}")
            
            if is_training:
                self.imputer_categorical = SimpleImputer(strategy='most_frequent')
                df[missing_categorical] = self.imputer_categorical.fit_transform(df[missing_categorical])
                joblib.dump(self.imputer_categorical, f"{self.config.MODELS_DIR}/imputer_categorical.pkl")
            else:
                if self.imputer_categorical is None:
                    self.imputer_categorical = joblib.load(f"{self.config.MODELS_DIR}/imputer_categorical.pkl")
                df[missing_categorical] = self.imputer_categorical.transform(df[missing_categorical])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if df[col].nunique() > 100:  # Skip high cardinality columns
                df = df.drop(col, axis=1)
                print(f"Dropped high cardinality column: {col}")
                continue
                
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                joblib.dump(le, f"{self.config.MODELS_DIR}/label_encoder_{col}.pkl")
            else:
                if col not in self.label_encoders:
                    try:
                        self.label_encoders[col] = joblib.load(f"{self.config.MODELS_DIR}/label_encoder_{col}.pkl")
                    except:
                        # Create new encoder if not found
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                else:
                    # Handle unseen categories
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except:
                        # If unknown categories, use most frequent
                        df[col] = self.label_encoders[col].transform(['unknown'] * len(df))
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Scale numerical features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove label column if present
        if 'unified_label' in numeric_cols:
            numeric_cols.remove('unified_label')
        if 'source_dataset' in numeric_cols:
            numeric_cols.remove('source_dataset')
        
        if numeric_cols:
            if is_training:
                self.scaler = StandardScaler()
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
                joblib.dump(self.scaler, f"{self.config.MODELS_DIR}/scaler.pkl")
            else:
                if self.scaler is None:
                    self.scaler = joblib.load(f"{self.config.MODELS_DIR}/scaler.pkl")
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def save_preprocessing_artifacts(self):
        """Save preprocessing artifacts"""
        artifacts = {
            'label_encoders': self.label_encoders,
            'selected_features': self.selected_features
        }
        joblib.dump(artifacts, f"{self.config.MODELS_DIR}/preprocessing_artifacts.pkl")
        print("Saved preprocessing artifacts")