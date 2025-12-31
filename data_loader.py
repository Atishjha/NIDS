import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from config import Config

class DataLoader:
    """Load and integrate multiple NIDS datasets"""
    
    def __init__(self):
        self.config = Config
        self.config.create_directories()
        
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load individual dataset"""
        dataset_config = self.config.DATASETS[dataset_name]
        
        try:
            if dataset_config.path.endswith('.parquet'):
                df = pd.read_parquet(dataset_config.path)
            else:
                df = pd.read_csv(dataset_config.path, nrows=100000)  # Limit rows for testing
            
            print(f"✓ Loaded {dataset_config.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            
            return df
            
        except Exception as e:
            print(f"✗ Error loading {dataset_config.name}: {e}")
            return pd.DataFrame()
    
    def explore_dataset(self, df: pd.DataFrame, dataset_name: str):
        """Explore dataset structure"""
        print(f"\nExploring {dataset_name}:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        # Look for potential label columns
        potential_labels = [col for col in df.columns if 'label' in col.lower() or 
                           'attack' in col.lower() or 'class' in col.lower()]
        print(f"\nPotential label columns: {potential_labels}")
        
        if potential_labels:
            for label_col in potential_labels:
                print(f"\nUnique values in '{label_col}':")
                print(df[label_col].value_counts().head())
    
    def prepare_labels(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Prepare labels for the dataset"""
        dataset_config = self.config.DATASETS[dataset_name]
        df = df.copy()
        
        # Check if label column exists
        if dataset_config.label_column in df.columns:
            print(f"Using label column: '{dataset_config.label_column}'")
            
            # Convert to string for comparison
            labels = df[dataset_config.label_column].astype(str).str.strip()
            
            # Create binary labels (normal vs attack)
            is_normal = labels.str.lower().str.contains(
                dataset_config.normal_label.lower(), na=False
            )
            
            # Initialize unified label as attack (1)
            df['unified_label'] = 1
            
            # Mark normal traffic as 0
            df.loc[is_normal, 'unified_label'] = 0
            
            print(f"Label distribution:")
            normal_count = is_normal.sum()
            attack_count = len(df) - normal_count
            print(f"  Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
            print(f"  Attack: {attack_count} ({attack_count/len(df)*100:.1f}%)")
            
        else:
            print(f"Warning: Label column '{dataset_config.label_column}' not found")
            print("Available columns:", df.columns.tolist())
            
            # Try to find alternative label column
            alt_labels = [col for col in df.columns if 'label' in col.lower() or 
                         'attack' in col.lower() or 'target' in col.lower()]
            
            if alt_labels:
                print(f"Trying alternative label column: {alt_labels[0]}")
                dataset_config.label_column = alt_labels[0]
                
                # Convert to string for comparison
                labels = df[dataset_config.label_column].astype(str).str.strip()
                
                # Try to identify normal traffic
                # Look for common normal indicators
                normal_indicators = ['normal', 'benign', 'legitimate', '0', 'false']
                is_normal = labels.str.lower().str.contains(
                    '|'.join(normal_indicators), na=False
                )
                
                df['unified_label'] = 1
                df.loc[is_normal, 'unified_label'] = 0
                
                print(f"Label distribution (estimated):")
                normal_count = is_normal.sum()
                attack_count = len(df) - normal_count
                print(f"  Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
                print(f"  Attack: {attack_count} ({attack_count/len(df)*100:.1f}%)")
            else:
                print("No label column found. Defaulting all to attack for safety.")
                df['unified_label'] = 1
        
        df['source_dataset'] = dataset_name
        return df
    
    def load_and_prepare_all(self):
        """Load and prepare all datasets"""
        all_data = []
        
        for dataset_name in self.config.DATASETS.keys():
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print('='*60)
            
            df = self.load_dataset(dataset_name)
            if df.empty:
                continue
                
            # Explore dataset structure
            self.explore_dataset(df, dataset_name)
            
            # Prepare labels
            df = self.prepare_labels(df, dataset_name)
            
            # Select only numeric columns and the label
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Keep label and source columns
            keep_cols = [col for col in numeric_cols if col != 'unified_label' and col != 'source_dataset']
            keep_cols = keep_cols[:50]  # Limit to 50 features for speed
            
            selected_cols = keep_cols + ['unified_label', 'source_dataset']
            df = df[selected_cols]
            
            all_data.append(df)
            print(f"Prepared {len(df)} samples with {len(keep_cols)} features")
        
        if all_data:
            unified_df = pd.concat(all_data, ignore_index=True)
            print(f"\n{'='*60}")
            print("UNIFIED DATASET SUMMARY")
            print('='*60)
            print(f"Total samples: {len(unified_df):,}")
            print(f"Total features: {unified_df.shape[1] - 2}")  # Excluding label and source
            
            label_dist = unified_df['unified_label'].value_counts()
            print(f"\nLabel distribution:")
            for label, count in label_dist.items():
                label_name = 'Normal' if label == 0 else 'Attack'
                print(f"  {label_name}: {count:,} ({count/len(unified_df)*100:.1f}%)")
            
            print(f"\nDataset sources:")
            source_dist = unified_df['source_dataset'].value_counts()
            for source, count in source_dist.items():
                print(f"  {source}: {count:,} ({count/len(unified_df)*100:.1f}%)")
            
            return unified_df
        else:
            print("No data was loaded successfully")
            return pd.DataFrame()
    
    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        if df.empty:
            print("Cannot split empty dataset")
            return pd.DataFrame(), pd.DataFrame()
        
        # Separate features and labels
        X = df.drop(['unified_label', 'source_dataset'], axis=1)
        y = df['unified_label']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Recombine
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df['source_dataset'] = df.loc[train_df.index, 'source_dataset'].values
        test_df['source_dataset'] = df.loc[test_df.index, 'source_dataset'].values
        
        print(f"\nSplit completed:")
        print(f"Training set: {len(train_df):,} samples")
        print(f"Test set: {len(test_df):,} samples")
        
        return train_df, test_df