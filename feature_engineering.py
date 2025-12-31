import pandas as pd 
import numpy as np 
from typing import List,Optional

class FeatureEngineer:
    """Create advanced features for intrusion detection"""
    def __init__(self):
        self.feature_stats = {} 
        
    def create_advanced_features(self,df:pd.DataFrame)->pd.DataFrame:
        """"Create advanced features for network intrusion detection"""
        df =df.copy()
        df = self._create_statistical_features(df)
        df = self._create_time_features(df)
        df = self._create_connection_features(df)
        df = self._create_traffic_pattern_features(df)
        df = self._create_entropy_features(df)
        return df 
    
    def _create_statistical_features(self,df:pd.DataFrame)->pd.DataFrame:
        """Create statistical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'unified_label' in numerical_cols:
            numerical_cols.remove('unified_label')
            
        for  col in numerical_cols[:10]:
            if len(df) > 100:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
                df[f'{col}_robust_scaled'] = (df[col] - df[col].median())/(df[col].quantile(0.75) - df[col].quantile(0.25))
                df[f'{col}_percetile'] = df[col].rank(pct=True)
                
        return df
    
    def _create_time_features(self,df: pd.DataFrame)->pd.DataFrame:
        """Create time based features"""
        time_cols = ['duration','response_time','timestamp']
        available_time_cols = [col for col in time_cols if col in df.columns]
        for col in available_time_cols:
            df[f'{col}_log'] = np.log1p(df[col].abs())
            df[f'{col}_binned'] = pd.qcut(df[col],5,labels=False,duplicates='drop')
        if 'duration' in df.columns and 'src_bytes' in df.columns:
            df['bytes_per_seconds'] = df['src_bytes'] / (df['duration']+1)
                
        return df 
    def _create_connection_features(self,df: pd.DataFrame) -> pd.DataFrame:
        """Create connection based features"""
        count_features = ['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
        available_counts = [col for col in count_features if col in df.columns]
        for col in available_counts:
            df[f'{col}_log'] = np.log1p(df[col])
            if 'duration' in df.columns:
                df[f'{col}_rate'] = df[col] / (df['duration']+1)
                
        if  all(col in df.columns for col in ['serror_rate','srv_serror_rate']):
            df['total_error_rate'] = (df['serror_rate'] + df['srv_serror_rate']) / 2 
        return df 
    def _create_traffic_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create traffic pattern features"""
        # Source-destination interaction
        if all(col in df.columns for col in ['src_bytes', 'dst_bytes']):
            df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
            df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
            df['bytes_diff'] = abs(df['src_bytes'] - df['dst_bytes'])
        
        # Service patterns
        service_cols = ['same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']
        available_service = [col for col in service_cols if col in df.columns]
        
        if available_service:
            df['service_concentration'] = df[available_service].mean(axis=1)
        
        return df
        
    def _create_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create entropy-based features for anomaly detection"""
        # Calculate entropy for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            if len(df[col].unique()) > 1:
                # Calculate probability distribution
                value_counts = df[col].value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                df[f'{col}_entropy'] = entropy
        
        # Packet size entropy (if packet size data available)
        byte_cols = [col for col in df.columns if 'byte' in col.lower()]
        for col in byte_cols[:3]:  # First 3 byte columns
            if df[col].nunique() > 1:
                # Discretize and calculate entropy
                binned = pd.cut(df[col], bins=10, labels=False)
                value_counts = pd.Series(binned).value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                df[f'{col}_entropy'] = entropy
        
        return df
    def reduce_dimensionality(self, df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """Reduce feature dimensionality using PCA"""
        from sklearn.decomposition import PCA
        
        # Separate features and labels
        if 'unified_label' in df.columns:
            labels = df['unified_label']
            features = df.drop('unified_label', axis=1)
        else:
            labels = None
            features = df
        
        # Handle categorical columns
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            features = pd.get_dummies(features, columns=categorical_cols)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, features.shape[1]))
        reduced_features = pca.fit_transform(features)
        
        # Create column names for PCA components
        pca_columns = [f'pca_{i+1}' for i in range(reduced_features.shape[1])]
        reduced_df = pd.DataFrame(reduced_features, columns=pca_columns)
        
        # Add back labels if they exist
        if labels is not None:
            reduced_df['unified_label'] = labels.values
        
        # Save explained variance
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_variance.sum():.2%}")
        print(f"Top 10 components explain: {explained_variance[:10].sum():.2%}")
        
        return reduced_df
        