"""
Anomaly detection module using statistical and ML methods
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config.settings import ANOMALY_PARAMS

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Handles anomaly detection using multiple methods"""
    
    def __init__(self, params: Dict = None):
        """
        Initialize AnomalyDetector
        
        Args:
            params: Anomaly detection parameters
        """
        self.params = params or ANOMALY_PARAMS
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.anomaly_results = None
        
    def detect_statistical_anomalies(self, df: pd.DataFrame, 
                                    feature_columns: List[str]) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods (Z-score and IQR)
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to analyze
            
        Returns:
            DataFrame with anomaly flags
        """
        logger.info("Detecting statistical anomalies")
        
        result_df = df.copy()
        
        # Initialize anomaly columns
        for method in ['zscore', 'iqr']:
            result_df[f'anomaly_{method}'] = 0
            for feature in feature_columns:
                result_df[f'anomaly_{method}_{feature}'] = 0
        
        # Z-score method
        for feature in feature_columns:
            if feature in result_df.columns:
                z_scores = np.abs(stats.zscore(result_df[feature].fillna(0)))
                result_df[f'anomaly_zscore_{feature}'] = (
                    z_scores > self.params['z_score_threshold']
                ).astype(int)
                result_df['anomaly_zscore'] += result_df[f'anomaly_zscore_{feature}']
        
        # IQR method
        for feature in feature_columns:
            if feature in result_df.columns:
                Q1 = result_df[feature].quantile(0.25)
                Q3 = result_df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.params['iqr_multiplier'] * IQR
                upper_bound = Q3 + self.params['iqr_multiplier'] * IQR
                
                result_df[f'anomaly_iqr_{feature}'] = (
                    (result_df[feature] < lower_bound) | 
                    (result_df[feature] > upper_bound)
                ).astype(int)
                result_df['anomaly_iqr'] += result_df[f'anomaly_iqr_{feature}']
        
        # Create combined statistical anomaly flag
        result_df['anomaly_statistical'] = (
            (result_df['anomaly_zscore'] > 0) | 
            (result_df['anomaly_iqr'] > 0)
        ).astype(int)
        
        logger.info(f"Statistical anomalies detected: {result_df['anomaly_statistical'].sum()}")
        return result_df
    
    def detect_ml_anomalies(self, df: pd.DataFrame, 
                           feature_columns: List[str]) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to use
            
        Returns:
            DataFrame with ML anomaly flags
        """
        logger.info("Detecting ML anomalies with Isolation Forest")
        
        # Prepare data
        X = df[feature_columns].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=self.params['isolation_forest_n_estimators'],
            contamination=self.params['isolation_forest_contamination'],
            random_state=42
        )
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        anomaly_predictions = self.isolation_forest.fit_predict(X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        result_df = df.copy()
        result_df['anomaly_ml'] = (anomaly_predictions == -1).astype(int)
        
        # Get anomaly scores
        result_df['anomaly_score'] = -self.isolation_forest.score_samples(X_scaled)
        
        logger.info(f"ML anomalies detected: {result_df['anomaly_ml'].sum()}")
        return result_df
    
    def detect_combined_anomalies(self, df: pd.DataFrame, 
                                 feature_columns: List[str]) -> pd.DataFrame:
        """
        Combine statistical and ML anomaly detection
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to use
            
        Returns:
            DataFrame with combined anomaly flags
        """
        logger.info("Detecting combined anomalies")
        
        # Detect statistical anomalies
        df_statistical = self.detect_statistical_anomalies(df, feature_columns)
        
        # Detect ML anomalies
        df_ml = self.detect_ml_anomalies(df, feature_columns)
        
        # Combine results
        result_df = df_statistical.copy()
        result_df['anomaly_ml'] = df_ml['anomaly_ml']
        result_df['anomaly_score'] = df_ml['anomaly_score']
        
        # Create combined anomaly flag
        result_df['anomaly_combined'] = (
            (result_df['anomaly_statistical'] == 1) | 
            (result_df['anomaly_ml'] == 1)
        ).astype(int)
        
        # Categorize anomalies
        result_df['anomaly_type'] = 'normal'
        result_df.loc[result_df['anomaly_combined'] == 1, 'anomaly_type'] = 'anomaly'
        result_df.loc[
            (result_df['anomaly_statistical'] == 1) & 
            (result_df['anomaly_ml'] == 1), 'anomaly_type'
        ] = 'severe_anomaly'
        
        self.anomaly_results = result_df
        
        logger.info(f"Total anomalies detected: {result_df['anomaly_combined'].sum()}")
        logger.info(f"Severe anomalies: {(result_df['anomaly_type'] == 'severe_anomaly').sum()}")
        
        return result_df
    
    def get_anomaly_analysis(self) -> Dict:
        """
        Analyze and summarize detected anomalies
        
        Returns:
            Dictionary with anomaly analysis
        """
        if self.anomaly_results is None:
            raise ValueError("No anomaly results available. Run detect_combined_anomalies() first.")
            
        analysis = {
            'total_trips': len(self.anomaly_results),
            'statistical_anomalies': self.anomaly_results['anomaly_statistical'].sum(),
            'ml_anomalies': self.anomaly_results['anomaly_ml'].sum(),
            'combined_anomalies': self.anomaly_results['anomaly_combined'].sum(),
            'severe_anomalies': (self.anomaly_results['anomaly_type'] == 'severe_anomaly').sum(),
            'anomaly_rate': self.anomaly_results['anomaly_combined'].mean() * 100,
        }
        
        # Analyze by feature
        feature_analysis = {}
        for col in self.anomaly_results.columns:
            if col.startswith('anomaly_zscore_'):
                feature_name = col.replace('anomaly_zscore_', '')
                feature_analysis[f'zscore_{feature_name}'] = self.anomaly_results[col].sum()
            elif col.startswith('anomaly_iqr_'):
                feature_name = col.replace('anomaly_iqr_', '')
                feature_analysis[f'iqr_{feature_name}'] = self.anomaly_results[col].sum()
        
        analysis['feature_contributions'] = feature_analysis
        
        return analysis
    
    def get_top_anomalous_trips(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N most anomalous trips
        
        Args:
            n: Number of trips to return
            
        Returns:
            DataFrame with top anomalous trips
        """
        if self.anomaly_results is None:
            raise ValueError("No anomaly results available")
            
        return self.anomaly_results.nlargest(n, 'anomaly_score')