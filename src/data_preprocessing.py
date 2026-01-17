"""
Data preprocessing module for cleaning and preparing trip data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple, List, Dict
import logging
from geopy.distance import geodesic
from config.settings import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self, params: Dict = None):
        """
        Initialize DataPreprocessor
        
        Args:
            params: Preprocessing parameters
        """
        self.params = params or FEATURE_PARAMS
        self.processed_data = None
        
    def parse_coordinates(self, point_str: str) -> Tuple[float, float]:
        """
        Parse coordinates from POINT string
        
        Args:
            point_str: String in format "POINT(lon lat)"
            
        Returns:
            Tuple of (longitude, latitude)
        """
        try:
            # Extract numbers from string
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", point_str)
            if len(numbers) >= 2:
                return float(numbers[0]), float(numbers[1])
            else:
                return np.nan, np.nan
        except:
            return np.nan, np.nan
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning")
        
        # Create a copy
        df = data.copy()
        
        # Parse coordinates
        logger.info("Parsing coordinates")
        coords_source = df['source_point'].apply(self.parse_coordinates)
        coords_target = df['target_point'].apply(self.parse_coordinates)
        
        df['source_lon'] = coords_source.apply(lambda x: x[0])
        df['source_lat'] = coords_source.apply(lambda x: x[1])
        df['target_lon'] = coords_target.apply(lambda x: x[0])
        df['target_lat'] = coords_target.apply(lambda x: x[1])
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate basic trip metrics
        logger.info("Calculating basic trip metrics")
        df['distance_km'] = df.apply(
            lambda row: geodesic(
                (row['source_lat'], row['source_lon']),
                (row['target_lat'], row['target_lon'])
            ).kilometers, axis=1
        )
        
        # Filter invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=['source_lon', 'source_lat', 'target_lon', 'target_lat'])
        logger.info(f"Removed {initial_count - len(df)} records with invalid coordinates")
        
        # Filter unrealistic distances (based on max speed and time)
        # Since we don't have trip duration in raw data, we'll filter extreme distances
        max_reasonable_distance = 100  # km
        df = df[df['distance_km'] <= max_reasonable_distance]
        df = df[df['distance_km'] > 0]
        
        logger.info(f"Final cleaned data shape: {df.shape}")
        self.processed_data = df
        return df
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Detect outliers using IQR method
        
        Args:
            df: DataFrame
            column: Column name to check for outliers
            
        Returns:
            Boolean Series indicating outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    def get_cleaning_report(self) -> Dict:
        """
        Generate a report of the cleaning process
        
        Returns:
            Dictionary with cleaning statistics
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
            
        report = {
            'total_records': len(self.processed_data),
            'columns': list(self.processed_data.columns),
            'taxi_count': self.processed_data['taxi_id'].nunique(),
            'trajectory_count': self.processed_data['trajectory_id'].nunique(),
            'date_range': {
                'start': self.processed_data['timestamp'].min(),
                'end': self.processed_data['timestamp'].max()
            },
            'distance_stats': {
                'mean': self.processed_data['distance_km'].mean(),
                'std': self.processed_data['distance_km'].std(),
                'min': self.processed_data['distance_km'].min(),
                'max': self.processed_data['distance_km'].max()
            }
        }
        
        return report