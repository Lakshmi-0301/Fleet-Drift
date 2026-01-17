"""
Feature engineering module for creating trip-level features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
from geopy.distance import geodesic
from scipy import stats
from config.settings import FEATURE_PARAMS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for trip data"""
    
    def __init__(self, params: Dict = None):
        """
        Initialize FeatureEngineer
        
        Args:
            params: Feature engineering parameters
        """
        self.params = params or FEATURE_PARAMS
        self.features_df = None
        
    def calculate_trip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive trip features
        
        Args:
            df: Cleaned DataFrame with trip data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Group by trajectory to create trip-level features
        trip_features = []
        
        # Since we only have source and target points in sample data,
        # we'll simulate some intermediate points for feature calculation
        # In real implementation, you would have full trajectory data
        
        for traj_id, group in df.groupby('trajectory_id'):
            if len(group) == 1:  # Only source and target points
                trip_data = group.iloc[0]
                
                # Basic features from available data
                features = {
                    'trajectory_id': traj_id,
                    'taxi_id': trip_data['taxi_id'],
                    'timestamp': trip_data['timestamp'],
                    
                    # Distance features
                    'distance_km': trip_data['distance_km'],
                    
                    # Location features
                    'source_lon': trip_data['source_lon'],
                    'source_lat': trip_data['source_lat'],
                    'target_lon': trip_data['target_lon'],
                    'target_lat': trip_data['target_lat'],
                    
                    # Derived location features
                    'lat_change': abs(trip_data['target_lat'] - trip_data['source_lat']),
                    'lon_change': abs(trip_data['target_lon'] - trip_data['source_lon']),
                    
                    # Time features (extracted from timestamp)
                    'hour_of_day': trip_data['timestamp'].hour,
                    'day_of_week': trip_data['timestamp'].weekday(),
                    'is_weekend': 1 if trip_data['timestamp'].weekday() >= 5 else 0,
                    
                    # Speed features (estimated - in real data would be calculated from trajectory)
                    'estimated_speed_kmh': trip_data['distance_km'] * 60,  # Assuming 1 minute trips
                    
                    # Additional derived features
                    'direction_angle': np.arctan2(
                        trip_data['target_lat'] - trip_data['source_lat'],
                        trip_data['target_lon'] - trip_data['source_lon']
                    ),
                }
                
                trip_features.append(features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(trip_features)
        
        # Calculate additional features
        if len(features_df) > 0:
            # Normalize coordinates
            features_df['norm_source_lon'] = (
                features_df['source_lon'] - features_df['source_lon'].mean()
            ) / features_df['source_lon'].std()
            features_df['norm_source_lat'] = (
                features_df['source_lat'] - features_df['source_lat'].mean()
            ) / features_df['source_lat'].std()
            
            # Distance-based features
            features_df['distance_zscore'] = stats.zscore(features_df['distance_km'])
            features_df['distance_category'] = pd.cut(
                features_df['distance_km'],
                bins=[0, 1, 5, 10, 20, 50, 100],
                labels=['0-1km', '1-5km', '5-10km', '10-20km', '20-50km', '50-100km']
            )
            
            # Speed-based features
            features_df['speed_zscore'] = stats.zscore(features_df['estimated_speed_kmh'])
            features_df['speed_category'] = pd.cut(
                features_df['estimated_speed_kmh'],
                bins=[0, 10, 30, 60, 90, 120],
                labels=['0-10', '10-30', '30-60', '60-90', '90-120']
            )
            
            # Time-based features
            features_df['time_of_day'] = pd.cut(
                features_df['hour_of_day'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
        
        logger.info(f"Created features for {len(features_df)} trips")
        self.features_df = features_df
        return features_df
    
    def get_feature_descriptions(self) -> Dict:
        """
        Get descriptions of all engineered features
        
        Returns:
            Dictionary with feature descriptions
        """
        descriptions = {
            'distance_km': 'Total trip distance in kilometers',
            'lat_change': 'Absolute change in latitude',
            'lon_change': 'Absolute change in longitude',
            'hour_of_day': 'Hour when trip started (0-23)',
            'day_of_week': 'Day of week (0=Monday, 6=Sunday)',
            'is_weekend': 'Whether trip was on weekend (1) or weekday (0)',
            'estimated_speed_kmh': 'Estimated average speed in km/h',
            'direction_angle': 'Direction of travel in radians',
            'distance_zscore': 'Z-score normalized distance',
            'speed_zscore': 'Z-score normalized speed',
            'distance_category': 'Categorical distance range',
            'speed_category': 'Categorical speed range',
            'time_of_day': 'Categorical time period',
            'norm_source_lon': 'Normalized source longitude',
            'norm_source_lat': 'Normalized source latitude',
        }
        
        return descriptions
    
    def get_feature_statistics(self) -> Dict:
        """
        Get statistics of engineered features
        
        Returns:
            Dictionary with feature statistics
        """
        if self.features_df is None:
            raise ValueError("No features available. Run calculate_trip_features() first.")
            
        stats_dict = {}
        for column in self.features_df.select_dtypes(include=[np.number]).columns:
            stats_dict[column] = {
                'mean': self.features_df[column].mean(),
                'std': self.features_df[column].std(),
                'min': self.features_df[column].min(),
                'max': self.features_df[column].max(),
                'median': self.features_df[column].median(),
            }
            
        return stats_dict