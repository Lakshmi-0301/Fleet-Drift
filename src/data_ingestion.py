"""
Data ingestion module for loading and validating trip data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from config.settings import RAW_DATA_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data loading and initial validation"""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataIngestion
        
        Args:
            data_path: Path to raw data file
        """
        self.data_path = data_path or RAW_DATA_FILE
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the taxi trajectory data
        
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found at {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def validate_data(self) -> Tuple[bool, dict]:
        """
        Perform basic data validation
        
        Returns:
            Tuple of (is_valid, validation_report)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        report = {
            'total_records': len(self.data),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_records': self.data.duplicated().sum(),
            'data_types': self.data.dtypes.to_dict(),
        }
        
        # Check required columns
        required_columns = ['taxi_id', 'trajectory_id', 'timestamp', 
                           'source_point', 'target_point']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False, report
            
        # Check for empty dataset
        if len(self.data) == 0:
            logger.error("Dataset is empty")
            return False, report
            
        logger.info("Data validation completed")
        return True, report
    
    def get_summary_statistics(self) -> dict:
        """
        Get summary statistics of the loaded data
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        summary = {
            'taxi_count': self.data['taxi_id'].nunique(),
            'trajectory_count': self.data['trajectory_id'].nunique(),
            'date_range': {
                'min': self.data['timestamp'].min(),
                'max': self.data['timestamp'].max()
            },
            'columns': list(self.data.columns),
        }
        
        return summary