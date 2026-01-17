"""
Configuration settings for the anomaly detection system
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file
RAW_DATA_FILE = RAW_DATA_DIR / "porto_taxi_trajectories.csv"

# Feature engineering parameters
FEATURE_PARAMS = {
    'speed_threshold_kmh': 5,  # Speed below which vehicle is considered idle
    'max_speed_kmh': 120,  # Maximum reasonable speed
    'min_trip_duration_min': 1,  # Minimum valid trip duration
    'max_trip_duration_hours': 24,  # Maximum valid trip duration
}

# Anomaly detection parameters
ANOMALY_PARAMS = {
    'z_score_threshold': 3,
    'iqr_multiplier': 1.5,
    'isolation_forest_contamination': 0.1,
    'isolation_forest_n_estimators': 100,
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
}