# Fleet Drift - Vehicle Trip Anomaly Detection System

## Overview
A comprehensive system for detecting anomalous vehicle trips using time-series feature engineering and unsupervised anomaly detection techniques. The system analyzes GPS telemetry data to identify abnormal driving patterns, sensor faults, traffic disruptions, and data inconsistencies in fleet operations.

## Features
- **Multi-Method Detection**: Combines statistical (Z-score/IQR) and machine learning (Isolation Forest) approaches
- **Comprehensive Feature Engineering**: Transforms raw telemetry into 15+ trip-level features including speed statistics, distance metrics, and temporal patterns
- **Fleet Health Analytics**: Provides vehicle-level insights and operational recommendations
- **Explainable Results**: Transparent anomaly explanations with feature contribution analysis
- **Production-Ready Pipeline**: Complete workflow from data ingestion to visualization
- **Scalable Architecture**: Modular design suitable for large datasets

## Project Structure
```
vehicle_anomaly_detection/
├── config/                          # Configuration settings
│   └── settings.py
├── data/                           # Data storage
│   ├── raw/                        # Raw input data
│   ├── processed/                  # Cleaned and feature data
│   └── outputs/                    # Results, reports, visualizations
├── notebooks/                      # Jupyter notebooks for EDA
│   └── 01_exploratory_analysis.ipynb
├── src/                           # Core source code
│   ├── data_ingestion.py          # Data loading and validation
│   ├── data_preprocessing.py      # Data cleaning and preparation
│   ├── feature_engineering.py     # Feature extraction and transformation
│   ├── anomaly_detection.py       # Statistical and ML anomaly detection
│   ├── visualization.py           # Results visualization
│   └── fleet_insights.py          # Fleet-level analytics
├── tests/                         # Unit tests
│   ├── test_preprocessing.py
│   └── test_detection.py
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project
2. Create and activate virtual environment:
```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For development mode (optional):
```bash
pip install -e .
```

## Quick Start

### 1. Prepare Your Data
Place your trip data in CSV format at `data/raw/porto_taxi_trajectories.csv`. The expected format includes:
- `taxi_id`: Vehicle identifier
- `trajectory_id`: Trip identifier
- `timestamp`: Start time of trip
- `source_point`: Starting location as "POINT(lon lat)"
- `target_point`: Destination location as "POINT(lon lat)"

*Note: Sample data is provided in the problem statement. For real-world use, you can use the Porto taxi dataset or similar GPS trajectory data.*

### 2. Run the Complete Pipeline
```bash
python main.py
```

This will execute all steps:
- Data ingestion and validation
- Cleaning and preprocessing
- Feature engineering
- Anomaly detection (statistical + ML)
- Visualization generation
- Fleet insights analysis

### 3. Explore Results
Check the output directory for results:
- `data/outputs/anomaly_results.csv`: All trips with anomaly flags
- `data/outputs/fleet_insights_report.txt`: Summary of fleet health
- `data/outputs/final_summary.json`: Complete pipeline summary
- Various PNG files: Visualizations of distributions, correlations, and patterns

## Usage Examples

### Run Exploratory Analysis
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Use Individual Components
```python
from src.data_ingestion import DataIngestion
from src.anomaly_detection import AnomalyDetector

# Load and validate data
ingestor = DataIngestion()
data = ingestor.load_data()

# Detect anomalies
detector = AnomalyDetector()
results = detector.detect_combined_anomalies(data, ['distance_km', 'speed_kmh'])
```

### Customize Configuration
Edit `config/settings.py` to adjust:
- Feature engineering thresholds
- Anomaly detection parameters
- Visualization settings
- File paths

## Detection Methods

### 1. Statistical Methods
- **Z-score Analysis**: Identifies extreme values (±3 standard deviations)
- **Interquartile Range (IQR)**: Detects outliers beyond 1.5×IQR from quartiles

### 2. Machine Learning
- **Isolation Forest**: Unsupervised ensemble method that isolates anomalies
- **Combined Scoring**: Integrates statistical and ML results for robust detection

### 3. Feature Engineering
The system extracts comprehensive trip features:
- **Distance Metrics**: Total distance, lat/lon changes, normalized coordinates
- **Speed Statistics**: Estimated speed, speed categories, z-scores
- **Temporal Features**: Hour of day, day of week, weekend flags
- **Geographic Features**: Direction angles, location clusters

## Output Interpretation

### Anomaly Types
- **Normal**: No anomalies detected
- **Anomaly**: Flagged by either statistical or ML method
- **Severe Anomaly**: Flagged by both methods (high confidence)

### Key Metrics
- **Anomaly Rate**: Percentage of trips flagged as anomalies
- **Feature Contributions**: Which features most frequently trigger anomalies
- **Temporal Patterns**: Time periods with highest anomaly rates
- **Vehicle Risk Scores**: Individual vehicle anomaly frequencies

### Fleet Insights
The system identifies:
- Vehicles with consistently high anomaly rates
- Peak anomaly hours and days
- Common anomaly patterns across the fleet
- Operational recommendations for maintenance and scheduling

## Testing

Run the test suite to ensure functionality:
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_detection.py -v

# Run with coverage report
pytest --cov=src tests/
```

## Configuration Options

Key parameters in `config/settings.py`:

### Feature Engineering
```python
FEATURE_PARAMS = {
    'speed_threshold_kmh': 5,      # Below this speed = idle
    'max_speed_kmh': 120,          # Maximum reasonable speed
    'min_trip_duration_min': 1,    # Minimum valid trip duration
    'max_trip_duration_hours': 24, # Maximum valid trip duration
}
```

### Anomaly Detection
```python
ANOMALY_PARAMS = {
    'z_score_threshold': 3,        # Standard deviations for Z-score
    'iqr_multiplier': 1.5,         # IQR multiplier for outlier detection
    'isolation_forest_contamination': 0.1,  # Expected anomaly proportion
    'isolation_forest_n_estimators': 100,   # Number of trees in ensemble
}
```

## Dependencies
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Geospatial**: geopy
- **Development**: jupyter, pytest, tqdm

See `requirements.txt` for complete version specifications.

## Troubleshooting

### Common Issues

1. **Missing Data File**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/porto_taxi_trajectories.csv'
   ```
   **Solution**: Ensure your data file exists in the correct location or update the path in `config/settings.py`

2. **Memory Issues with Large Datasets**
   **Solution**: Process data in chunks or increase system memory

3. **Import Errors**
   **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

4. **Visualization Display Issues**
   **Solution**: For headless environments, use `plt.savefig()` instead of `plt.show()`

### Logging
The system uses Python's logging module. Adjust log levels in `main.py`:
```python
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more details
```
