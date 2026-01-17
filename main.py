"""
Main execution script for the anomaly detection system
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.settings import OUTPUTS_DIR, PROCESSED_DATA_DIR
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.anomaly_detection import AnomalyDetector
from src.visualization import AnomalyVisualizer
from src.fleet_insights import FleetInsightsAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    logger.info("Starting Vehicle Trip Anomaly Detection System")
    
    # Create output directories
    OUTPUTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    
    try:
        # Step 1: Data Ingestion
        logger.info("\n" + "="*60)
        logger.info("Step 1: Data Ingestion")
        logger.info("="*60)
        
        ingestor = DataIngestion()
        raw_data = ingestor.load_data()
        
        is_valid, validation_report = ingestor.validate_data()
        if not is_valid:
            logger.error("Data validation failed")
            return
        
        summary = ingestor.get_summary_statistics()
        logger.info(f"Data Summary: {summary}")
        
        # Step 2: Data Preprocessing
        logger.info("\n" + "="*60)
        logger.info("Step 2: Data Preprocessing")
        logger.info("="*60)
        
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(raw_data)
        
        cleaning_report = preprocessor.get_cleaning_report()
        logger.info(f"Cleaning Report: {cleaning_report}")
        
        # Save cleaned data
        cleaned_data_path = PROCESSED_DATA_DIR / "cleaned_trips.csv"
        cleaned_data.to_csv(cleaned_data_path, index=False)
        logger.info(f"Saved cleaned data to {cleaned_data_path}")
        
        # Step 3: Feature Engineering
        logger.info("\n" + "="*60)
        logger.info("Step 3: Feature Engineering")
        logger.info("="*60)
        
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.calculate_trip_features(cleaned_data)
        
        feature_descriptions = feature_engineer.get_feature_descriptions()
        feature_stats = feature_engineer.get_feature_statistics()
        
        logger.info(f"Created {len(features_df.columns)} features")
        logger.info(f"Feature descriptions: {list(feature_descriptions.keys())}")
        
        # Save features
        features_path = PROCESSED_DATA_DIR / "trip_features.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved features to {features_path}")
        
        # Step 4: Anomaly Detection
        logger.info("\n" + "="*60)
        logger.info("Step 4: Anomaly Detection")
        logger.info("="*60)
        
        # Select features for anomaly detection
        feature_columns = [
            'distance_km', 'lat_change', 'lon_change',
            'estimated_speed_kmh', 'hour_of_day',
            'distance_zscore', 'speed_zscore'
        ]
        
        # Filter to only include columns that exist
        feature_columns = [col for col in feature_columns if col in features_df.columns]
        
        detector = AnomalyDetector()
        anomaly_results = detector.detect_combined_anomalies(features_df, feature_columns)
        
        # Analyze anomalies
        anomaly_analysis = detector.get_anomaly_analysis()
        logger.info(f"Anomaly Analysis: {anomaly_analysis}")
        
        # Get top anomalous trips
        top_anomalies = detector.get_top_anomalous_trips(5)
        logger.info("\nTop 5 Anomalous Trips:")
        for idx, row in top_anomalies.iterrows():
            logger.info(f"  Trip {row['trajectory_id']}: Score={row['anomaly_score']:.3f}")
        
        # Save anomaly results
        anomaly_path = OUTPUTS_DIR / "anomaly_results.csv"
        anomaly_results.to_csv(anomaly_path, index=False)
        logger.info(f"Saved anomaly results to {anomaly_path}")
        
        # Step 5: Visualization
        logger.info("\n" + "="*60)
        logger.info("Step 5: Visualization")
        logger.info("="*60)
        
        visualizer = AnomalyVisualizer()
        
        # Plot feature distributions
        vis_path = OUTPUTS_DIR / "feature_distributions.png"
        visualizer.plot_feature_distributions(
            anomaly_results, 
            feature_columns[:6],  # First 6 features
            save_path=vis_path
        )
        
        # Plot scatter plot
        if 'distance_km' in anomaly_results.columns and 'estimated_speed_kmh' in anomaly_results.columns:
            vis_path = OUTPUTS_DIR / "anomaly_scatter.png"
            visualizer.plot_anomaly_scatter(
                anomaly_results,
                x_feature='distance_km',
                y_feature='estimated_speed_kmh',
                save_path=vis_path
            )
        
        # Plot timeline
        vis_path = OUTPUTS_DIR / "anomaly_timeline.png"
        visualizer.plot_anomaly_timeline(anomaly_results, save_path=vis_path)
        
        # Plot correlation heatmap
        vis_path = OUTPUTS_DIR / "correlation_heatmap.png"
        visualizer.plot_correlation_heatmap(
            anomaly_results,
            feature_columns,
            save_path=vis_path
        )
        
        # Plot summary
        vis_path = OUTPUTS_DIR / "anomaly_summary.png"
        visualizer.plot_anomaly_summary(anomaly_analysis, save_path=vis_path)
        
        # Step 6: Fleet Insights
        logger.info("\n" + "="*60)
        logger.info("Step 6: Fleet Insights")
        logger.info("="*60)
        
        insights_analyzer = FleetInsightsAnalyzer()
        fleet_insights = insights_analyzer.analyze_fleet_health(anomaly_results)
        
        # Generate insights report
        insights_report = insights_analyzer.generate_insights_report()
        logger.info("\n" + insights_report)
        
        # Get recommendations
        recommendations = insights_analyzer.get_vehicle_recommendations()
        logger.info("\nFleet Recommendations:")
        for rec in recommendations['fleet_level']:
            logger.info(f"  - {rec}")
        
        # Save insights report
        report_path = OUTPUTS_DIR / "fleet_insights_report.txt"
        with open(report_path, 'w') as f:
            f.write(insights_report)
        logger.info(f"Saved insights report to {report_path}")
        
        # Step 7: Save final results
        logger.info("\n" + "="*60)
        logger.info("Step 7: Final Results Summary")
        logger.info("="*60)
        
        final_summary = {
            'data_ingestion': {
                'raw_records': len(raw_data),
                'validated': is_valid,
                'taxi_count': summary['taxi_count'],
                'trajectory_count': summary['trajectory_count'],
            },
            'preprocessing': {
                'cleaned_records': len(cleaned_data),
                'removed_records': len(raw_data) - len(cleaned_data),
            },
            'feature_engineering': {
                'features_created': len(feature_descriptions),
                'trips_analyzed': len(features_df),
            },
            'anomaly_detection': anomaly_analysis,
            'fleet_insights': {
                'problematic_vehicles': fleet_insights['problematic_vehicles']['count'],
                'overall_anomaly_rate': fleet_insights['overall_anomaly_rate'],
            }
        }
        
        # Save final summary
        import json
        summary_path = OUTPUTS_DIR / "final_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        logger.info("\n" + "="*60)
        logger.info("ANOMALY DETECTION SYSTEM COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"\nResults saved to: {OUTPUTS_DIR}")
        logger.info(f"Final summary saved to: {summary_path}")
        
        return final_summary
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        logger.info("\nSummary of results:")
        logger.info(f"Total trips analyzed: {results['feature_engineering']['trips_analyzed']}")
        logger.info(f"Anomalies detected: {results['anomaly_detection']['combined_anomalies']}")
        logger.info(f"Anomaly rate: {results['anomaly_detection']['anomaly_rate']:.2f}%")
    else:
        logger.error("Anomaly detection failed")