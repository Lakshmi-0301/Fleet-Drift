"""
Fleet insights module for analyzing vehicle and fleet health
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FleetInsightsAnalyzer:
    """Analyzes fleet-level insights from anomaly detection results"""
    
    def __init__(self):
        """Initialize FleetInsightsAnalyzer"""
        self.insights = None
        
    def analyze_fleet_health(self, df: pd.DataFrame) -> Dict:
        """
        Analyze overall fleet health metrics
        
        Args:
            df: DataFrame with anomaly results
            
        Returns:
            Dictionary with fleet health insights
        """
        logger.info("Analyzing fleet health")
        
        if 'taxi_id' not in df.columns:
            raise ValueError("DataFrame must contain 'taxi_id' column")
        
        insights = {
            'total_vehicles': df['taxi_id'].nunique(),
            'total_trips': len(df),
            'overall_anomaly_rate': df['anomaly_combined'].mean() * 100,
        }
        
        # Vehicle-level analysis
        vehicle_stats = df.groupby('taxi_id').agg({
            'trajectory_id': 'count',
            'anomaly_combined': 'sum',
            'anomaly_score': 'mean'
        }).rename(columns={
            'trajectory_id': 'total_trips',
            'anomaly_combined': 'anomaly_count',
            'anomaly_score': 'avg_anomaly_score'
        })
        
        vehicle_stats['anomaly_rate'] = (
            vehicle_stats['anomaly_count'] / vehicle_stats['total_trips'] * 100
        )
        
        insights['vehicle_stats'] = vehicle_stats
        
        # Identify problematic vehicles
        high_anomaly_threshold = 50  # Vehicles with >50% anomaly rate
        problematic_vehicles = vehicle_stats[
            vehicle_stats['anomaly_rate'] > high_anomaly_threshold
        ]
        
        insights['problematic_vehicles'] = {
            'count': len(problematic_vehicles),
            'ids': problematic_vehicles.index.tolist(),
            'avg_anomaly_rate': problematic_vehicles['anomaly_rate'].mean(),
        }
        
        # Time-based patterns
        time_patterns = {}
        if 'hour_of_day' in df.columns:
            hourly_pattern = df.groupby('hour_of_day')['anomaly_combined'].mean() * 100
            peak_hours = hourly_pattern.nlargest(3)
            time_patterns['hourly'] = {
                'peak_hours': peak_hours.index.tolist(),
                'peak_rates': peak_hours.values.tolist(),
                'pattern': hourly_pattern.to_dict(),
            }
        
        if 'day_of_week' in df.columns:
            daily_pattern = df.groupby('day_of_week')['anomaly_combined'].mean() * 100
            time_patterns['daily'] = {
                'worst_day': daily_pattern.idxmax(),
                'worst_rate': daily_pattern.max(),
                'pattern': daily_pattern.to_dict(),
            }
        
        insights['time_patterns'] = time_patterns
        
        # Anomaly type distribution
        if 'anomaly_type' in df.columns:
            anomaly_dist = df['anomaly_type'].value_counts(normalize=True) * 100
            insights['anomaly_type_distribution'] = anomaly_dist.to_dict()
        
        # Feature impact analysis
        feature_impact = {}
        for col in df.columns:
            if col.startswith('anomaly_zscore_') or col.startswith('anomaly_iqr_'):
                feature_name = col.replace('anomaly_zscore_', '').replace('anomaly_iqr_', '')
                if feature_name not in feature_impact:
                    feature_impact[feature_name] = 0
                feature_impact[feature_name] += df[col].sum()
        
        insights['feature_impact'] = feature_impact
        
        self.insights = insights
        return insights
    
    def get_vehicle_recommendations(self, vehicle_id: str = None) -> Dict:
        """
        Get recommendations for specific vehicle or entire fleet
        
        Args:
            vehicle_id: Optional specific vehicle ID
            
        Returns:
            Dictionary with recommendations
        """
        if self.insights is None:
            raise ValueError("Run analyze_fleet_health() first")
        
        recommendations = {
            'fleet_level': [],
            'vehicle_level': {},
        }
        
        # Fleet-level recommendations
        overall_rate = self.insights['overall_anomaly_rate']
        
        if overall_rate > 20:
            recommendations['fleet_level'].append(
                "High overall anomaly rate detected. Consider fleet-wide maintenance check."
            )
        elif overall_rate > 10:
            recommendations['fleet_level'].append(
                "Moderate anomaly rate. Monitor fleet performance closely."
            )
        
        if self.insights['time_patterns'].get('hourly', {}).get('peak_rates', [0])[0] > 30:
            peak_hour = self.insights['time_patterns']['hourly']['peak_hours'][0]
            recommendations['fleet_level'].append(
                f"High anomaly rate during hour {peak_hour}:00. "
                f"Consider adjusting schedules or increasing supervision."
            )
        
        # Vehicle-specific recommendations
        if vehicle_id:
            vehicle_stats = self.insights['vehicle_stats']
            if vehicle_id in vehicle_stats.index:
                stats = vehicle_stats.loc[vehicle_id]
                
                vehicle_recs = []
                
                if stats['anomaly_rate'] > 50:
                    vehicle_recs.append(
                        f"Critical: Vehicle {vehicle_id} has {stats['anomaly_rate']:.1f}% "
                        f"anomaly rate. Immediate inspection recommended."
                    )
                elif stats['anomaly_rate'] > 30:
                    vehicle_recs.append(
                        f"Warning: Vehicle {vehicle_id} has {stats['anomaly_rate']:.1f}% "
                        f"anomaly rate. Schedule maintenance soon."
                    )
                
                if stats['avg_anomaly_score'] > 0.8:
                    vehicle_recs.append(
                        f"High anomaly severity detected. "
                        f"Review recent trips for safety concerns."
                    )
                
                recommendations['vehicle_level'][vehicle_id] = vehicle_recs
        
        return recommendations
    
    def generate_insights_report(self) -> str:
        """
        Generate a comprehensive insights report
        
        Returns:
            Formatted insights report
        """
        if self.insights is None:
            raise ValueError("Run analyze_fleet_health() first")
        
        report_lines = [
            "=" * 60,
            "FLEET HEALTH INSIGHTS REPORT",
            "=" * 60,
            f"\nTotal Vehicles: {self.insights['total_vehicles']}",
            f"Total Trips Analyzed: {self.insights['total_trips']:,}",
            f"Overall Anomaly Rate: {self.insights['overall_anomaly_rate']:.2f}%",
        ]
        
        # Problematic vehicles
        problematic = self.insights['problematic_vehicles']
        report_lines.extend([
            f"\n--- Problematic Vehicles ---",
            f"Count: {problematic['count']} vehicles",
            f"Average Anomaly Rate: {problematic['avg_anomaly_rate']:.1f}%",
        ])
        
        if problematic['ids']:
            report_lines.append(f"Vehicle IDs: {', '.join(map(str, problematic['ids'][:10]))}")
            if len(problematic['ids']) > 10:
                report_lines.append(f"... and {len(problematic['ids']) - 10} more")
        
        # Time patterns
        if 'time_patterns' in self.insights:
            report_lines.append("\n--- Time-Based Patterns ---")
            
            hourly = self.insights['time_patterns'].get('hourly', {})
            if hourly:
                report_lines.append(
                    f"Peak Anomaly Hours: {', '.join(map(str, hourly.get('peak_hours', [])))}:00"
                )
            
            daily = self.insights['time_patterns'].get('daily', {})
            if daily:
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                               'Friday', 'Saturday', 'Sunday']
                worst_day = daily.get('worst_day', 0)
                report_lines.append(
                    f"Worst Day: {weekday_names[worst_day]} "
                    f"({daily.get('worst_rate', 0):.1f}% anomaly rate)"
                )
        
        # Anomaly types
        if 'anomaly_type_distribution' in self.insights:
            report_lines.append("\n--- Anomaly Type Distribution ---")
            for anomaly_type, percentage in self.insights['anomaly_type_distribution'].items():
                report_lines.append(f"{anomaly_type}: {percentage:.1f}%")
        
        # Top contributing features
        if 'feature_impact' in self.insights:
            report_lines.append("\n--- Top Contributing Features ---")
            sorted_features = sorted(
                self.insights['feature_impact'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for feature, count in sorted_features:
                report_lines.append(f"{feature}: {count:,} anomalies")
        
        report_lines.extend([
            "\n" + "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])
        
        return "\n".join(report_lines)