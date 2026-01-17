"""
Visualization module for anomaly detection results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import logging
from pathlib import Path
from config.settings import VISUALIZATION_PARAMS, OUTPUTS_DIR

logger = logging.getLogger(__name__)


class AnomalyVisualizer:
    """Handles visualization of anomaly detection results"""
    
    def __init__(self, params: Dict = None):
        """
        Initialize AnomalyVisualizer
        
        Args:
            params: Visualization parameters
        """
        self.params = params or VISUALIZATION_PARAMS
        plt.style.use(self.params['style'])
        
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                  feature_columns: List[str],
                                  save_path: Optional[Path] = None):
        """
        Plot distributions of features with anomaly highlights
        
        Args:
            df: DataFrame with features and anomaly flags
            feature_columns: List of feature columns to plot
            save_path: Optional path to save the figure
        """
        n_features = len(feature_columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(self.params['figsize'][0], 
                                        n_rows * 4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(feature_columns):
            if idx < len(axes) and feature in df.columns:
                ax = axes[idx]
                
                # Plot normal points
                normal_data = df[df['anomaly_combined'] == 0][feature].dropna()
                ax.hist(normal_data, bins=50, alpha=0.7, 
                       label='Normal', color='blue', density=True)
                
                # Plot anomaly points
                anomaly_data = df[df['anomaly_combined'] == 1][feature].dropna()
                if len(anomaly_data) > 0:
                    ax.hist(anomaly_data, bins=50, alpha=0.7, 
                           label='Anomaly', color='red', density=True)
                
                ax.set_title(f'Distribution: {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(feature_columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.params['dpi'], bbox_inches='tight')
            logger.info(f"Saved feature distributions to {save_path}")
        
        plt.show()
    
    def plot_anomaly_scatter(self, df: pd.DataFrame, 
                            x_feature: str, y_feature: str,
                            save_path: Optional[Path] = None):
        """
        Create scatter plot with anomaly highlights
        
        Args:
            df: DataFrame with features and anomaly flags
            x_feature: Feature for x-axis
            y_feature: Feature for y-axis
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.params['figsize'])
        
        # Plot normal points
        normal_mask = df['anomaly_combined'] == 0
        ax.scatter(df.loc[normal_mask, x_feature], 
                  df.loc[normal_mask, y_feature],
                  alpha=0.6, s=20, label='Normal', color='blue')
        
        # Plot anomaly points
        anomaly_mask = df['anomaly_combined'] == 1
        if anomaly_mask.any():
            ax.scatter(df.loc[anomaly_mask, x_feature], 
                      df.loc[anomaly_mask, y_feature],
                      alpha=0.8, s=50, label='Anomaly', color='red',
                      edgecolors='black', linewidth=1)
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Anomaly Detection: {x_feature} vs {y_feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.params['dpi'], bbox_inches='tight')
            logger.info(f"Saved anomaly scatter plot to {save_path}")
        
        plt.show()
    
    def plot_anomaly_timeline(self, df: pd.DataFrame, 
                             save_path: Optional[Path] = None):
        """
        Plot anomalies over time
        
        Args:
            df: DataFrame with timestamp and anomaly flags
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Anomalies by hour of day
        if 'hour_of_day' in df.columns:
            hourly_anomalies = df.groupby('hour_of_day')['anomaly_combined'].mean() * 100
            
            ax1.bar(hourly_anomalies.index, hourly_anomalies.values,
                   color='skyblue', edgecolor='black')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Anomaly Rate (%)')
            ax1.set_title('Anomaly Rate by Hour of Day')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_xticks(range(0, 24, 2))
        
        # Plot 2: Anomalies by day of week
        if 'day_of_week' in df.columns:
            weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_anomalies = df.groupby('day_of_week')['anomaly_combined'].mean() * 100
            
            ax2.bar(range(len(weekday_names)), daily_anomalies.values,
                   color='lightcoral', edgecolor='black')
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Anomaly Rate (%)')
            ax2.set_title('Anomaly Rate by Day of Week')
            ax2.set_xticks(range(len(weekday_names)))
            ax2.set_xticklabels(weekday_names)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.params['dpi'], bbox_inches='tight')
            logger.info(f"Saved anomaly timeline to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                feature_columns: List[str],
                                save_path: Optional[Path] = None):
        """
        Plot correlation heatmap of features
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to include
            save_path: Optional path to save the figure
        """
        # Select only numeric features
        numeric_features = [f for f in feature_columns if f in df.columns and 
                          pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.warning("Not enough numeric features for correlation heatmap")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})
        
        ax.set_title('Feature Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.params['dpi'], bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {save_path}")
        
        plt.show()
    
    def plot_anomaly_summary(self, analysis: Dict, 
                            save_path: Optional[Path] = None):
        """
        Plot summary of anomaly detection results
        
        Args:
            analysis: Dictionary with anomaly analysis
            save_path: Optional path to save the figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Overall anomaly distribution
        labels = ['Normal', 'Anomalies']
        sizes = [
            analysis['total_trips'] - analysis['combined_anomalies'],
            analysis['combined_anomalies']
        ]
        colors = ['lightblue', 'lightcoral']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0, 0.1))
        ax1.set_title('Overall Anomaly Distribution')
        
        # Plot 2: Comparison of detection methods
        methods = ['Statistical', 'ML', 'Both']
        counts = [
            analysis['statistical_anomalies'],
            analysis['ml_anomalies'],
            analysis['severe_anomalies']
        ]
        
        ax2.bar(methods, counts, color=['skyblue', 'lightgreen', 'salmon'])
        ax2.set_ylabel('Number of Trips')
        ax2.set_title('Anomalies by Detection Method')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            ax2.text(i, v + max(counts)*0.01, str(v), 
                    ha='center', va='bottom')
        
        # Plot 3: Anomaly rate
        ax3.bar(['Anomaly Rate'], [analysis['anomaly_rate']], 
               color='orange')
        ax3.set_ylabel('Rate (%)')
        ax3.set_title(f'Overall Anomaly Rate: {analysis["anomaly_rate"]:.2f}%')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, max(analysis['anomaly_rate'] * 2, 10))
        
        # Plot 4: Feature contributions (top 5)
        if 'feature_contributions' in analysis:
            feature_contrib = analysis['feature_contributions']
            top_features = sorted(feature_contrib.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:5]
            
            feature_names = [f[0] for f in top_features]
            feature_counts = [f[1] for f in top_features]
            
            ax4.barh(range(len(feature_names)), feature_counts, 
                    color='lightseagreen')
            ax4.set_yticks(range(len(feature_names)))
            ax4.set_yticklabels(feature_names)
            ax4.set_xlabel('Number of Anomalies')
            ax4.set_title('Top 5 Feature Contributions to Anomalies')
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.params['dpi'], bbox_inches='tight')
            logger.info(f"Saved anomaly summary to {save_path}")
        
        plt.show()