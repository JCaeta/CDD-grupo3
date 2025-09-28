"""
Person 2: Lighting System Analysis
Responsible for: lights column

This module handles the analysis and cleaning of lighting system data.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

class LightingAnalyzer:
    """
    Analyzer for lighting system data: lights column.
    """
    
    def __init__(self, data):
        """
        Initialize the lighting analyzer.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data.copy()
        self.columns = ['lights']
        self.log = []
        
    def exploratory_analysis(self):
        """
        Perform exploratory analysis on lighting data.
        """
        st.write("### Lighting System Exploratory Analysis")
        
        if 'lights' in self.data.columns:
            st.write("**Lights Column Analysis:**")
            
            # Basic statistics
            lights_stats = self.data['lights'].describe()
            st.write("Basic Statistics:")
            st.write(lights_stats)
            
            # Value counts and unique values
            unique_values = self.data['lights'].unique()
            st.write(f"- Unique values: {sorted(unique_values)}")
            st.write(f"- Number of unique values: {len(unique_values)}")
            
            # Value distribution
            value_counts = self.data['lights'].value_counts().head(10)
            st.write("Top 10 most common values:")
            st.write(value_counts)
            
            # Check for missing values
            missing_count = self.data['lights'].isna().sum()
            if missing_count > 0:
                st.warning(f"Found {missing_count} missing values ({missing_count/len(self.data)*100:.2f}%)")
                self.log_action(f"Missing values detected: {missing_count}", "WARNING")
            else:
                st.success("No missing values found")
            
            # Check for negative values
            negative_count = (self.data['lights'] < 0).sum()
            if negative_count > 0:
                st.warning(f"Found {negative_count} negative values")
                self.log_action(f"Negative values detected: {negative_count}", "WARNING")
            
            # Check data type
            st.write(f"- Data type: {self.data['lights'].dtype}")
            
            # Statistical tests for normality
            if len(self.data['lights'].dropna()) > 3:
                shapiro_stat, shapiro_p = stats.shapiro(self.data['lights'].dropna()[:5000])  # Limit for performance
                st.write(f"- Shapiro-Wilk normality test p-value: {shapiro_p:.6f}")
                if shapiro_p < 0.05:
                    st.write("  → Data is not normally distributed")
                else:
                    st.write("  → Data appears to be normally distributed")
            
            # Outlier detection
            Q1 = self.data['lights'].quantile(0.25)
            Q3 = self.data['lights'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data['lights'] < lower_bound) | 
                               (self.data['lights'] > upper_bound)]
            
            st.write(f"- Outliers detected (IQR method): {len(outliers)} ({len(outliers)/len(self.data)*100:.2f}%)")
            self.log_action(f"Outliers detected: {len(outliers)}")
            
            # Check for discrete vs continuous nature
            if len(unique_values) <= 20:
                st.write("- Data appears to be discrete (few unique values)")
                self.log_action("Lights data identified as discrete")
            else:
                st.write("- Data appears to be continuous")
                self.log_action("Lights data identified as continuous")
    
    def visualize_data(self):
        """
        Create visualizations for lighting data.
        """
        st.write("### Lighting Data Visualizations")
        
        if 'lights' in self.data.columns:
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Histogram
            self.data['lights'].hist(bins=30, ax=axes[0,0], alpha=0.7, color='gold', edgecolor='black')
            axes[0,0].set_title('Distribution of Lights Values')
            axes[0,0].set_xlabel('Lights (Wh)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Box plot
            self.data.boxplot(column='lights', ax=axes[0,1])
            axes[0,1].set_title('Lights Box Plot')
            axes[0,1].set_ylabel('Lights (Wh)')
            
            # 3. Value counts bar plot (for discrete values)
            unique_values = self.data['lights'].unique()
            if len(unique_values) <= 20:
                value_counts = self.data['lights'].value_counts().head(15)
                value_counts.plot(kind='bar', ax=axes[1,0], color='lightblue')
                axes[1,0].set_title('Lights Value Frequency')
                axes[1,0].set_xlabel('Lights Value')
                axes[1,0].set_ylabel('Count')
                axes[1,0].tick_params(axis='x', rotation=45)
            else:
                # For continuous data, show a density plot
                self.data['lights'].plot(kind='density', ax=axes[1,0], color='green')
                axes[1,0].set_title('Lights Density Plot')
                axes[1,0].set_xlabel('Lights (Wh)')
            
            # 4. Time series plot (if date column exists)
            if 'date' in self.data.columns:
                # Sample data for performance if dataset is large
                sample_size = min(1000, len(self.data))
                sample_data = self.data.sample(n=sample_size).sort_values('date')
                
                axes[1,1].plot(sample_data['date'], sample_data['lights'], 
                              alpha=0.7, linewidth=0.8, color='purple')
                axes[1,1].set_title('Lights Over Time (Sample)')
                axes[1,1].set_xlabel('Date')
                axes[1,1].set_ylabel('Lights (Wh)')
                axes[1,1].tick_params(axis='x', rotation=45)
            else:
                # Q-Q plot for normality assessment
                stats.probplot(self.data['lights'].dropna(), dist="norm", plot=axes[1,1])
                axes[1,1].set_title('Q-Q Plot (Normal Distribution)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Additional analysis if temporal data is available
            if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
                # Hourly patterns
                if 'hour' in self.data.columns or pd.api.types.is_datetime64_any_dtype(self.data['date']):
                    hour_col = 'hour' if 'hour' in self.data.columns else self.data['date'].dt.hour
                    hourly_avg = self.data.groupby(hour_col)['lights'].mean()
                    
                    fig2, ax = plt.subplots(figsize=(12, 6))
                    hourly_avg.plot(kind='line', ax=ax, marker='o', color='orange', linewidth=2)
                    ax.set_title('Average Lights Usage by Hour of Day')
                    ax.set_xlabel('Hour')
                    ax.set_ylabel('Average Lights (Wh)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    
                    # Find peak usage hours
                    peak_hour = hourly_avg.idxmax()
                    min_hour = hourly_avg.idxmin()
                    st.write(f"Peak lights usage at hour: {peak_hour}")
                    st.write(f"Minimum lights usage at hour: {min_hour}")
    
    def clean_data(self):
        """
        Clean the lighting data based on analysis findings.
        """
        st.write("### Lighting Data Cleaning")
        
        initial_rows = len(self.data)
        
        if 'lights' in self.data.columns:
            # Handle missing values
            missing_count = self.data['lights'].isna().sum()
            if missing_count > 0:
                # Strategy 1: Forward fill (assuming lights tend to stay on/off for periods)
                self.data['lights'].fillna(method='ffill', inplace=True)
                
                # If still missing values, backward fill
                remaining_missing = self.data['lights'].isna().sum()
                if remaining_missing > 0:
                    self.data['lights'].fillna(method='bfill', inplace=True)
                
                # If still missing, use median
                final_missing = self.data['lights'].isna().sum()
                if final_missing > 0:
                    median_value = self.data['lights'].median()
                    self.data['lights'].fillna(median_value, inplace=True)
                
                self.log_action(f"Filled {missing_count} missing values using forward/backward fill and median")
            
            # Handle negative values (lights energy cannot be negative)
            negative_mask = self.data['lights'] < 0
            if negative_mask.sum() > 0:
                # Option 1: Set to 0 (assuming negative means lights off)
                self.data.loc[negative_mask, 'lights'] = 0
                self.log_action(f"Set {negative_mask.sum()} negative values to 0")
            
            # Data validation - ensure reasonable range
            max_reasonable = self.data['lights'].quantile(0.99)  # 99th percentile
            extreme_high = self.data['lights'] > max_reasonable * 10  # Values 10x higher than 99th percentile
            
            if extreme_high.sum() > 0:
                st.write(f"Found {extreme_high.sum()} extremely high values (>{max_reasonable * 10:.1f})")
                # Cap extreme values
                self.data.loc[extreme_high, 'lights'] = max_reasonable
                self.log_action(f"Capped {extreme_high.sum()} extreme values to {max_reasonable:.1f}")
            
            # Convert to appropriate data type if needed
            if self.data['lights'].dtype == 'object':
                try:
                    self.data['lights'] = pd.to_numeric(self.data['lights'], errors='coerce')
                    self.log_action("Converted lights column to numeric type")
                except:
                    self.log_action("Failed to convert lights to numeric", "WARNING")
        
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            st.write(f"Removed {removed_rows} rows during cleaning ({removed_rows/initial_rows*100:.2f}%)")
        else:
            st.write("No rows were removed during cleaning")
    
    def feature_engineering(self):
        """
        Create additional features related to lighting.
        """
        st.write("### Lighting Feature Engineering")
        
        if 'lights' in self.data.columns:
            # Create categorical features
            # Lights intensity categories
            lights_quantiles = self.data['lights'].quantile([0.25, 0.5, 0.75])
            
            conditions = [
                self.data['lights'] == 0,  # Off
                (self.data['lights'] > 0) & (self.data['lights'] <= lights_quantiles[0.25]),  # Low
                (self.data['lights'] > lights_quantiles[0.25]) & (self.data['lights'] <= lights_quantiles[0.75]),  # Medium
                self.data['lights'] > lights_quantiles[0.75]  # High
            ]
            choices = ['Off', 'Low', 'Medium', 'High']
            self.data['lights_category'] = np.select(conditions, choices, default='Unknown')
            
            # Binary feature: lights on/off
            self.data['lights_on'] = (self.data['lights'] > 0).astype(int)
            
            # Lights efficiency (if appliances data is available)
            if 'Appliances' in self.data.columns:
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.data['lights_to_appliances_ratio'] = np.where(
                        self.data['Appliances'] > 0,
                        self.data['lights'] / self.data['Appliances'],
                        0
                    )
                    # Cap extreme ratios
                    self.data['lights_to_appliances_ratio'] = np.clip(
                        self.data['lights_to_appliances_ratio'], 0, 1
                    )
            
            # Moving averages for trend analysis
            if len(self.data) > 12:  # Need enough data points
                self.data['lights_ma_6'] = self.data['lights'].rolling(window=6, min_periods=1).mean()
                self.data['lights_ma_12'] = self.data['lights'].rolling(window=12, min_periods=1).mean()
                
                # Deviation from moving average
                self.data['lights_dev_ma6'] = self.data['lights'] - self.data['lights_ma_6']
            
            new_features = ['lights_category', 'lights_on']
            
            if 'Appliances' in self.data.columns:
                new_features.append('lights_to_appliances_ratio')
            
            if len(self.data) > 12:
                new_features.extend(['lights_ma_6', 'lights_ma_12', 'lights_dev_ma6'])
            
            st.write(f"Created {len(new_features)} new lighting features:")
            for feature in new_features:
                st.write(f"- {feature}")
            
            self.log_action(f"Created lighting features: {', '.join(new_features)}")
            
            # Show feature distribution
            if 'lights_category' in self.data.columns:
                st.write("Lights category distribution:")
                st.write(self.data['lights_category'].value_counts())
    
    def analyze_and_clean(self):
        """
        Run the complete analysis and cleaning process.
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        self.exploratory_analysis()
        self.visualize_data()
        self.clean_data()
        self.feature_engineering()
        
        return self.data
    
    def get_log(self):
        """Return the processing log."""
        return self.log
    
    def log_action(self, message, level="INFO"):
        """Log an action with timestamp."""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'level': level,
            'message': f"[Person 2 - Lighting] {message}",
            'person': 2,
            'columns': self.columns
        }
        self.log.append(log_entry)