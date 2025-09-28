"""
Person 1: Temporal Data Analysis
Responsible for: Appliances, date columns

This module handles the analysis and cleaning of temporal data including
appliance energy consumption and datetime information.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class TemporalDataAnalyzer:
    """
    Analyzer for temporal data columns: Appliances and date.
    """
    
    def __init__(self, data):
        """
        Initialize the temporal data analyzer.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data.copy()
        self.columns = ['date', 'Appliances']
        self.log = []
        
    def exploratory_analysis(self):
        """
        Perform exploratory analysis on temporal data.
        """
        st.write("### Temporal Data Exploratory Analysis")
        
        # Date column analysis
        if 'date' in self.data.columns:
            st.write("**Date Column Analysis:**")
            
            # Convert date column
            try:
                self.data['date'] = pd.to_datetime(self.data['date'], format='%d-%m-%Y %H:%M')
                self.log_action("Date column successfully converted to datetime")
                
                # Date range
                date_range = f"{self.data['date'].min()} to {self.data['date'].max()}"
                st.write(f"- Date range: {date_range}")
                
                # Check for gaps in timeline
                date_diff = self.data['date'].diff().dropna()
                expected_freq = pd.Timedelta(minutes=10)  # Assuming 10-minute intervals
                gaps = date_diff[date_diff > expected_freq]
                
                if len(gaps) > 0:
                    st.warning(f"Found {len(gaps)} gaps in the timeline")
                    self.log_action(f"Timeline gaps detected: {len(gaps)}", "WARNING")
                else:
                    st.success("No significant gaps in timeline")
                
            except Exception as e:
                st.error(f"Error processing date column: {str(e)}")
                self.log_action(f"Date conversion error: {str(e)}", "ERROR")
        
        # Appliances column analysis
        if 'Appliances' in self.data.columns:
            st.write("**Appliances Energy Consumption Analysis:**")
            
            # Basic statistics
            appliances_stats = self.data['Appliances'].describe()
            st.write("Basic Statistics:")
            st.write(appliances_stats)
            
            # Check for negative values
            negative_count = (self.data['Appliances'] < 0).sum()
            if negative_count > 0:
                st.warning(f"Found {negative_count} negative values in Appliances")
                self.log_action(f"Negative values in Appliances: {negative_count}", "WARNING")
            
            # Outlier detection using IQR method
            Q1 = self.data['Appliances'].quantile(0.25)
            Q3 = self.data['Appliances'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data['Appliances'] < lower_bound) | 
                               (self.data['Appliances'] > upper_bound)]
            
            st.write(f"- Outliers detected (IQR method): {len(outliers)} ({len(outliers)/len(self.data)*100:.2f}%)")
            self.log_action(f"Outliers in Appliances: {len(outliers)}")
    
    def visualize_data(self):
        """
        Create visualizations for temporal data.
        """
        st.write("### Temporal Data Visualizations")
        
        if 'date' in self.data.columns and 'Appliances' in self.data.columns:
            # Time series plot
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(self.data['date'], self.data['Appliances'], alpha=0.7, linewidth=0.5)
            ax1.set_title('Appliances Energy Consumption Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Energy (Wh)')
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            
            # Hourly patterns
            if pd.api.types.is_datetime64_any_dtype(self.data['date']):
                hourly_avg = self.data.groupby(self.data['date'].dt.hour)['Appliances'].mean()
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                hourly_avg.plot(kind='bar', ax=ax2, color='skyblue')
                ax2.set_title('Average Appliances Consumption by Hour of Day')
                ax2.set_xlabel('Hour')
                ax2.set_ylabel('Average Energy (Wh)')
                st.pyplot(fig2)
                
                # Daily patterns
                daily_avg = self.data.groupby(self.data['date'].dt.dayofweek)['Appliances'].mean()
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                daily_avg.plot(kind='bar', ax=ax3, color='lightgreen')
                ax3.set_title('Average Appliances Consumption by Day of Week')
                ax3.set_xlabel('Day of Week')
                ax3.set_ylabel('Average Energy (Wh)')
                ax3.set_xticklabels(day_names, rotation=45)
                st.pyplot(fig3)
        
        # Distribution plot
        if 'Appliances' in self.data.columns:
            fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            self.data['Appliances'].hist(bins=50, ax=ax4, alpha=0.7, color='orange')
            ax4.set_title('Appliances Consumption Distribution')
            ax4.set_xlabel('Energy (Wh)')
            ax4.set_ylabel('Frequency')
            
            # Box plot
            self.data.boxplot(column='Appliances', ax=ax5)
            ax5.set_title('Appliances Consumption Box Plot')
            ax5.set_ylabel('Energy (Wh)')
            
            st.pyplot(fig4)
    
    def clean_data(self):
        """
        Clean the temporal data based on analysis findings.
        """
        st.write("### Temporal Data Cleaning")
        
        initial_rows = len(self.data)
        
        # Handle date column
        if 'date' in self.data.columns:
            # Ensure proper datetime format
            if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
                try:
                    self.data['date'] = pd.to_datetime(self.data['date'], format='%d-%m-%Y %H:%M')
                    self.log_action("Date column converted to datetime format")
                except Exception as e:
                    self.log_action(f"Date conversion failed: {str(e)}", "ERROR")
            
            # Remove rows with invalid dates
            invalid_dates = self.data['date'].isna()
            if invalid_dates.sum() > 0:
                self.data = self.data[~invalid_dates]
                self.log_action(f"Removed {invalid_dates.sum()} rows with invalid dates")
        
        # Handle Appliances column
        if 'Appliances' in self.data.columns:
            # Remove negative values (energy consumption cannot be negative)
            negative_mask = self.data['Appliances'] < 0
            if negative_mask.sum() > 0:
                self.data = self.data[~negative_mask]
                self.log_action(f"Removed {negative_mask.sum()} rows with negative appliance values")
            
            # Handle null values in Appliances
            null_appliances = self.data['Appliances'].isna()
            if null_appliances.sum() > 0:
                # Option 1: Remove rows with null values
                # self.data = self.data[~null_appliances]
                
                # Option 2: Impute with median (more conservative for energy data)
                median_value = self.data['Appliances'].median()
                self.data['Appliances'].fillna(median_value, inplace=True)
                self.log_action(f"Imputed {null_appliances.sum()} null values with median: {median_value}")
            
            # Handle extreme outliers (optional - be careful with energy data)
            Q1 = self.data['Appliances'].quantile(0.25)
            Q3 = self.data['Appliances'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more conservative outlier removal
            upper_bound = Q3 + 3 * IQR
            
            extreme_outliers = (self.data['Appliances'] < lower_bound) | (self.data['Appliances'] > upper_bound)
            if extreme_outliers.sum() > 0:
                st.write(f"Option: Remove {extreme_outliers.sum()} extreme outliers?")
                # For now, just log them
                self.log_action(f"Identified {extreme_outliers.sum()} extreme outliers (not removed)")
        
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            st.write(f"Removed {removed_rows} rows during cleaning ({removed_rows/initial_rows*100:.2f}%)")
            self.log_action(f"Total rows removed: {removed_rows}")
        else:
            st.write("No rows were removed during cleaning")
    
    def feature_engineering(self):
        """
        Create additional temporal features.
        """
        st.write("### Temporal Feature Engineering")
        
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            # Extract temporal components
            self.data['year'] = self.data['date'].dt.year
            self.data['month'] = self.data['date'].dt.month
            self.data['day'] = self.data['date'].dt.day
            self.data['hour'] = self.data['date'].dt.hour
            self.data['minute'] = self.data['date'].dt.minute
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['day_of_year'] = self.data['date'].dt.dayofyear
            self.data['week_of_year'] = self.data['date'].dt.isocalendar().week
            
            # Create categorical features
            self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
            
            # Time of day categories
            conditions = [
                (self.data['hour'] >= 6) & (self.data['hour'] < 12),   # Morning
                (self.data['hour'] >= 12) & (self.data['hour'] < 18),  # Afternoon
                (self.data['hour'] >= 18) & (self.data['hour'] < 22),  # Evening
            ]
            choices = ['Morning', 'Afternoon', 'Evening']
            self.data['time_of_day'] = np.select(conditions, choices, default='Night')
            
            # Season based on month
            season_conditions = [
                self.data['month'].isin([12, 1, 2]),   # Winter
                self.data['month'].isin([3, 4, 5]),    # Spring
                self.data['month'].isin([6, 7, 8]),    # Summer
                self.data['month'].isin([9, 10, 11])   # Fall
            ]
            season_choices = ['Winter', 'Spring', 'Summer', 'Fall']
            self.data['season'] = np.select(season_conditions, season_choices, default='Unknown')
            
            new_features = ['year', 'month', 'day', 'hour', 'minute', 'day_of_week', 
                          'day_of_year', 'week_of_year', 'is_weekend', 'time_of_day', 'season']
            
            st.write(f"Created {len(new_features)} new temporal features:")
            for feature in new_features:
                st.write(f"- {feature}")
            
            self.log_action(f"Created temporal features: {', '.join(new_features)}")
    
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
            'message': f"[Person 1 - Temporal] {message}",
            'person': 1,
            'columns': self.columns
        }
        self.log.append(log_entry)