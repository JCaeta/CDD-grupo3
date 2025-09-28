"""
Person 5: External Meteorological Data (Part 1) Analysis
Responsible for: T_out, RH_out, Tdewpoint columns

This module handles the analysis and cleaning of external weather data part 1.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class WeatherPart1Analyzer:
    """
    Analyzer for external weather data part 1: T_out, RH_out, Tdewpoint columns.
    """
    
    def __init__(self, data):
        """
        Initialize the weather part 1 analyzer.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data.copy()
        self.columns = ['T_out', 'RH_out', 'Tdewpoint']
        self.available_columns = [col for col in self.columns if col in self.data.columns]
        self.log = []
        
    def exploratory_analysis(self):
        """
        Perform exploratory analysis on weather data part 1.
        """
        st.write("### External Weather Data (Part 1) Exploratory Analysis")
        
        if not self.available_columns:
            st.error("No weather part 1 columns found in dataset")
            return
        
        st.write(f"**Available weather variables:** {', '.join(self.available_columns)}")
        
        # Basic statistics
        weather_stats = self.data[self.available_columns].describe()
        st.write("**Weather Statistics:**")
        st.write(weather_stats)
        
        # Check for missing values
        missing_analysis = self.data[self.available_columns].isnull().sum()
        missing_percentage = (missing_analysis / len(self.data)) * 100
        
        if missing_analysis.sum() > 0:
            st.write("**Missing Values Analysis:**")
            missing_df = pd.DataFrame({
                'Missing_Count': missing_analysis,
                'Missing_Percentage': missing_percentage
            })
            st.write(missing_df[missing_df['Missing_Count'] > 0])
            self.log_action(f"Missing values found: {missing_analysis[missing_analysis > 0].to_dict()}")
        else:
            st.success("No missing values in weather data")
        
        # Validate weather data ranges
        st.write("**Weather Data Validation:**")
        
        # Temperature validation (T_out)
        if 'T_out' in self.available_columns:
            temp_out = self.data['T_out'].dropna()
            min_temp = temp_out.min()
            max_temp = temp_out.max()
            
            st.write(f"- T_out: {min_temp:.2f}°C to {max_temp:.2f}°C")
            
            # Flag extreme temperatures (outside typical range for most climates)
            extreme_cold = (temp_out < -30).sum()
            extreme_hot = (temp_out > 50).sum()
            
            if extreme_cold > 0:
                st.warning(f"  → {extreme_cold} extremely cold readings (< -30°C)")
                self.log_action(f"T_out: {extreme_cold} extremely cold readings", "WARNING")
            if extreme_hot > 0:
                st.warning(f"  → {extreme_hot} extremely hot readings (> 50°C)")
                self.log_action(f"T_out: {extreme_hot} extremely hot readings", "WARNING")
        
        # Humidity validation (RH_out)
        if 'RH_out' in self.available_columns:
            rh_out = self.data['RH_out'].dropna()
            min_rh = rh_out.min()
            max_rh = rh_out.max()
            
            st.write(f"- RH_out: {min_rh:.2f}% to {max_rh:.2f}%")
            
            # Flag invalid humidity values
            invalid_low = (rh_out < 0).sum()
            invalid_high = (rh_out > 100).sum()
            
            if invalid_low > 0:
                st.error(f"  → {invalid_low} invalid readings (< 0%)")
                self.log_action(f"RH_out: {invalid_low} invalid low readings", "ERROR")
            if invalid_high > 0:
                st.error(f"  → {invalid_high} invalid readings (> 100%)")
                self.log_action(f"RH_out: {invalid_high} invalid high readings", "ERROR")
        
        # Dew point validation (Tdewpoint)
        if 'Tdewpoint' in self.available_columns:
            dewpoint = self.data['Tdewpoint'].dropna()
            min_dew = dewpoint.min()
            max_dew = dewpoint.max()
            
            st.write(f"- Tdewpoint: {min_dew:.2f}°C to {max_dew:.2f}°C")
            
            # Dew point should always be <= air temperature
            if 'T_out' in self.available_columns:
                invalid_dew = (self.data['Tdewpoint'] > self.data['T_out']).sum()
                if invalid_dew > 0:
                    st.error(f"  → {invalid_dew} readings where dew point > air temperature (impossible)")
                    self.log_action(f"Tdewpoint: {invalid_dew} impossible readings (dewpoint > air temp)", "ERROR")
        
        # Weather relationships validation
        st.write("**Weather Relationships Analysis:**")
        
        if len(self.available_columns) > 1:
            # Correlation analysis
            correlation_matrix = self.data[self.available_columns].corr()
            st.write("Variable correlations:")
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    st.write(f"- {var1} ↔ {var2}: {corr_value:.3f}")
            
            # Expected relationships
            if 'T_out' in self.available_columns and 'Tdewpoint' in self.available_columns:
                temp_dew_corr = correlation_matrix.loc['T_out', 'Tdewpoint']
                if temp_dew_corr < 0.5:
                    st.warning(f"Low correlation between T_out and Tdewpoint ({temp_dew_corr:.3f}) - unusual")
                    self.log_action(f"Unusual T_out-Tdewpoint correlation: {temp_dew_corr:.3f}", "WARNING")
        
        # Outlier detection
        st.write("**Outlier Analysis:**")
        for col in self.available_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.data)) * 100
            
            if outlier_count > 0:
                st.write(f"- {col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
                self.log_action(f"{col}: {outlier_count} outliers detected")
        
        # Seasonal analysis (if date available)
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            st.write("**Seasonal Analysis:**")
            
            for col in self.available_columns:
                if col in ['T_out', 'Tdewpoint']:  # Temperature variables
                    monthly_avg = self.data.groupby(self.data['date'].dt.month)[col].mean()
                    temp_range = monthly_avg.max() - monthly_avg.min()
                    st.write(f"- {col} seasonal range: {temp_range:.2f}°C")
                elif col == 'RH_out':  # Humidity
                    monthly_avg = self.data.groupby(self.data['date'].dt.month)[col].mean()
                    humidity_range = monthly_avg.max() - monthly_avg.min()
                    st.write(f"- {col} seasonal range: {humidity_range:.2f}%")
    
    def visualize_data(self):
        """
        Create visualizations for weather data part 1.
        """
        st.write("### Weather Data (Part 1) Visualizations")
        
        if not self.available_columns:
            return
        
        # 1. Distribution plots
        n_cols = len(self.available_columns)
        fig1, axes1 = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        if n_cols == 1:
            axes1 = [axes1]
        
        colors = ['red', 'blue', 'green']
        
        for i, col in enumerate(self.available_columns):
            self.data[col].hist(bins=30, ax=axes1[i], alpha=0.7, 
                              color=colors[i % len(colors)], edgecolor='black')
            axes1[i].set_title(f'{col} Distribution')
            if 'T_out' in col or 'Tdewpoint' in col:
                axes1[i].set_xlabel('Temperature (°C)')
            elif 'RH_out' in col:
                axes1[i].set_xlabel('Relative Humidity (%)')
            axes1[i].set_ylabel('Frequency')
            axes1[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        # 2. Correlation heatmap (if multiple variables)
        if len(self.available_columns) > 1:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            correlation_matrix = self.data[self.available_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       fmt='.3f', ax=ax2, cbar_kws={'label': 'Correlation'})
            ax2.set_title('Weather Variables Correlation Matrix')
            st.pyplot(fig2)
        
        # 3. Box plots
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        self.data[self.available_columns].boxplot(ax=ax3)
        ax3.set_title('Weather Variables Box Plots')
        ax3.set_ylabel('Values')
        plt.xticks(rotation=45)
        st.pyplot(fig3)
        
        # 4. Time series plots (if date available)
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            # Sample data for performance
            sample_size = min(2000, len(self.data))
            sample_data = self.data.sample(n=sample_size).sort_values('date')
            
            fig4, axes4 = plt.subplots(len(self.available_columns), 1, figsize=(15, 4*len(self.available_columns)))
            
            if len(self.available_columns) == 1:
                axes4 = [axes4]
            
            for i, col in enumerate(self.available_columns):
                axes4[i].plot(sample_data['date'], sample_data[col], 
                             alpha=0.7, linewidth=0.8, color=colors[i % len(colors)])
                axes4[i].set_title(f'{col} Over Time (Sample Data)')
                axes4[i].set_xlabel('Date')
                if 'T_out' in col or 'Tdewpoint' in col:
                    axes4[i].set_ylabel('Temperature (°C)')
                elif 'RH_out' in col:
                    axes4[i].set_ylabel('Relative Humidity (%)')
                axes4[i].grid(True, alpha=0.3)
                plt.setp(axes4[i].xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig4)
            
            # 5. Seasonal patterns
            if len(self.data) > 100:  # Need sufficient data for seasonal analysis
                fig5, axes5 = plt.subplots(1, len(self.available_columns), figsize=(5*len(self.available_columns), 6))
                
                if len(self.available_columns) == 1:
                    axes5 = [axes5]
                
                for i, col in enumerate(self.available_columns):
                    monthly_avg = self.data.groupby(self.data['date'].dt.month)[col].mean()
                    monthly_avg.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_avg)]
                    
                    monthly_avg.plot(kind='line', ax=axes5[i], marker='o', 
                                   color=colors[i % len(colors)], linewidth=2, markersize=6)
                    axes5[i].set_title(f'{col} - Monthly Averages')
                    axes5[i].set_xlabel('Month')
                    if 'T_out' in col or 'Tdewpoint' in col:
                        axes5[i].set_ylabel('Temperature (°C)')
                    elif 'RH_out' in col:
                        axes5[i].set_ylabel('Relative Humidity (%)')
                    axes5[i].grid(True, alpha=0.3)
                    plt.setp(axes5[i].xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig5)
        
        # 6. Relationship plots
        if 'T_out' in self.available_columns and 'Tdewpoint' in self.available_columns:
            fig6, ax6 = plt.subplots(figsize=(8, 6))
            
            # Sample for performance
            sample_size = min(1000, len(self.data))
            sample_data = self.data.sample(n=sample_size)
            
            scatter = ax6.scatter(sample_data['T_out'], sample_data['Tdewpoint'], 
                                alpha=0.6, c=sample_data.get('RH_out', 'blue'), cmap='viridis')
            
            # Add perfect correlation line (dewpoint = temperature would be 100% humidity)
            min_temp = sample_data['T_out'].min()
            max_temp = sample_data['T_out'].max()
            ax6.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', 
                    alpha=0.7, label='Dewpoint = Temperature (100% RH)')
            
            ax6.set_xlabel('Outdoor Temperature (°C)')
            ax6.set_ylabel('Dew Point Temperature (°C)')
            ax6.set_title('Outdoor Temperature vs Dew Point')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            
            if 'RH_out' in self.available_columns:
                plt.colorbar(scatter, ax=ax6, label='Outdoor Humidity (%)')
            
            st.pyplot(fig6)
    
    def clean_data(self):
        """
        Clean the weather data part 1 based on analysis findings.
        """
        st.write("### Weather Data (Part 1) Cleaning")
        
        initial_rows = len(self.data)
        
        # Clean each weather variable
        for col in self.available_columns:
            st.write(f"**Cleaning {col}:**")
            
            if col == 'T_out':
                # Handle extreme temperatures
                extreme_mask = (self.data[col] < -50) | (self.data[col] > 60)
                if extreme_mask.sum() > 0:
                    self.data.loc[extreme_mask, col] = np.nan
                    st.write(f"- Set {extreme_mask.sum()} extreme temperature values to NaN")
                    self.log_action(f"T_out: Set {extreme_mask.sum()} extreme values to NaN")
                    
            elif col == 'RH_out':
                # Handle invalid humidity values
                invalid_mask = (self.data[col] < 0) | (self.data[col] > 100)
                if invalid_mask.sum() > 0:
                    self.data.loc[invalid_mask, col] = np.nan
                    st.write(f"- Set {invalid_mask.sum()} invalid humidity values to NaN")
                    self.log_action(f"RH_out: Set {invalid_mask.sum()} invalid values to NaN")
                    
            elif col == 'Tdewpoint':
                # Handle extreme dew points
                extreme_mask = (self.data[col] < -60) | (self.data[col] > 50)
                if extreme_mask.sum() > 0:
                    self.data.loc[extreme_mask, col] = np.nan
                    st.write(f"- Set {extreme_mask.sum()} extreme dew point values to NaN")
                    self.log_action(f"Tdewpoint: Set {extreme_mask.sum()} extreme values to NaN")
        
        # Physical relationship validation
        if 'T_out' in self.available_columns and 'Tdewpoint' in self.available_columns:
            # Dew point cannot be higher than air temperature
            impossible_mask = self.data['Tdewpoint'] > self.data['T_out']
            if impossible_mask.sum() > 0:
                # Set the higher value to NaN (could be either T_out or Tdewpoint that's wrong)
                self.data.loc[impossible_mask, 'Tdewpoint'] = np.nan
                st.write(f"- Set {impossible_mask.sum()} impossible dew point values to NaN (dewpoint > temperature)")
                self.log_action(f"Set {impossible_mask.sum()} impossible dewpoint values to NaN")
        
        # Handle missing values
        missing_cols = [col for col in self.available_columns if self.data[col].isnull().sum() > 0]
        
        if missing_cols:
            st.write("**Missing Value Imputation:**")
            
            # Strategy 1: Forward fill for short gaps (weather tends to be continuous)
            for col in missing_cols:
                is_missing = self.data[col].isnull()
                gap_lengths = is_missing.groupby((~is_missing).cumsum()).cumsum()
                short_gaps = (gap_lengths <= 2) & is_missing  # Very short gaps for weather
                
                if short_gaps.sum() > 0:
                    self.data.loc[short_gaps, col] = self.data[col].fillna(method='ffill').loc[short_gaps]
                    st.write(f"- {col}: Forward filled {short_gaps.sum()} short gaps")
            
            # Strategy 2: Linear interpolation for weather data (natural continuity)
            for col in missing_cols:
                initial_missing = self.data[col].isnull().sum()
                
                if initial_missing > 0:
                    # Use linear interpolation
                    self.data[col] = self.data[col].interpolate(method='linear', limit_direction='both')
                    
                    final_missing = self.data[col].isnull().sum()
                    interpolated = initial_missing - final_missing
                    
                    if interpolated > 0:
                        st.write(f"- {col}: Interpolated {interpolated} values")
                        self.log_action(f"{col}: Interpolated {interpolated} values")
            
            # Strategy 3: Use seasonal median for any remaining missing values
            for col in self.available_columns:
                remaining_missing = self.data[col].isnull().sum()
                
                if remaining_missing > 0:
                    if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
                        # Use monthly median
                        monthly_median = self.data.groupby(self.data['date'].dt.month)[col].median()
                        
                        # Fill missing values with corresponding monthly median
                        for month in range(1, 13):
                            if month in monthly_median.index:
                                month_mask = (self.data['date'].dt.month == month) & self.data[col].isnull()
                                if month_mask.sum() > 0:
                                    self.data.loc[month_mask, col] = monthly_median[month]
                    else:
                        # Use overall median if no date information
                        overall_median = self.data[col].median()
                        self.data[col].fillna(overall_median, inplace=True)
                    
                    final_missing = self.data[col].isnull().sum()
                    filled = remaining_missing - final_missing
                    
                    if filled > 0:
                        st.write(f"- {col}: Filled {filled} values with seasonal median")
                        self.log_action(f"{col}: Filled {filled} values with seasonal median")
        
        # Outlier treatment (conservative for weather data)
        st.write("**Outlier Treatment:**")
        for col in self.available_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use conservative bounds (2.5*IQR) as weather can have legitimate extremes
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            if outlier_mask.sum() > 0:
                st.write(f"- {col}: Identified {outlier_mask.sum()} potential outliers (kept - weather extremes can be real)")
                self.log_action(f"{col}: Identified {outlier_mask.sum()} outliers (not removed)")
        
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            st.write(f"Removed {removed_rows} rows during cleaning ({removed_rows/initial_rows*100:.2f}%)")
        else:
            st.write("No rows were removed during cleaning")
    
    def feature_engineering(self):
        """
        Create additional features from weather data part 1.
        """
        st.write("### Weather (Part 1) Feature Engineering")
        
        # 1. Heat Index (if T_out and RH_out available)
        if 'T_out' in self.available_columns and 'RH_out' in self.available_columns:
            # Convert Celsius to Fahrenheit for heat index calculation
            T_f = self.data['T_out'] * 9/5 + 32
            RH = self.data['RH_out']
            
            # Simplified heat index formula (accurate for T > 80°F and RH > 40%)
            heat_index_f = (0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094)))
            
            # For higher temperatures, use more complex formula
            high_temp_mask = T_f >= 80
            if high_temp_mask.sum() > 0:
                T_high = T_f[high_temp_mask]
                RH_high = RH[high_temp_mask]
                
                hi_complex = (-42.379 + 2.04901523 * T_high + 10.14333127 * RH_high
                             - 0.22475541 * T_high * RH_high - 0.00683783 * T_high**2
                             - 0.05481717 * RH_high**2 + 0.00122874 * T_high**2 * RH_high
                             + 0.00085282 * T_high * RH_high**2 - 0.00000199 * T_high**2 * RH_high**2)
                
                heat_index_f.loc[high_temp_mask] = hi_complex
            
            # Convert back to Celsius
            self.data['heat_index'] = (heat_index_f - 32) * 5/9
        
        # 2. Apparent Temperature (feels like)
        if 'T_out' in self.available_columns and 'RH_out' in self.available_columns:
            # Simplified apparent temperature formula
            T = self.data['T_out']
            RH = self.data['RH_out']
            
            self.data['apparent_temp'] = T + 0.33 * (RH / 100 * 6.105 * np.exp(17.27 * T / (237.7 + T))) - 0.70 - 4.00
        
        # 3. Dew point depression (T_out - Tdewpoint)
        if 'T_out' in self.available_columns and 'Tdewpoint' in self.available_columns:
            self.data['dewpoint_depression'] = self.data['T_out'] - self.data['Tdewpoint']
            
            # Relative humidity estimation from dew point depression
            # RH ≈ 100 - 5 * (T - Td) for rough approximation
            self.data['rh_estimated'] = np.clip(100 - 5 * self.data['dewpoint_depression'], 0, 100)
        
        # 4. Weather categories
        if 'T_out' in self.available_columns:
            # Temperature categories
            temp_conditions = [
                self.data['T_out'] < 0,   # Freezing
                (self.data['T_out'] >= 0) & (self.data['T_out'] < 10),   # Cold
                (self.data['T_out'] >= 10) & (self.data['T_out'] < 20),  # Cool
                (self.data['T_out'] >= 20) & (self.data['T_out'] < 30),  # Mild
                self.data['T_out'] >= 30  # Hot
            ]
            temp_choices = ['Freezing', 'Cold', 'Cool', 'Mild', 'Hot']
            self.data['temp_category'] = np.select(temp_conditions, temp_choices, default='Unknown')
        
        if 'RH_out' in self.available_columns:
            # Humidity categories
            humidity_conditions = [
                self.data['RH_out'] < 30,   # Very Dry
                (self.data['RH_out'] >= 30) & (self.data['RH_out'] < 50),   # Dry
                (self.data['RH_out'] >= 50) & (self.data['RH_out'] <= 70),  # Comfortable
                (self.data['RH_out'] > 70) & (self.data['RH_out'] <= 85),   # Humid
                self.data['RH_out'] > 85  # Very Humid
            ]
            humidity_choices = ['Very_Dry', 'Dry', 'Comfortable', 'Humid', 'Very_Humid']
            self.data['humidity_category'] = np.select(humidity_conditions, humidity_choices, default='Unknown')
        
        # 5. Comfort index
        if 'T_out' in self.available_columns and 'RH_out' in self.available_columns:
            # Discomfort index (temperature-humidity index)
            T = self.data['T_out']
            RH = self.data['RH_out']
            
            self.data['discomfort_index'] = T - 0.55 * (1 - RH/100) * (T - 14.5)
            
            # Comfort categories based on discomfort index
            comfort_conditions = [
                self.data['discomfort_index'] < 15,   # Too Cold
                (self.data['discomfort_index'] >= 15) & (self.data['discomfort_index'] < 20),   # Cool
                (self.data['discomfort_index'] >= 20) & (self.data['discomfort_index'] <= 26),  # Comfortable
                (self.data['discomfort_index'] > 26) & (self.data['discomfort_index'] <= 30),   # Warm
                self.data['discomfort_index'] > 30  # Too Hot
            ]
            comfort_choices = ['Too_Cold', 'Cool', 'Comfortable', 'Warm', 'Too_Hot']
            self.data['comfort_category'] = np.select(comfort_conditions, comfort_choices, default='Unknown')
        
        # 6. Temporal weather features (if date available)
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            # Moving averages for weather variables
            window_size = min(24, len(self.data) // 10)  # Adaptive window size
            
            for col in self.available_columns:
                if window_size > 1:
                    self.data[f'{col}_ma{window_size}h'] = self.data[col].rolling(
                        window=window_size, min_periods=1
                    ).mean()
                    
                    # Weather change rates
                    self.data[f'{col}_change'] = self.data[col].diff()
                    self.data[f'{col}_change_rate'] = self.data[f'{col}_change'].rolling(
                        window=min(6, window_size), min_periods=1
                    ).mean()
        
        # List all new features
        base_features = []
        
        if 'heat_index' in self.data.columns:
            base_features.append('heat_index')
        if 'apparent_temp' in self.data.columns:
            base_features.append('apparent_temp')
        if 'dewpoint_depression' in self.data.columns:
            base_features.extend(['dewpoint_depression', 'rh_estimated'])
        if 'temp_category' in self.data.columns:
            base_features.append('temp_category')
        if 'humidity_category' in self.data.columns:
            base_features.append('humidity_category')
        if 'discomfort_index' in self.data.columns:
            base_features.extend(['discomfort_index', 'comfort_category'])
        
        # Add moving average features
        ma_features = [col for col in self.data.columns if any(orig in col and 'ma' in col for orig in self.available_columns)]
        change_features = [col for col in self.data.columns if any(orig in col and 'change' in col for orig in self.available_columns)]
        
        new_features = base_features + ma_features + change_features
        
        if new_features:
            st.write(f"Created {len(new_features)} weather-related features:")
            for feature in new_features:
                st.write(f"- {feature}")
            
            self.log_action(f"Created weather features: {', '.join(new_features)}")
            
            # Show feature summary
            if 'heat_index' in self.data.columns:
                st.write(f"Average heat index: {self.data['heat_index'].mean():.2f}°C")
            
            if 'comfort_category' in self.data.columns:
                st.write("Comfort category distribution:")
                comfort_dist = self.data['comfort_category'].value_counts(normalize=True) * 100
                for category, percentage in comfort_dist.items():
                    st.write(f"  - {category}: {percentage:.1f}%")
        else:
            st.write("No additional features could be created from available data")
    
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
            'message': f"[Person 5 - Weather P1] {message}",
            'person': 5,
            'columns': self.columns
        }
        self.log.append(log_entry)