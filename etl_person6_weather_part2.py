"""
Person 6: External Meteorological Data (Part 2) Analysis
Responsible for: Pressure, Wind speed, Visibility columns

This module handles the analysis and cleaning of external weather data part 2.
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

class WeatherPart2Analyzer:
    """
    Analyzer for external weather data part 2: Pressure, Wind speed, Visibility columns.
    """
    
    def __init__(self, data):
        """
        Initialize the weather part 2 analyzer.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data.copy()
        # Check for different possible column names
        self.pressure_cols = [col for col in self.data.columns if 'press' in col.lower()]
        self.windspeed_cols = [col for col in self.data.columns if 'wind' in col.lower()]
        self.visibility_cols = [col for col in self.data.columns if 'visib' in col.lower()]
        
        self.columns = self.pressure_cols + self.windspeed_cols + self.visibility_cols
        self.available_columns = self.columns
        self.log = []
        
    def exploratory_analysis(self):
        """
        Perform exploratory analysis on weather data part 2.
        """
        st.write("### External Weather Data (Part 2) Exploratory Analysis")
        
        if not self.available_columns:
            st.error("No weather part 2 columns found in dataset")
            st.write("Looking for columns containing: 'press', 'wind', 'visib'")
            st.write(f"Available columns: {list(self.data.columns)}")
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
        
        # Validate each weather variable type
        st.write("**Weather Data Validation:**")
        
        # Pressure validation
        for col in self.pressure_cols:
            pressure_data = self.data[col].dropna()
            min_pressure = pressure_data.min()
            max_pressure = pressure_data.max()
            
            st.write(f"- {col}: {min_pressure:.2f} to {max_pressure:.2f}")
            
            # Check units and reasonable ranges
            if max_pressure > 2000:  # Likely in hPa/mb
                st.write(f"  → Appears to be in hPa/mbar units")
                # Normal sea level pressure: 950-1050 hPa
                low_pressure = (pressure_data < 900).sum()
                high_pressure = (pressure_data > 1100).sum()
            elif max_pressure > 500:  # Likely in mmHg
                st.write(f"  → Appears to be in mmHg units")
                # Normal sea level pressure: 710-790 mmHg
                low_pressure = (pressure_data < 650).sum()
                high_pressure = (pressure_data > 850).sum()
            else:  # Likely in inHg or other units
                st.write(f"  → Units unclear, assuming reasonable for unit type")
                low_pressure = 0
                high_pressure = 0
            
            if low_pressure > 0:
                st.warning(f"  → {low_pressure} extremely low pressure readings")
                self.log_action(f"{col}: {low_pressure} extremely low pressure readings", "WARNING")
            if high_pressure > 0:
                st.warning(f"  → {high_pressure} extremely high pressure readings")
                self.log_action(f"{col}: {high_pressure} extremely high pressure readings", "WARNING")
        
        # Wind speed validation
        for col in self.windspeed_cols:
            windspeed_data = self.data[col].dropna()
            min_wind = windspeed_data.min()
            max_wind = windspeed_data.max()
            
            st.write(f"- {col}: {min_wind:.2f} to {max_wind:.2f}")
            
            # Wind speed should be non-negative
            negative_wind = (windspeed_data < 0).sum()
            if negative_wind > 0:
                st.error(f"  → {negative_wind} negative wind speed values (impossible)")
                self.log_action(f"{col}: {negative_wind} negative wind speeds", "ERROR")
            
            # Check for extreme wind speeds
            if max_wind > 200:  # Likely in km/h
                st.write(f"  → Appears to be in km/h units")
                extreme_wind = (windspeed_data > 300).sum()  # > 300 km/h is extreme
            elif max_wind > 100:  # Likely in mph
                st.write(f"  → Appears to be in mph units")
                extreme_wind = (windspeed_data > 200).sum()  # > 200 mph is extreme
            else:  # Likely in m/s
                st.write(f"  → Appears to be in m/s units")
                extreme_wind = (windspeed_data > 50).sum()  # > 50 m/s is extreme
            
            if extreme_wind > 0:
                st.warning(f"  → {extreme_wind} extreme wind speed readings")
                self.log_action(f"{col}: {extreme_wind} extreme wind speeds", "WARNING")
        
        # Visibility validation
        for col in self.visibility_cols:
            visibility_data = self.data[col].dropna()
            min_vis = visibility_data.min()
            max_vis = visibility_data.max()
            
            st.write(f"- {col}: {min_vis:.2f} to {max_vis:.2f}")
            
            # Visibility should be non-negative
            negative_vis = (visibility_data < 0).sum()
            if negative_vis > 0:
                st.error(f"  → {negative_vis} negative visibility values (impossible)")
                self.log_action(f"{col}: {negative_vis} negative visibility", "ERROR")
            
            # Check reasonable visibility ranges
            if max_vis > 100:  # Likely in km
                st.write(f"  → Appears to be in km units")
                extreme_vis = (visibility_data > 500).sum()  # > 500 km is unusual
            else:  # Likely in miles or other units
                st.write(f"  → Units unclear, checking for extremes")
                # Use quartiles to detect extremes
                Q3 = visibility_data.quantile(0.75)
                extreme_vis = (visibility_data > Q3 * 10).sum()
            
            if extreme_vis > 0:
                st.warning(f"  → {extreme_vis} extreme visibility readings")
                self.log_action(f"{col}: {extreme_vis} extreme visibility", "WARNING")
        
        # Cross-variable relationships
        if len(self.available_columns) > 1:
            st.write("**Weather Relationships Analysis:**")
            correlation_matrix = self.data[self.available_columns].corr()
            
            st.write("Variable correlations:")
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    st.write(f"- {var1} ↔ {var2}: {corr_value:.3f}")
            
            # Expected relationships
            pressure_wind_pairs = [(p, w) for p in self.pressure_cols for w in self.windspeed_cols]
            for p_col, w_col in pressure_wind_pairs:
                if p_col in correlation_matrix.index and w_col in correlation_matrix.index:
                    corr = correlation_matrix.loc[p_col, w_col]
                    if abs(corr) > 0.3:
                        st.write(f"  → Notable pressure-wind correlation: {corr:.3f}")
        
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
        
        # Temporal patterns (if date available)
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            st.write("**Temporal Patterns:**")
            
            for col in self.available_columns:
                # Check for seasonal patterns
                if len(self.data) > 100:
                    monthly_stats = self.data.groupby(self.data['date'].dt.month)[col].agg(['mean', 'std'])
                    seasonal_variation = monthly_stats['mean'].max() - monthly_stats['mean'].min()
                    st.write(f"- {col} seasonal variation: {seasonal_variation:.2f}")
                    
                    # Check for daily patterns
                    if len(self.data) > 1000:
                        hourly_stats = self.data.groupby(self.data['date'].dt.hour)[col].agg(['mean', 'std'])
                        daily_variation = hourly_stats['mean'].max() - hourly_stats['mean'].min()
                        st.write(f"- {col} daily variation: {daily_variation:.2f}")
    
    def visualize_data(self):
        """
        Create visualizations for weather data part 2.
        """
        st.write("### Weather Data (Part 2) Visualizations")
        
        if not self.available_columns:
            return
        
        # 1. Distribution plots
        n_cols = len(self.available_columns)
        if n_cols > 0:
            n_rows = (n_cols + 2) // 3  # Up to 3 columns per row
            fig1, axes1 = plt.subplots(n_rows, min(3, n_cols), figsize=(15, 5*n_rows))
            
            if n_cols == 1:
                axes1 = [axes1]
            elif n_rows == 1:
                axes1 = axes1 if n_cols > 1 else [axes1]
            else:
                axes1 = axes1.flatten()
            
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            
            for i, col in enumerate(self.available_columns):
                if i < len(axes1):
                    self.data[col].hist(bins=30, ax=axes1[i], alpha=0.7, 
                                      color=colors[i % len(colors)], edgecolor='black')
                    axes1[i].set_title(f'{col} Distribution')
                    
                    # Set appropriate labels based on variable type
                    if any(p in col.lower() for p in ['press']):
                        axes1[i].set_xlabel('Pressure')
                    elif any(w in col.lower() for w in ['wind']):
                        axes1[i].set_xlabel('Wind Speed')
                    elif any(v in col.lower() for v in ['visib']):
                        axes1[i].set_xlabel('Visibility')
                    
                    axes1[i].set_ylabel('Frequency')
                    axes1[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(self.available_columns), len(axes1)):
                axes1[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        # 2. Correlation heatmap (if multiple variables)
        if len(self.available_columns) > 1:
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            correlation_matrix = self.data[self.available_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       fmt='.3f', ax=ax2, cbar_kws={'label': 'Correlation'})
            ax2.set_title('Weather Variables (Part 2) Correlation Matrix')
            st.pyplot(fig2)
        
        # 3. Box plots
        if len(self.available_columns) > 0:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            # Normalize data for better comparison if scales are very different
            normalized_data = self.data[self.available_columns].copy()
            for col in self.available_columns:
                col_std = self.data[col].std()
                col_mean = self.data[col].mean()
                if col_std > 0:
                    normalized_data[col] = (self.data[col] - col_mean) / col_std
            
            normalized_data.boxplot(ax=ax3)
            ax3.set_title('Weather Variables Box Plots (Normalized)')
            ax3.set_ylabel('Normalized Values')
            plt.xticks(rotation=45)
            st.pyplot(fig3)
        
        # 4. Time series plots (if date available)
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            # Sample data for performance
            sample_size = min(2000, len(self.data))
            sample_data = self.data.sample(n=sample_size).sort_values('date')
            
            fig4, axes4 = plt.subplots(len(self.available_columns), 1, 
                                     figsize=(15, 4*len(self.available_columns)))
            
            if len(self.available_columns) == 1:
                axes4 = [axes4]
            
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            
            for i, col in enumerate(self.available_columns):
                axes4[i].plot(sample_data['date'], sample_data[col], 
                             alpha=0.7, linewidth=0.8, color=colors[i % len(colors)])
                axes4[i].set_title(f'{col} Over Time (Sample Data)')
                axes4[i].set_xlabel('Date')
                axes4[i].set_ylabel(col)
                axes4[i].grid(True, alpha=0.3)
                plt.setp(axes4[i].xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig4)
            
            # 5. Daily and seasonal patterns
            if len(self.data) > 100:
                fig5, axes5 = plt.subplots(2, len(self.available_columns), 
                                         figsize=(5*len(self.available_columns), 10))
                
                if len(self.available_columns) == 1:
                    axes5 = axes5.reshape(-1, 1)
                
                for i, col in enumerate(self.available_columns):
                    # Monthly averages
                    monthly_avg = self.data.groupby(self.data['date'].dt.month)[col].mean()
                    monthly_avg.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_avg)]
                    
                    monthly_avg.plot(kind='line', ax=axes5[0, i], marker='o', 
                                   color=colors[i % len(colors)], linewidth=2, markersize=6)
                    axes5[0, i].set_title(f'{col} - Monthly Averages')
                    axes5[0, i].set_xlabel('Month')
                    axes5[0, i].set_ylabel(col)
                    axes5[0, i].grid(True, alpha=0.3)
                    plt.setp(axes5[0, i].xaxis.get_majorticklabels(), rotation=45)
                    
                    # Hourly averages (if enough data)
                    if len(self.data) > 1000:
                        hourly_avg = self.data.groupby(self.data['date'].dt.hour)[col].mean()
                        
                        hourly_avg.plot(kind='line', ax=axes5[1, i], marker='s', 
                                       color=colors[i % len(colors)], linewidth=2, markersize=4)
                        axes5[1, i].set_title(f'{col} - Hourly Averages')
                        axes5[1, i].set_xlabel('Hour of Day')
                        axes5[1, i].set_ylabel(col)
                        axes5[1, i].grid(True, alpha=0.3)
                    else:
                        axes5[1, i].text(0.5, 0.5, 'Not enough data\nfor hourly analysis', 
                                       transform=axes5[1, i].transAxes, ha='center', va='center')
                        axes5[1, i].set_title(f'{col} - Insufficient Data')
                
                plt.tight_layout()
                st.pyplot(fig5)
        
        # 6. Relationship plots between variables
        if len(self.available_columns) >= 2:
            # Sample for performance
            sample_size = min(1000, len(self.data))
            sample_data = self.data.sample(n=sample_size)
            
            # Create pairplot for all combinations
            fig6, axes6 = plt.subplots(len(self.available_columns), len(self.available_columns), 
                                     figsize=(4*len(self.available_columns), 4*len(self.available_columns)))
            
            if len(self.available_columns) == 2:
                axes6 = axes6.reshape(2, 2)
            
            for i, col1 in enumerate(self.available_columns):
                for j, col2 in enumerate(self.available_columns):
                    if i == j:
                        # Diagonal: histogram
                        sample_data[col1].hist(ax=axes6[i, j], bins=20, alpha=0.7)
                        axes6[i, j].set_title(f'{col1} Distribution')
                    else:
                        # Off-diagonal: scatter plot
                        axes6[i, j].scatter(sample_data[col2], sample_data[col1], alpha=0.6, s=10)
                        axes6[i, j].set_xlabel(col2)
                        axes6[i, j].set_ylabel(col1)
                        
                        # Add correlation coefficient
                        corr = sample_data[col1].corr(sample_data[col2])
                        axes6[i, j].text(0.05, 0.95, f'r = {corr:.2f}', 
                                        transform=axes6[i, j].transAxes, 
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig6)
    
    def clean_data(self):
        """
        Clean the weather data part 2 based on analysis findings.
        """
        st.write("### Weather Data (Part 2) Cleaning")
        
        initial_rows = len(self.data)
        
        # Clean pressure data
        for col in self.pressure_cols:
            st.write(f"**Cleaning {col}:**")
            
            # Detect likely units and set reasonable bounds
            max_val = self.data[col].max()
            if max_val > 2000:  # Likely hPa/mb
                lower_bound, upper_bound = 800, 1200  # Extreme weather bounds
                unit = "hPa"
            elif max_val > 500:  # Likely mmHg
                lower_bound, upper_bound = 600, 900
                unit = "mmHg"
            else:  # Other units
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 4 * IQR
                upper_bound = Q3 + 4 * IQR
                unit = "unknown"
            
            extreme_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            if extreme_mask.sum() > 0:
                self.data.loc[extreme_mask, col] = np.nan
                st.write(f"- Set {extreme_mask.sum()} extreme pressure values to NaN (unit: {unit})")
                self.log_action(f"{col}: Set {extreme_mask.sum()} extreme values to NaN")
        
        # Clean wind speed data
        for col in self.windspeed_cols:
            st.write(f"**Cleaning {col}:**")
            
            # Wind speed cannot be negative
            negative_mask = self.data[col] < 0
            if negative_mask.sum() > 0:
                self.data.loc[negative_mask, col] = 0  # Set to calm conditions
                st.write(f"- Set {negative_mask.sum()} negative wind speeds to 0")
                self.log_action(f"{col}: Set {negative_mask.sum()} negative values to 0")
            
            # Handle extreme wind speeds based on likely units
            max_val = self.data[col].max()
            if max_val > 200:  # Likely km/h
                extreme_threshold = 400  # > 400 km/h is impossible except in tornadoes
            elif max_val > 100:  # Likely mph
                extreme_threshold = 250  # > 250 mph is impossible except in tornadoes
            else:  # Likely m/s
                extreme_threshold = 70  # > 70 m/s is extremely rare
            
            extreme_mask = self.data[col] > extreme_threshold
            if extreme_mask.sum() > 0:
                # Cap at reasonable maximum instead of removing
                self.data.loc[extreme_mask, col] = extreme_threshold
                st.write(f"- Capped {extreme_mask.sum()} extreme wind speeds at {extreme_threshold}")
                self.log_action(f"{col}: Capped {extreme_mask.sum()} extreme values")
        
        # Clean visibility data
        for col in self.visibility_cols:
            st.write(f"**Cleaning {col}:**")
            
            # Visibility cannot be negative
            negative_mask = self.data[col] < 0
            if negative_mask.sum() > 0:
                self.data.loc[negative_mask, col] = 0  # Set to zero visibility (fog/storm)
                st.write(f"- Set {negative_mask.sum()} negative visibility values to 0")
                self.log_action(f"{col}: Set {negative_mask.sum()} negative values to 0")
            
            # Handle extreme visibility values
            max_val = self.data[col].max()
            Q99 = self.data[col].quantile(0.99)
            
            # Use 99th percentile * 2 as upper bound for extreme values
            extreme_threshold = Q99 * 2
            extreme_mask = self.data[col] > extreme_threshold
            
            if extreme_mask.sum() > 0:
                self.data.loc[extreme_mask, col] = extreme_threshold
                st.write(f"- Capped {extreme_mask.sum()} extreme visibility values at {extreme_threshold:.1f}")
                self.log_action(f"{col}: Capped {extreme_mask.sum()} extreme values")
        
        # Handle missing values
        missing_cols = [col for col in self.available_columns if self.data[col].isnull().sum() > 0]
        
        if missing_cols:
            st.write("**Missing Value Imputation:**")
            
            # Strategy 1: Forward fill for short gaps (weather is continuous)
            for col in missing_cols:
                is_missing = self.data[col].isnull()
                gap_lengths = is_missing.groupby((~is_missing).cumsum()).cumsum()
                short_gaps = (gap_lengths <= 3) & is_missing  # Allow slightly longer gaps
                
                if short_gaps.sum() > 0:
                    self.data.loc[short_gaps, col] = self.data[col].fillna(method='ffill').loc[short_gaps]
                    st.write(f"- {col}: Forward filled {short_gaps.sum()} short gaps")
            
            # Strategy 2: Linear interpolation
            for col in missing_cols:
                initial_missing = self.data[col].isnull().sum()
                
                if initial_missing > 0:
                    self.data[col] = self.data[col].interpolate(method='linear', limit_direction='both')
                    
                    final_missing = self.data[col].isnull().sum()
                    interpolated = initial_missing - final_missing
                    
                    if interpolated > 0:
                        st.write(f"- {col}: Interpolated {interpolated} values")
                        self.log_action(f"{col}: Interpolated {interpolated} values")
            
            # Strategy 3: Use seasonal patterns or overall statistics
            for col in self.available_columns:
                remaining_missing = self.data[col].isnull().sum()
                
                if remaining_missing > 0:
                    if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
                        # Use monthly median for seasonality
                        monthly_median = self.data.groupby(self.data['date'].dt.month)[col].median()
                        
                        for month in range(1, 13):
                            if month in monthly_median.index:
                                month_mask = (self.data['date'].dt.month == month) & self.data[col].isnull()
                                if month_mask.sum() > 0:
                                    self.data.loc[month_mask, col] = monthly_median[month]
                    else:
                        # Use overall median
                        overall_median = self.data[col].median()
                        self.data[col].fillna(overall_median, inplace=True)
                    
                    final_missing = self.data[col].isnull().sum()
                    filled = remaining_missing - final_missing
                    
                    if filled > 0:
                        st.write(f"- {col}: Filled {filled} values with seasonal/overall median")
                        self.log_action(f"{col}: Filled {filled} values with median")
        
        # Conservative outlier treatment (weather can have legitimate extremes)
        st.write("**Outlier Treatment:**")
        for col in self.available_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Very conservative bounds (3*IQR) as weather extremes are real
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            if outlier_mask.sum() > 0:
                st.write(f"- {col}: Identified {outlier_mask.sum()} potential outliers (kept - weather extremes are valid)")
                self.log_action(f"{col}: Identified {outlier_mask.sum()} outliers (not removed)")
        
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            st.write(f"Removed {removed_rows} rows during cleaning ({removed_rows/initial_rows*100:.2f}%)")
        else:
            st.write("No rows were removed during cleaning")
    
    def feature_engineering(self):
        """
        Create additional features from weather data part 2.
        """
        st.write("### Weather (Part 2) Feature Engineering")
        
        new_features = []
        
        # 1. Pressure-based features
        if self.pressure_cols:
            pressure_col = self.pressure_cols[0]  # Use first pressure column
            
            # Pressure categories (relative to local conditions)
            pressure_median = self.data[pressure_col].median()
            pressure_std = self.data[pressure_col].std()
            
            # Categorize based on deviation from median
            pressure_conditions = [
                self.data[pressure_col] < (pressure_median - pressure_std),  # Low
                (self.data[pressure_col] >= (pressure_median - pressure_std)) & 
                (self.data[pressure_col] <= (pressure_median + pressure_std)),  # Normal
                self.data[pressure_col] > (pressure_median + pressure_std)  # High
            ]
            pressure_choices = ['Low_Pressure', 'Normal_Pressure', 'High_Pressure']
            self.data['pressure_category'] = np.select(pressure_conditions, pressure_choices, default='Unknown')
            new_features.append('pressure_category')
            
            # Pressure change rate (barometric tendency)
            if len(self.data) > 3:
                self.data['pressure_change'] = self.data[pressure_col].diff()
                self.data['pressure_tendency'] = self.data['pressure_change'].rolling(window=3, min_periods=1).mean()
                new_features.extend(['pressure_change', 'pressure_tendency'])
                
                # Pressure tendency categories
                tend_conditions = [
                    self.data['pressure_tendency'] < -0.5,  # Falling
                    abs(self.data['pressure_tendency']) <= 0.5,  # Steady
                    self.data['pressure_tendency'] > 0.5  # Rising
                ]
                tend_choices = ['Falling', 'Steady', 'Rising']
                self.data['pressure_trend'] = np.select(tend_conditions, tend_choices, default='Unknown')
                new_features.append('pressure_trend')
        
        # 2. Wind-based features
        if self.windspeed_cols:
            wind_col = self.windspeed_cols[0]  # Use first wind speed column
            
            # Wind speed categories (Beaufort scale inspired)
            wind_conditions = [
                self.data[wind_col] == 0,  # Calm
                (self.data[wind_col] > 0) & (self.data[wind_col] <= 5),  # Light
                (self.data[wind_col] > 5) & (self.data[wind_col] <= 15),  # Moderate
                (self.data[wind_col] > 15) & (self.data[wind_col] <= 25),  # Fresh
                (self.data[wind_col] > 25) & (self.data[wind_col] <= 35),  # Strong
                self.data[wind_col] > 35  # Very Strong
            ]
            wind_choices = ['Calm', 'Light', 'Moderate', 'Fresh', 'Strong', 'Very_Strong']
            self.data['wind_category'] = np.select(wind_conditions, wind_choices, default='Unknown')
            new_features.append('wind_category')
            
            # Wind gusts indicator (high variability)
            if len(self.data) > 6:
                wind_std = self.data[wind_col].rolling(window=6, min_periods=3).std()
                self.data['wind_variability'] = wind_std
                
                # Gusty conditions (high short-term variability)
                gust_threshold = wind_std.quantile(0.75)
                self.data['is_gusty'] = (wind_std > gust_threshold).astype(int)
                new_features.extend(['wind_variability', 'is_gusty'])
        
        # 3. Visibility-based features
        if self.visibility_cols:
            vis_col = self.visibility_cols[0]  # Use first visibility column
            
            # Visibility categories
            vis_conditions = [
                self.data[vis_col] < 1,  # Very Poor (fog/heavy precipitation)
                (self.data[vis_col] >= 1) & (self.data[vis_col] < 5),  # Poor
                (self.data[vis_col] >= 5) & (self.data[vis_col] < 15),  # Moderate
                (self.data[vis_col] >= 15) & (self.data[vis_col] < 30),  # Good
                self.data[vis_col] >= 30  # Excellent
            ]
            vis_choices = ['Very_Poor', 'Poor', 'Moderate', 'Good', 'Excellent']
            self.data['visibility_category'] = np.select(vis_conditions, vis_choices, default='Unknown')
            new_features.append('visibility_category')
            
            # Fog indicator (very low visibility)
            self.data['fog_indicator'] = (self.data[vis_col] < 1).astype(int)
            new_features.append('fog_indicator')
        
        # 4. Combined weather features
        if self.pressure_cols and self.windspeed_cols:
            pressure_col = self.pressure_cols[0]
            wind_col = self.windspeed_cols[0]
            
            # Storm indicator (low pressure + high wind)
            pressure_low = self.data[pressure_col] < self.data[pressure_col].quantile(0.25)
            wind_high = self.data[wind_col] > self.data[wind_col].quantile(0.75)
            self.data['storm_indicator'] = (pressure_low & wind_high).astype(int)
            new_features.append('storm_indicator')
            
            # Weather stability index (combination of pressure stability and wind)
            if 'pressure_tendency' in self.data.columns and 'wind_variability' in self.data.columns:
                # Normalize both components
                pressure_stability = 1 / (1 + abs(self.data['pressure_tendency']))
                wind_stability = 1 / (1 + self.data['wind_variability'])
                self.data['weather_stability'] = (pressure_stability + wind_stability) / 2
                new_features.append('weather_stability')
        
        # 5. Weather system indicators
        if len(self.available_columns) >= 2:
            # High pressure system (high pressure + low wind + good visibility)
            conditions_high_pressure = []
            
            if self.pressure_cols:
                pressure_col = self.pressure_cols[0]
                conditions_high_pressure.append(
                    self.data[pressure_col] > self.data[pressure_col].quantile(0.6)
                )
            
            if self.windspeed_cols:
                wind_col = self.windspeed_cols[0]
                conditions_high_pressure.append(
                    self.data[wind_col] < self.data[wind_col].quantile(0.4)
                )
            
            if self.visibility_cols:
                vis_col = self.visibility_cols[0]
                conditions_high_pressure.append(
                    self.data[vis_col] > self.data[vis_col].quantile(0.6)
                )
            
            if len(conditions_high_pressure) >= 2:
                high_pressure_system = np.logical_and.reduce(conditions_high_pressure)
                self.data['high_pressure_system'] = high_pressure_system.astype(int)
                new_features.append('high_pressure_system')
        
        # 6. Temporal weather features (if date available)
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            # Moving averages for all weather variables
            window_size = min(12, len(self.data) // 10)
            
            for col in self.available_columns:
                if window_size > 1:
                    ma_col = f'{col}_ma{window_size}h'
                    self.data[ma_col] = self.data[col].rolling(
                        window=window_size, min_periods=1
                    ).mean()
                    new_features.append(ma_col)
                    
                    # Weather change rates
                    change_col = f'{col}_change_rate'
                    self.data[f'{col}_change'] = self.data[col].diff()
                    self.data[change_col] = self.data[f'{col}_change'].rolling(
                        window=min(3, window_size), min_periods=1
                    ).mean()
                    new_features.extend([f'{col}_change', change_col])
        
        # 7. Weather extremes indicators
        for col in self.available_columns:
            # Mark extreme values (beyond 95th percentile or below 5th percentile)
            P05 = self.data[col].quantile(0.05)
            P95 = self.data[col].quantile(0.95)
            
            extreme_col = f'{col}_extreme'
            self.data[extreme_col] = ((self.data[col] < P05) | (self.data[col] > P95)).astype(int)
            new_features.append(extreme_col)
        
        if new_features:
            st.write(f"Created {len(new_features)} weather-related features:")
            for feature in new_features:
                if feature in self.data.columns:
                    st.write(f"- {feature}")
            
            self.log_action(f"Created weather features: {', '.join(new_features)}")
            
            # Show feature summary
            if 'pressure_category' in self.data.columns:
                st.write("Pressure category distribution:")
                pressure_dist = self.data['pressure_category'].value_counts(normalize=True) * 100
                for category, percentage in pressure_dist.items():
                    st.write(f"  - {category}: {percentage:.1f}%")
            
            if 'wind_category' in self.data.columns:
                st.write("Wind category distribution:")
                wind_dist = self.data['wind_category'].value_counts(normalize=True) * 100
                for category, percentage in wind_dist.items():
                    st.write(f"  - {category}: {percentage:.1f}%")
            
            if 'storm_indicator' in self.data.columns:
                storm_percentage = self.data['storm_indicator'].mean() * 100
                st.write(f"Storm conditions: {storm_percentage:.1f}% of time")
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
            'message': f"[Person 6 - Weather P2] {message}",
            'person': 6,
            'columns': self.columns
        }
        self.log.append(log_entry)