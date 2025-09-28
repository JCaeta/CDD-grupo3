"""
Person 3: Internal Temperature Sensors Analysis
Responsible for: T1, T2, T3, T4, T5, T6, T7, T8, T9 columns

This module handles the analysis and cleaning of internal temperature sensor data.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class TempSensorAnalyzer:
    """
    Analyzer for internal temperature sensor data: T1-T9 columns.
    """
    
    def __init__(self, data):
        """
        Initialize the temperature sensor analyzer.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data.copy()
        self.columns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
        self.available_columns = [col for col in self.columns if col in self.data.columns]
        self.log = []
        
    def exploratory_analysis(self):
        """
        Perform exploratory analysis on temperature sensor data.
        """
        st.write("### Internal Temperature Sensors Exploratory Analysis")
        
        if not self.available_columns:
            st.error("No temperature sensor columns found in dataset")
            return
        
        st.write(f"**Available temperature sensors:** {', '.join(self.available_columns)}")
        
        # Basic statistics for all temperature columns
        temp_stats = self.data[self.available_columns].describe()
        st.write("**Temperature Statistics (°C):**")
        st.write(temp_stats)
        
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
            self.log_action(f"Missing values found in sensors: {missing_analysis[missing_analysis > 0].to_dict()}")
        else:
            st.success("No missing values in temperature sensors")
        
        # Temperature range validation (reasonable indoor temperatures)
        st.write("**Temperature Range Validation:**")
        for col in self.available_columns:
            temp_data = self.data[col].dropna()
            min_temp = temp_data.min()
            max_temp = temp_data.max()
            
            # Flag unreasonable temperatures (outside typical indoor range)
            unreasonable_low = (temp_data < 5).sum()  # Below 5°C
            unreasonable_high = (temp_data > 40).sum()  # Above 40°C
            
            st.write(f"- {col}: {min_temp:.2f}°C to {max_temp:.2f}°C")
            if unreasonable_low > 0:
                st.warning(f"  → {unreasonable_low} values below 5°C")
            if unreasonable_high > 0:
                st.warning(f"  → {unreasonable_high} values above 40°C")
        
        # Correlation analysis between sensors
        if len(self.available_columns) > 1:
            st.write("**Sensor Correlation Analysis:**")
            correlation_matrix = self.data[self.available_columns].corr()
            
            # Find highly correlated pairs (>0.9)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.9:
                        high_corr_pairs.append(
                            (correlation_matrix.columns[i], 
                             correlation_matrix.columns[j], 
                             corr_value)
                        )
            
            if high_corr_pairs:
                st.write("Highly correlated sensor pairs (|r| > 0.9):")
                for sensor1, sensor2, corr in high_corr_pairs:
                    st.write(f"- {sensor1} ↔ {sensor2}: {corr:.3f}")
                    self.log_action(f"High correlation detected: {sensor1}-{sensor2} ({corr:.3f})")
            else:
                st.write("No extremely high correlations found between sensors")
        
        # Outlier detection using IQR method
        st.write("**Outlier Analysis:**")
        outlier_summary = {}
        for col in self.available_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.data)) * 100
            
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage
            }
            
            if outlier_count > 0:
                st.write(f"- {col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
        
        self.log_action(f"Outlier analysis completed: {outlier_summary}")
        
        # Variance analysis
        st.write("**Temperature Variance Analysis:**")
        variances = self.data[self.available_columns].var().sort_values(ascending=False)
        st.write("Sensor variance (descending order):")
        for sensor, variance in variances.items():
            st.write(f"- {sensor}: {variance:.4f}")
        
        # Identify potentially faulty sensors (extremely low variance might indicate sensor malfunction)
        low_variance_threshold = variances.mean() * 0.1  # 10% of mean variance
        faulty_sensors = variances[variances < low_variance_threshold]
        if len(faulty_sensors) > 0:
            st.warning(f"Potentially faulty sensors (low variance): {list(faulty_sensors.index)}")
            self.log_action(f"Low variance sensors detected: {list(faulty_sensors.index)}", "WARNING")
    
    def visualize_data(self):
        """
        Create visualizations for temperature sensor data.
        """
        st.write("### Temperature Sensor Visualizations")
        
        if not self.available_columns:
            return
        
        # 1. Correlation heatmap
        if len(self.available_columns) > 1:
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            correlation_matrix = self.data[self.available_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                       fmt='.2f', ax=ax1, cbar_kws={'label': 'Correlation'})
            ax1.set_title('Temperature Sensors Correlation Matrix')
            st.pyplot(fig1)
        
        # 2. Distribution plots
        fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
        axes2 = axes2.flatten()
        
        for i, col in enumerate(self.available_columns[:9]):  # Max 9 subplots
            if i < len(axes2):
                self.data[col].hist(bins=30, ax=axes2[i], alpha=0.7, color='skyblue', edgecolor='black')
                axes2[i].set_title(f'{col} Distribution')
                axes2[i].set_xlabel('Temperature (°C)')
                axes2[i].set_ylabel('Frequency')
                axes2[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.available_columns), len(axes2)):
            axes2[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # 3. Box plots for outlier visualization
        if len(self.available_columns) > 0:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            self.data[self.available_columns].boxplot(ax=ax3)
            ax3.set_title('Temperature Sensors Box Plots')
            ax3.set_xlabel('Sensor')
            ax3.set_ylabel('Temperature (°C)')
            plt.xticks(rotation=45)
            st.pyplot(fig3)
        
        # 4. Time series plot (if date column exists)
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            # Sample data for performance
            sample_size = min(1000, len(self.data))
            sample_data = self.data.sample(n=sample_size).sort_values('date')
            
            fig4, ax4 = plt.subplots(figsize=(15, 8))
            
            for col in self.available_columns[:5]:  # Show first 5 sensors to avoid clutter
                ax4.plot(sample_data['date'], sample_data[col], 
                        label=col, alpha=0.7, linewidth=1)
            
            ax4.set_title('Temperature Sensors Over Time (Sample Data)')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Temperature (°C)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig4)
        
        # 5. Average temperature profile
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        avg_temps = self.data[self.available_columns].mean().sort_values()
        avg_temps.plot(kind='bar', ax=ax5, color='orange', alpha=0.7)
        ax5.set_title('Average Temperature by Sensor')
        ax5.set_xlabel('Sensor')
        ax5.set_ylabel('Average Temperature (°C)')
        ax5.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig5)
    
    def clean_data(self):
        """
        Clean the temperature sensor data based on analysis findings.
        """
        st.write("### Temperature Sensor Data Cleaning")
        
        initial_rows = len(self.data)
        
        for col in self.available_columns:
            st.write(f"**Cleaning {col}:**")
            
            # 1. Handle unreasonable temperature values
            # Remove physically impossible temperatures (below -50°C or above 70°C)
            extreme_mask = (self.data[col] < -50) | (self.data[col] > 70)
            if extreme_mask.sum() > 0:
                self.data.loc[extreme_mask, col] = np.nan
                st.write(f"- Set {extreme_mask.sum()} extreme values to NaN")
                self.log_action(f"{col}: Set {extreme_mask.sum()} extreme values to NaN")
            
            # 2. Handle unreasonable indoor temperatures (more conservative)
            unreasonable_mask = (self.data[col] < 0) | (self.data[col] > 50)
            if unreasonable_mask.sum() > 0:
                # Option: Replace with NaN or clamp to reasonable range
                st.write(f"- Found {unreasonable_mask.sum()} unreasonable indoor temperatures")
                # Clamp to reasonable range instead of removing
                self.data.loc[self.data[col] < 0, col] = np.nan  # Too cold
                self.data.loc[self.data[col] > 50, col] = np.nan  # Too hot
                self.log_action(f"{col}: Handled {unreasonable_mask.sum()} unreasonable values")
        
        # 3. Handle missing values using advanced imputation
        missing_cols = [col for col in self.available_columns if self.data[col].isnull().sum() > 0]
        
        if missing_cols:
            st.write("**Missing Value Imputation:**")
            
            # Strategy 1: Forward fill for short gaps (<=3 consecutive missing values)
            for col in missing_cols:
                # Identify short gaps
                is_missing = self.data[col].isnull()
                gap_lengths = is_missing.groupby((~is_missing).cumsum()).cumsum()
                short_gaps = (gap_lengths <= 3) & is_missing
                
                if short_gaps.sum() > 0:
                    self.data.loc[short_gaps, col] = self.data[col].fillna(method='ffill').loc[short_gaps]
                    st.write(f"- {col}: Forward filled {short_gaps.sum()} short gaps")
            
            # Strategy 2: KNN imputation for longer gaps using other sensors
            remaining_missing = [col for col in self.available_columns if self.data[col].isnull().sum() > 0]
            
            if remaining_missing and len(self.available_columns) > 1:
                st.write("- Applying KNN imputation using correlated sensors...")
                
                # Use KNN imputation
                imputer = KNNImputer(n_neighbors=min(5, len(self.available_columns)-1))
                
                # Only impute if we have enough non-missing sensors
                non_missing_ratio = self.data[self.available_columns].notna().sum(axis=1) / len(self.available_columns)
                sufficient_data_mask = non_missing_ratio >= 0.5  # At least 50% of sensors have data
                
                if sufficient_data_mask.sum() > 0:
                    temp_subset = self.data.loc[sufficient_data_mask, self.available_columns]
                    imputed_subset = pd.DataFrame(
                        imputer.fit_transform(temp_subset),
                        index=temp_subset.index,
                        columns=temp_subset.columns
                    )
                    
                    # Update only the missing values
                    for col in remaining_missing:
                        missing_mask = self.data[col].isnull()
                        valid_imputed = missing_mask & sufficient_data_mask
                        if valid_imputed.sum() > 0:
                            self.data.loc[valid_imputed, col] = imputed_subset.loc[valid_imputed, col]
                            st.write(f"- {col}: KNN imputed {valid_imputed.sum()} values")
                            self.log_action(f"{col}: KNN imputed {valid_imputed.sum()} values")
            
            # Strategy 3: Use median for any remaining missing values
            for col in self.available_columns:
                remaining_missing_count = self.data[col].isnull().sum()
                if remaining_missing_count > 0:
                    median_value = self.data[col].median()
                    self.data[col].fillna(median_value, inplace=True)
                    st.write(f"- {col}: Median imputed {remaining_missing_count} values ({median_value:.2f}°C)")
                    self.log_action(f"{col}: Median imputed {remaining_missing_count} values")
        
        # 4. Outlier treatment (conservative approach for temperature sensors)
        st.write("**Outlier Treatment:**")
        for col in self.available_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use more conservative bounds (3*IQR instead of 1.5*IQR)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            if outlier_mask.sum() > 0:
                # Cap outliers instead of removing
                self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                st.write(f"- {col}: Capped {outlier_mask.sum()} extreme outliers")
                self.log_action(f"{col}: Capped {outlier_mask.sum()} extreme outliers")
        
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            st.write(f"Removed {removed_rows} rows during cleaning ({removed_rows/initial_rows*100:.2f}%)")
        else:
            st.write("No rows were removed during cleaning")
    
    def feature_engineering(self):
        """
        Create additional features from temperature sensor data.
        """
        st.write("### Temperature Feature Engineering")
        
        if len(self.available_columns) < 2:
            st.write("Need at least 2 temperature sensors for feature engineering")
            return
        
        # 1. Average internal temperature
        self.data['T_internal_avg'] = self.data[self.available_columns].mean(axis=1)
        
        # 2. Temperature range (max - min)
        self.data['T_internal_range'] = (self.data[self.available_columns].max(axis=1) - 
                                        self.data[self.available_columns].min(axis=1))
        
        # 3. Temperature standard deviation (thermal uniformity)
        self.data['T_internal_std'] = self.data[self.available_columns].std(axis=1)
        
        # 4. Number of sensors within normal range (18-24°C)
        normal_range_mask = (self.data[self.available_columns] >= 18) & (self.data[self.available_columns] <= 24)
        self.data['T_sensors_normal_count'] = normal_range_mask.sum(axis=1)
        
        # 5. Temperature zones (identify different thermal zones)
        if len(self.available_columns) >= 3:
            # Identify potential room groupings based on correlation
            correlation_matrix = self.data[self.available_columns].corr()
            
            # Simple clustering based on temperature similarity
            temp_medians = self.data[self.available_columns].median()
            
            # Group sensors by temperature level
            low_temp_sensors = temp_medians[temp_medians <= temp_medians.quantile(0.33)].index
            mid_temp_sensors = temp_medians[(temp_medians > temp_medians.quantile(0.33)) & 
                                          (temp_medians <= temp_medians.quantile(0.67))].index
            high_temp_sensors = temp_medians[temp_medians > temp_medians.quantile(0.67)].index
            
            if len(low_temp_sensors) > 0:
                self.data['T_zone_cool_avg'] = self.data[low_temp_sensors].mean(axis=1)
            if len(mid_temp_sensors) > 0:
                self.data['T_zone_moderate_avg'] = self.data[mid_temp_sensors].mean(axis=1)
            if len(high_temp_sensors) > 0:
                self.data['T_zone_warm_avg'] = self.data[high_temp_sensors].mean(axis=1)
        
        # 6. Temperature trends (if enough temporal data)
        if len(self.data) > 12:
            # Moving averages
            self.data['T_internal_ma6'] = self.data['T_internal_avg'].rolling(window=6, min_periods=1).mean()
            self.data['T_internal_ma12'] = self.data['T_internal_avg'].rolling(window=12, min_periods=1).mean()
            
            # Temperature change rate
            self.data['T_internal_change'] = self.data['T_internal_avg'].diff()
            self.data['T_internal_change_rate'] = self.data['T_internal_change'].rolling(window=3, min_periods=1).mean()
        
        # 7. Comfort index (subjective comfort based on temperature and uniformity)
        # Lower std and temperature in comfort range = higher comfort
        comfort_temp_score = 1 - abs(self.data['T_internal_avg'] - 21) / 10  # 21°C is optimal
        comfort_uniformity_score = 1 - self.data['T_internal_std'] / 5  # Lower std is better
        self.data['T_comfort_index'] = np.clip((comfort_temp_score + comfort_uniformity_score) / 2, 0, 1)
        
        # List all new features
        base_features = ['T_internal_avg', 'T_internal_range', 'T_internal_std', 'T_sensors_normal_count', 'T_comfort_index']
        new_features = base_features.copy()
        
        # Add zone features if they exist
        zone_features = [col for col in self.data.columns if col.startswith('T_zone_')]
        new_features.extend(zone_features)
        
        # Add trend features if they exist
        if len(self.data) > 12:
            trend_features = ['T_internal_ma6', 'T_internal_ma12', 'T_internal_change', 'T_internal_change_rate']
            new_features.extend(trend_features)
        
        st.write(f"Created {len(new_features)} temperature-related features:")
        for feature in new_features:
            if feature in self.data.columns:
                st.write(f"- {feature}")
        
        self.log_action(f"Created temperature features: {', '.join(new_features)}")
        
        # Show feature summary
        if 'T_internal_avg' in self.data.columns:
            st.write(f"Average internal temperature: {self.data['T_internal_avg'].mean():.2f}°C")
            st.write(f"Temperature range variation: {self.data['T_internal_range'].mean():.2f}°C")
            st.write(f"Average thermal uniformity (std): {self.data['T_internal_std'].mean():.2f}°C")
    
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
            'message': f"[Person 3 - Temperature] {message}",
            'person': 3,
            'columns': self.columns
        }
        self.log.append(log_entry)