"""
Person 4: Internal Humidity Sensors Analysis
Responsible for: RH_1, RH_2, RH_3, RH_4, RH_5, RH_6, RH_7, RH_8, RH_9 columns

This module handles the analysis and cleaning of internal humidity sensor data.
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

class HumiditySensorAnalyzer:
    """
    Analyzer for internal humidity sensor data: RH_1 through RH_9 columns.
    """
    
    def __init__(self, data):
        """
        Initialize the humidity sensor analyzer.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.data = data.copy()
        self.columns = ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9']
        self.available_columns = [col for col in self.columns if col in self.data.columns]
        self.log = []
        
    def exploratory_analysis(self):
        """
        Perform exploratory analysis on humidity sensor data.
        """
        st.write("### Internal Humidity Sensors Exploratory Analysis")
        
        if not self.available_columns:
            st.error("No humidity sensor columns found in dataset")
            return
        
        st.write(f"**Available humidity sensors:** {', '.join(self.available_columns)}")
        
        # Basic statistics for all humidity columns
        humidity_stats = self.data[self.available_columns].describe()
        st.write("**Humidity Statistics (%):**")
        st.write(humidity_stats)
        
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
            st.success("No missing values in humidity sensors")
        
        # Humidity range validation (0-100%)
        st.write("**Humidity Range Validation:**")
        for col in self.available_columns:
            humidity_data = self.data[col].dropna()
            min_humidity = humidity_data.min()
            max_humidity = humidity_data.max()
            
            # Flag invalid humidity values (outside 0-100% range)
            invalid_low = (humidity_data < 0).sum()
            invalid_high = (humidity_data > 100).sum()
            
            st.write(f"- {col}: {min_humidity:.2f}% to {max_humidity:.2f}%")
            if invalid_low > 0:
                st.error(f"  → {invalid_low} values below 0%")
                self.log_action(f"{col}: {invalid_low} values below 0%", "ERROR")
            if invalid_high > 0:
                st.error(f"  → {invalid_high} values above 100%")
                self.log_action(f"{col}: {invalid_high} values above 100%", "ERROR")
            
            # Flag extreme but possible values
            extreme_low = ((humidity_data >= 0) & (humidity_data < 10)).sum()
            extreme_high = ((humidity_data <= 100) & (humidity_data > 90)).sum()
            
            if extreme_low > 0:
                st.warning(f"  → {extreme_low} extremely low values (0-10%)")
            if extreme_high > 0:
                st.warning(f"  → {extreme_high} extremely high values (90-100%)")
        
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
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_count > 0:
                st.write(f"- {col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
        
        self.log_action(f"Outlier analysis completed: {outlier_summary}")
        
        # Variance analysis
        st.write("**Humidity Variance Analysis:**")
        variances = self.data[self.available_columns].var().sort_values(ascending=False)
        st.write("Sensor variance (descending order):")
        for sensor, variance in variances.items():
            st.write(f"- {sensor}: {variance:.4f}")
        
        # Identify potentially faulty sensors
        low_variance_threshold = variances.mean() * 0.1  # 10% of mean variance
        faulty_sensors = variances[variances < low_variance_threshold]
        if len(faulty_sensors) > 0:
            st.warning(f"Potentially faulty sensors (low variance): {list(faulty_sensors.index)}")
            self.log_action(f"Low variance sensors detected: {list(faulty_sensors.index)}", "WARNING")
        
        # Humidity comfort analysis (40-60% is typically comfortable)
        st.write("**Humidity Comfort Analysis:**")
        for col in self.available_columns:
            humidity_data = self.data[col].dropna()
            comfortable_count = ((humidity_data >= 40) & (humidity_data <= 60)).sum()
            comfortable_percentage = (comfortable_count / len(humidity_data)) * 100
            st.write(f"- {col}: {comfortable_percentage:.1f}% of readings in comfort range (40-60%)")
    
    def visualize_data(self):
        """
        Create visualizations for humidity sensor data.
        """
        st.write("### Humidity Sensor Visualizations")
        
        if not self.available_columns:
            return
        
        # 1. Correlation heatmap
        if len(self.available_columns) > 1:
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            correlation_matrix = self.data[self.available_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                       fmt='.2f', ax=ax1, cbar_kws={'label': 'Correlation'})
            ax1.set_title('Humidity Sensors Correlation Matrix')
            st.pyplot(fig1)
        
        # 2. Distribution plots
        fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
        axes2 = axes2.flatten()
        
        for i, col in enumerate(self.available_columns[:9]):  # Max 9 subplots
            if i < len(axes2):
                self.data[col].hist(bins=30, ax=axes2[i], alpha=0.7, color='lightgreen', edgecolor='black')
                axes2[i].set_title(f'{col} Distribution')
                axes2[i].set_xlabel('Relative Humidity (%)')
                axes2[i].set_ylabel('Frequency')
                axes2[i].grid(True, alpha=0.3)
                
                # Add comfort zone shading
                axes2[i].axvspan(40, 60, alpha=0.2, color='green', label='Comfort Zone')
        
        # Hide unused subplots
        for i in range(len(self.available_columns), len(axes2)):
            axes2[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # 3. Box plots for outlier visualization
        if len(self.available_columns) > 0:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            box_plot = self.data[self.available_columns].boxplot(ax=ax3)
            ax3.set_title('Humidity Sensors Box Plots')
            ax3.set_xlabel('Sensor')
            ax3.set_ylabel('Relative Humidity (%)')
            ax3.axhspan(40, 60, alpha=0.2, color='green', label='Comfort Zone')
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
            
            ax4.set_title('Humidity Sensors Over Time (Sample Data)')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Relative Humidity (%)')
            ax4.axhspan(40, 60, alpha=0.2, color='green', label='Comfort Zone')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig4)
        
        # 5. Average humidity profile
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        avg_humidity = self.data[self.available_columns].mean().sort_values()
        bars = avg_humidity.plot(kind='bar', ax=ax5, color='cyan', alpha=0.7)
        ax5.set_title('Average Humidity by Sensor')
        ax5.set_xlabel('Sensor')
        ax5.set_ylabel('Average Relative Humidity (%)')
        ax5.axhspan(40, 60, alpha=0.2, color='green', label='Comfort Zone')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig5)
        
        # 6. Humidity distribution summary
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        
        # Create humidity range categories
        ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        range_counts = []
        
        for col in self.available_columns:
            col_ranges = []
            humidity_data = self.data[col].dropna()
            col_ranges.append(((humidity_data >= 0) & (humidity_data < 20)).sum())
            col_ranges.append(((humidity_data >= 20) & (humidity_data < 40)).sum())
            col_ranges.append(((humidity_data >= 40) & (humidity_data <= 60)).sum())
            col_ranges.append(((humidity_data > 60) & (humidity_data <= 80)).sum())
            col_ranges.append(((humidity_data > 80) & (humidity_data <= 100)).sum())
            range_counts.append(col_ranges)
        
        # Convert to percentages
        range_percentages = []
        for i, col in enumerate(self.available_columns):
            total = sum(range_counts[i])
            percentages = [(count/total)*100 if total > 0 else 0 for count in range_counts[i]]
            range_percentages.append(percentages)
        
        # Create stacked bar chart
        bottom = np.zeros(len(self.available_columns))
        colors = ['red', 'orange', 'green', 'yellow', 'blue']
        
        for i, range_name in enumerate(ranges):
            values = [range_percentages[j][i] for j in range(len(self.available_columns))]
            ax6.bar(self.available_columns, values, bottom=bottom, 
                   label=range_name, color=colors[i], alpha=0.7)
            bottom += values
        
        ax6.set_title('Humidity Range Distribution by Sensor (%)')
        ax6.set_xlabel('Sensor')
        ax6.set_ylabel('Percentage of Readings')
        ax6.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig6)
    
    def clean_data(self):
        """
        Clean the humidity sensor data based on analysis findings.
        """
        st.write("### Humidity Sensor Data Cleaning")
        
        initial_rows = len(self.data)
        
        for col in self.available_columns:
            st.write(f"**Cleaning {col}:**")
            
            # 1. Handle invalid humidity values (outside 0-100% range)
            invalid_mask = (self.data[col] < 0) | (self.data[col] > 100)
            if invalid_mask.sum() > 0:
                self.data.loc[invalid_mask, col] = np.nan
                st.write(f"- Set {invalid_mask.sum()} invalid values (outside 0-100%) to NaN")
                self.log_action(f"{col}: Set {invalid_mask.sum()} invalid values to NaN")
            
            # 2. Handle extreme but technically possible values with caution
            extreme_low_mask = (self.data[col] >= 0) & (self.data[col] < 5)  # Very dry
            extreme_high_mask = (self.data[col] <= 100) & (self.data[col] > 95)  # Very humid
            
            extreme_count = extreme_low_mask.sum() + extreme_high_mask.sum()
            if extreme_count > 0:
                st.write(f"- Found {extreme_count} extreme values (kept as potentially valid)")
                self.log_action(f"{col}: Found {extreme_count} extreme but valid values")
        
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
                    humidity_subset = self.data.loc[sufficient_data_mask, self.available_columns]
                    imputed_subset = pd.DataFrame(
                        imputer.fit_transform(humidity_subset),
                        index=humidity_subset.index,
                        columns=humidity_subset.columns
                    )
                    
                    # Ensure imputed values are within valid range
                    imputed_subset = imputed_subset.clip(0, 100)
                    
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
                    st.write(f"- {col}: Median imputed {remaining_missing_count} values ({median_value:.2f}%)")
                    self.log_action(f"{col}: Median imputed {remaining_missing_count} values")
        
        # 4. Outlier treatment (conservative approach for humidity sensors)
        st.write("**Outlier Treatment:**")
        for col in self.available_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use more conservative bounds (3*IQR instead of 1.5*IQR)
            lower_bound = max(0, Q1 - 3 * IQR)  # Cannot be below 0%
            upper_bound = min(100, Q3 + 3 * IQR)  # Cannot be above 100%
            
            outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            if outlier_mask.sum() > 0:
                # Cap outliers instead of removing (since they might be real extreme conditions)
                self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                st.write(f"- {col}: Capped {outlier_mask.sum()} extreme outliers")
                self.log_action(f"{col}: Capped {outlier_mask.sum()} extreme outliers")
        
        # 5. Final validation
        for col in self.available_columns:
            # Ensure all values are within 0-100% after cleaning
            invalid_final = ((self.data[col] < 0) | (self.data[col] > 100)).sum()
            if invalid_final > 0:
                st.error(f"- {col}: Still has {invalid_final} invalid values after cleaning!")
                self.log_action(f"{col}: {invalid_final} invalid values remain after cleaning", "ERROR")
        
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            st.write(f"Removed {removed_rows} rows during cleaning ({removed_rows/initial_rows*100:.2f}%)")
        else:
            st.write("No rows were removed during cleaning")
    
    def feature_engineering(self):
        """
        Create additional features from humidity sensor data.
        """
        st.write("### Humidity Feature Engineering")
        
        if len(self.available_columns) < 2:
            st.write("Need at least 2 humidity sensors for feature engineering")
            return
        
        # 1. Average internal humidity
        self.data['RH_internal_avg'] = self.data[self.available_columns].mean(axis=1)
        
        # 2. Humidity range (max - min)
        self.data['RH_internal_range'] = (self.data[self.available_columns].max(axis=1) - 
                                         self.data[self.available_columns].min(axis=1))
        
        # 3. Humidity standard deviation (uniformity)
        self.data['RH_internal_std'] = self.data[self.available_columns].std(axis=1)
        
        # 4. Number of sensors within comfort range (40-60%)
        comfort_range_mask = (self.data[self.available_columns] >= 40) & (self.data[self.available_columns] <= 60)
        self.data['RH_sensors_comfort_count'] = comfort_range_mask.sum(axis=1)
        
        # 5. Humidity zones classification
        if len(self.available_columns) >= 3:
            # Group sensors by humidity level
            humidity_medians = self.data[self.available_columns].median()
            
            dry_sensors = humidity_medians[humidity_medians <= humidity_medians.quantile(0.33)].index
            moderate_sensors = humidity_medians[(humidity_medians > humidity_medians.quantile(0.33)) & 
                                              (humidity_medians <= humidity_medians.quantile(0.67))].index
            humid_sensors = humidity_medians[humidity_medians > humidity_medians.quantile(0.67)].index
            
            if len(dry_sensors) > 0:
                self.data['RH_zone_dry_avg'] = self.data[dry_sensors].mean(axis=1)
            if len(moderate_sensors) > 0:
                self.data['RH_zone_moderate_avg'] = self.data[moderate_sensors].mean(axis=1)
            if len(humid_sensors) > 0:
                self.data['RH_zone_humid_avg'] = self.data[humid_sensors].mean(axis=1)
        
        # 6. Humidity trends (if enough temporal data)
        if len(self.data) > 12:
            # Moving averages
            self.data['RH_internal_ma6'] = self.data['RH_internal_avg'].rolling(window=6, min_periods=1).mean()
            self.data['RH_internal_ma12'] = self.data['RH_internal_avg'].rolling(window=12, min_periods=1).mean()
            
            # Humidity change rate
            self.data['RH_internal_change'] = self.data['RH_internal_avg'].diff()
            self.data['RH_internal_change_rate'] = self.data['RH_internal_change'].rolling(window=3, min_periods=1).mean()
        
        # 7. Comfort index based on humidity
        # Ideal humidity is around 45-55%
        optimal_humidity = 50
        humidity_deviation = abs(self.data['RH_internal_avg'] - optimal_humidity)
        self.data['RH_comfort_index'] = np.clip(1 - (humidity_deviation / 50), 0, 1)
        
        # 8. Humidity categories
        conditions = [
            self.data['RH_internal_avg'] < 30,  # Too dry
            (self.data['RH_internal_avg'] >= 30) & (self.data['RH_internal_avg'] < 40),  # Dry
            (self.data['RH_internal_avg'] >= 40) & (self.data['RH_internal_avg'] <= 60),  # Comfortable
            (self.data['RH_internal_avg'] > 60) & (self.data['RH_internal_avg'] <= 70),  # Humid
            self.data['RH_internal_avg'] > 70  # Too humid
        ]
        choices = ['Too_Dry', 'Dry', 'Comfortable', 'Humid', 'Too_Humid']
        self.data['RH_category'] = np.select(conditions, choices, default='Unknown')
        
        # 9. Mold risk indicator (high humidity areas)
        # High risk if humidity > 70% for extended periods
        high_humidity_mask = self.data['RH_internal_avg'] > 70
        if len(self.data) > 6:  # Need some temporal data
            # Rolling count of high humidity periods
            self.data['RH_mold_risk_score'] = high_humidity_mask.astype(int).rolling(
                window=6, min_periods=1
            ).sum() / 6  # Proportion of high humidity in last 6 periods
        else:
            self.data['RH_mold_risk_score'] = high_humidity_mask.astype(int)
        
        # 10. Humidity variability indicator
        # High variability might indicate poor HVAC control
        if 'RH_internal_std' in self.data.columns:
            self.data['RH_variability_category'] = pd.cut(
                self.data['RH_internal_std'],
                bins=[0, 5, 10, 20, 100],
                labels=['Low', 'Moderate', 'High', 'Very_High']
            )
        
        # List all new features
        base_features = ['RH_internal_avg', 'RH_internal_range', 'RH_internal_std', 
                        'RH_sensors_comfort_count', 'RH_comfort_index', 'RH_category',
                        'RH_mold_risk_score']
        new_features = base_features.copy()
        
        # Add zone features if they exist
        zone_features = [col for col in self.data.columns if col.startswith('RH_zone_')]
        new_features.extend(zone_features)
        
        # Add trend features if they exist
        if len(self.data) > 12:
            trend_features = ['RH_internal_ma6', 'RH_internal_ma12', 'RH_internal_change', 'RH_internal_change_rate']
            new_features.extend(trend_features)
        
        # Add variability category if it exists
        if 'RH_variability_category' in self.data.columns:
            new_features.append('RH_variability_category')
        
        st.write(f"Created {len(new_features)} humidity-related features:")
        for feature in new_features:
            if feature in self.data.columns:
                st.write(f"- {feature}")
        
        self.log_action(f"Created humidity features: {', '.join(new_features)}")
        
        # Show feature summary
        if 'RH_internal_avg' in self.data.columns:
            st.write(f"Average internal humidity: {self.data['RH_internal_avg'].mean():.2f}%")
            st.write(f"Humidity range variation: {self.data['RH_internal_range'].mean():.2f}%")
            st.write(f"Average humidity uniformity (std): {self.data['RH_internal_std'].mean():.2f}%")
            
            # Comfort analysis
            comfort_percentage = (self.data['RH_category'] == 'Comfortable').mean() * 100
            st.write(f"Percentage of time in comfort range: {comfort_percentage:.1f}%")
            
            # Category distribution
            if 'RH_category' in self.data.columns:
                st.write("Humidity category distribution:")
                st.write(self.data['RH_category'].value_counts(normalize=True) * 100)
    
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
            'message': f"[Person 4 - Humidity] {message}",
            'person': 4,
            'columns': self.columns
        }
        self.log.append(log_entry)