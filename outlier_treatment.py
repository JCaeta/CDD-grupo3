"""
Outlier Detection and Treatment Module
=====================================

This module provides comprehensive outlier detection and treatment capabilities
for the ETL pipeline. It implements multiple methods for outlier detection
and various treatment strategies.

Author: ETL Team
Date: 28 de septiembre de 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    Comprehensive outlier detection and treatment class.
    """
    
    def __init__(self, data, method='iqr', contamination=0.1):
        """
        Initialize the OutlierDetector.
        
        Args:
            data (pd.DataFrame): Input dataset
            method (str): Detection method ('iqr', 'zscore', 'isolation_forest', 'modified_zscore')
            contamination (float): Expected proportion of outliers (for isolation forest)
        """
        self.data = data.copy()
        self.method = method
        self.contamination = contamination
        self.outlier_info = {}
        self.treatment_log = []
        
    def log_action(self, message, level="INFO"):
        """Log actions for tracking."""
        self.treatment_log.append(f"[{level}] {message}")
        print(f"[{level}] {message}")
    
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            column (str): Column name to analyze
            multiplier (float): IQR multiplier (default 1.5, stricter: 1.0, looser: 2.0)
        
        Returns:
            dict: Outlier detection results
        """
        if column not in self.data.columns:
            return None
            
        # Calculate quartiles and IQR
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Identify outliers
        outliers = (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
        
        # Calculate whisker values (actual data points within bounds)
        lower_whisker = self.data[column][self.data[column] >= lower_bound].min()
        upper_whisker = self.data[column][self.data[column] <= upper_bound].max()
        
        return {
            'method': 'IQR',
            'column': column,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'lower_whisker': lower_whisker,
            'upper_whisker': upper_whisker,
            'outlier_mask': outliers,
            'outlier_count': outliers.sum(),
            'outlier_percentage': (outliers.sum() / len(self.data)) * 100
        }
    
    def detect_outliers_zscore(self, column, threshold=3):
        """
        Detect outliers using Z-Score method.
        
        Args:
            column (str): Column name to analyze
            threshold (float): Z-score threshold (default 3)
        
        Returns:
            dict: Outlier detection results
        """
        if column not in self.data.columns:
            return None
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(self.data[column].dropna()))
        outliers = z_scores > threshold
        
        # Expand to full dataset size (handle NaN)
        full_outliers = pd.Series(False, index=self.data.index)
        full_outliers.loc[self.data[column].dropna().index] = outliers
        
        return {
            'method': 'Z-Score',
            'column': column,
            'threshold': threshold,
            'mean': self.data[column].mean(),
            'std': self.data[column].std(),
            'outlier_mask': full_outliers,
            'outlier_count': full_outliers.sum(),
            'outlier_percentage': (full_outliers.sum() / len(self.data)) * 100
        }
    
    def detect_outliers_modified_zscore(self, column, threshold=3.5):
        """
        Detect outliers using Modified Z-Score (using median and MAD).
        
        Args:
            column (str): Column name to analyze
            threshold (float): Modified z-score threshold (default 3.5)
        
        Returns:
            dict: Outlier detection results
        """
        if column not in self.data.columns:
            return None
        
        # Calculate modified z-scores using median and MAD
        median = self.data[column].median()
        mad = np.median(np.abs(self.data[column] - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = 1.4826 * np.median(np.abs(self.data[column] - median))
        
        modified_z_scores = 0.6745 * (self.data[column] - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        
        return {
            'method': 'Modified Z-Score',
            'column': column,
            'threshold': threshold,
            'median': median,
            'mad': mad,
            'outlier_mask': outliers,
            'outlier_count': outliers.sum(),
            'outlier_percentage': (outliers.sum() / len(self.data)) * 100
        }
    
    def detect_outliers_isolation_forest(self, columns, contamination=None):
        """
        Detect outliers using Isolation Forest (multivariate).
        
        Args:
            columns (list): List of columns to analyze together
            contamination (float): Expected proportion of outliers
        
        Returns:
            dict: Outlier detection results
        """
        if contamination is None:
            contamination = self.contamination
        
        # Select and prepare data
        data_subset = self.data[columns].dropna()
        
        if len(data_subset) == 0:
            return None
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers_subset = iso_forest.fit_predict(scaled_data) == -1
        
        # Map back to full dataset
        full_outliers = pd.Series(False, index=self.data.index)
        full_outliers.loc[data_subset.index] = outliers_subset
        
        return {
            'method': 'Isolation Forest',
            'columns': columns,
            'contamination': contamination,
            'outlier_mask': full_outliers,
            'outlier_count': full_outliers.sum(),
            'outlier_percentage': (full_outliers.sum() / len(self.data)) * 100
        }
    
    def analyze_column_outliers(self, column, methods=['iqr', 'zscore'], **kwargs):
        """
        Analyze outliers in a specific column using multiple methods.
        
        Args:
            column (str): Column to analyze
            methods (list): Methods to use ['iqr', 'zscore', 'modified_zscore']
            **kwargs: Additional parameters for specific methods
        
        Returns:
            dict: Results from all methods
        """
        results = {}
        
        for method in methods:
            if method == 'iqr':
                multiplier = kwargs.get('iqr_multiplier', 1.5)
                results[method] = self.detect_outliers_iqr(column, multiplier)
            elif method == 'zscore':
                threshold = kwargs.get('zscore_threshold', 3)
                results[method] = self.detect_outliers_zscore(column, threshold)
            elif method == 'modified_zscore':
                threshold = kwargs.get('mod_zscore_threshold', 3.5)
                results[method] = self.detect_outliers_modified_zscore(column, threshold)
        
        # Store results
        self.outlier_info[column] = results
        
        # Log analysis
        self.log_action(f"Analyzed outliers in '{column}' using methods: {methods}")
        
        return results
    
    def treat_outliers(self, column, method='cap', detection_method='iqr', **kwargs):
        """
        Treat outliers in a specific column.
        
        Args:
            column (str): Column to treat
            method (str): Treatment method ('remove', 'cap', 'median', 'mean', 'interpolate')
            detection_method (str): Which detection method results to use
            **kwargs: Additional parameters
        
        Returns:
            pd.DataFrame: Treated data
        """
        if column not in self.outlier_info or detection_method not in self.outlier_info[column]:
            # Run detection first
            self.analyze_column_outliers(column, [detection_method], **kwargs)
        
        outlier_info = self.outlier_info[column][detection_method]
        outlier_mask = outlier_info['outlier_mask']
        
        original_count = len(self.data)
        outlier_count = outlier_mask.sum()
        
        if outlier_count == 0:
            self.log_action(f"No outliers found in '{column}' using {detection_method} method")
            return self.data
        
        self.log_action(f"Treating {outlier_count} outliers in '{column}' using '{method}' method")
        
        if method == 'remove':
            # Remove rows with outliers
            self.data = self.data[~outlier_mask]
            self.log_action(f"Removed {outlier_count} rows. Dataset size: {original_count} -> {len(self.data)}")
            
        elif method == 'cap':
            # Cap outliers to whisker values or bounds
            if detection_method == 'iqr':
                lower_bound = outlier_info['lower_whisker']
                upper_bound = outlier_info['upper_whisker']
            else:
                # For other methods, use percentiles
                lower_bound = self.data[column].quantile(0.05)
                upper_bound = self.data[column].quantile(0.95)
            
            self.data.loc[self.data[column] < lower_bound, column] = lower_bound
            self.data.loc[self.data[column] > upper_bound, column] = upper_bound
            self.log_action(f"Capped outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")
            
        elif method == 'median':
            # Replace outliers with median
            median_value = self.data[column].median()
            self.data.loc[outlier_mask, column] = median_value
            self.log_action(f"Replaced outliers with median value: {median_value:.2f}")
            
        elif method == 'mean':
            # Replace outliers with mean (excluding outliers)
            clean_mean = self.data.loc[~outlier_mask, column].mean()
            self.data.loc[outlier_mask, column] = clean_mean
            self.log_action(f"Replaced outliers with clean mean: {clean_mean:.2f}")
            
        elif method == 'interpolate':
            # Interpolate outliers
            self.data.loc[outlier_mask, column] = np.nan
            self.data[column] = self.data[column].interpolate(method='linear')
            self.log_action(f"Interpolated {outlier_count} outlier values")
        
        return self.data
    
    def visualize_outliers(self, column, figsize=(15, 5)):
        """
        Create visualizations for outlier analysis.
        
        Args:
            column (str): Column to visualize
            figsize (tuple): Figure size
        """
        if column not in self.outlier_info:
            self.analyze_column_outliers(column)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Boxplot
        self.data.boxplot(column=column, ax=axes[0])
        axes[0].set_title(f'Boxplot - {column}')
        axes[0].grid(True)
        
        # Histogram with outlier regions
        axes[1].hist(self.data[column].dropna(), bins=50, alpha=0.7, edgecolor='black')
        
        # Add outlier bounds if IQR method was used
        if 'iqr' in self.outlier_info[column]:
            iqr_info = self.outlier_info[column]['iqr']
            axes[1].axvline(iqr_info['lower_bound'], color='red', linestyle='--', 
                           label=f"Lower Bound: {iqr_info['lower_bound']:.2f}")
            axes[1].axvline(iqr_info['upper_bound'], color='red', linestyle='--', 
                           label=f"Upper Bound: {iqr_info['upper_bound']:.2f}")
            axes[1].legend()
        
        axes[1].set_title(f'Distribution - {column}')
        axes[1].set_xlabel(column)
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)
        
        # Q-Q plot
        stats.probplot(self.data[column].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot - {column}')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_outlier_report(self, columns=None):
        """
        Generate a comprehensive outlier analysis report.
        
        Args:
            columns (list): Columns to analyze (None for all numeric columns)
        
        Returns:
            pd.DataFrame: Summary report
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        report_data = []
        
        for column in columns:
            if column not in self.outlier_info:
                self.analyze_column_outliers(column)
            
            for method, info in self.outlier_info[column].items():
                report_data.append({
                    'Column': column,
                    'Method': info['method'],
                    'Outlier_Count': info['outlier_count'],
                    'Outlier_Percentage': f"{info['outlier_percentage']:.2f}%",
                    'Details': self._format_method_details(info)
                })
        
        return pd.DataFrame(report_data)
    
    def _format_method_details(self, info):
        """Format method-specific details for the report."""
        if info['method'] == 'IQR':
            return f"Q1: {info['Q1']:.2f}, Q3: {info['Q3']:.2f}, IQR: {info['IQR']:.2f}"
        elif info['method'] in ['Z-Score', 'Modified Z-Score']:
            return f"Threshold: {info['threshold']}"
        elif info['method'] == 'Isolation Forest':
            return f"Contamination: {info['contamination']:.3f}"
        return ""
    
    def get_treatment_log(self):
        """Return the treatment log."""
        return self.treatment_log


class OutlierTreatmentPipeline:
    """
    Pipeline class for systematic outlier treatment across multiple columns.
    """
    
    def __init__(self, data):
        """Initialize the pipeline."""
        self.data = data.copy()
        self.original_data = data.copy()
        self.detectors = {}
        self.treatment_history = []
    
    def auto_treat_numeric_columns(self, method='iqr', treatment='cap', exclude_columns=None):
        """
        Automatically treat outliers in all numeric columns.
        
        Args:
            method (str): Detection method
            treatment (str): Treatment method
            exclude_columns (list): Columns to exclude from treatment
        """
        if exclude_columns is None:
            exclude_columns = []
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        columns_to_treat = [col for col in numeric_columns if col not in exclude_columns]
        
        for column in columns_to_treat:
            detector = OutlierDetector(self.data, method=method)
            treated_data = detector.treat_outliers(column, method=treatment, detection_method=method)
            self.data = treated_data
            self.detectors[column] = detector
            
            # Log treatment
            self.treatment_history.append({
                'column': column,
                'detection_method': method,
                'treatment_method': treatment,
                'outliers_found': detector.outlier_info[column][method]['outlier_count']
            })
    
    def get_summary_report(self):
        """Get a summary of all treatments applied."""
        if not self.treatment_history:
            return "No treatments have been applied yet."
        
        summary_df = pd.DataFrame(self.treatment_history)
        
        print("=== OUTLIER TREATMENT SUMMARY ===")
        print(f"Original dataset shape: {self.original_data.shape}")
        print(f"Final dataset shape: {self.data.shape}")
        print(f"Rows removed: {len(self.original_data) - len(self.data)}")
        print("\nTreatment by column:")
        print(summary_df.to_string(index=False))
        
        return summary_df


# Example usage and testing functions
def demo_outlier_treatment(dataset_path='dataset.csv'):
    """
    Demonstrate outlier treatment capabilities.
    
    Args:
        dataset_path (str): Path to the dataset
    """
    print("=== OUTLIER TREATMENT DEMONSTRATION ===")
    
    # Load data
    try:
        data = pd.read_csv(dataset_path)
        print(f"Dataset loaded: {data.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize pipeline
    pipeline = OutlierTreatmentPipeline(data)
    
    # Example 1: Analyze specific column
    print("\n1. Analyzing outliers in 'Appliances' column:")
    detector = OutlierDetector(data)
    results = detector.analyze_column_outliers('Appliances', methods=['iqr', 'zscore'])
    
    for method, info in results.items():
        print(f"   {method}: {info['outlier_count']} outliers ({info['outlier_percentage']:.2f}%)")
    
    # Example 2: Treat outliers in a specific column
    print("\n2. Treating outliers in 'Appliances' using capping:")
    detector.treat_outliers('Appliances', method='cap', detection_method='iqr')
    
    # Example 3: Auto-treat all numeric columns
    print("\n3. Auto-treating all numeric columns:")
    pipeline.auto_treat_numeric_columns(method='iqr', treatment='cap', 
                                       exclude_columns=['date'])
    
    # Generate summary
    summary = pipeline.get_summary_report()
    
    return pipeline.data


if __name__ == "__main__":
    # Run demonstration
    demo_outlier_treatment()