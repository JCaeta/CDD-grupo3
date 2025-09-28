"""
ETL Main Pipeline - Data Cleaning and Preparation
Assignment 2: Complete ETL Process for Energy Consumption Dataset

This module orchestrates the entire ETL process, coordinating individual
analyses and creating the final clean dataset.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import individual analysis modules
from etl_person1_temporal import TemporalDataAnalyzer
from etl_person2_lighting import LightingAnalyzer
from etl_person3_temp_sensors import TempSensorAnalyzer
from etl_person4_humidity_sensors import HumiditySensorAnalyzer
from etl_person5_weather_part1 import WeatherPart1Analyzer
from etl_person6_weather_part2 import WeatherPart2Analyzer

class ETLPipeline:
    """
    Main ETL Pipeline class that coordinates the entire data cleaning process.
    """
    
    def __init__(self, dataset_path='dataset.csv'):
        """
        Initialize the ETL pipeline.
        
        Args:
            dataset_path (str): Path to the input dataset
        """
        self.dataset_path = dataset_path
        self.original_data = None
        self.processed_data = None
        self.cleaning_log = []
        
    def load_data(self):
        """Load the original dataset."""
        try:
            self.original_data = pd.read_csv(self.dataset_path)
            self.processed_data = self.original_data.copy()
            
            # Log basic info
            self.log_action(f"Dataset loaded successfully: {self.original_data.shape}")
            self.log_action(f"Columns: {list(self.original_data.columns)}")
            
            return True
        except Exception as e:
            self.log_action(f"Error loading dataset: {str(e)}", "ERROR")
            return False
    
    def run_individual_analyses(self):
        """
        Run individual analyses for each person's assigned columns.
        """
        st.header("Individual Column Analyses")
        
        # Person 1: Temporal Data (Appliances, date)
        st.subheader("Person 1: Temporal Data Analysis")
        person1 = TemporalDataAnalyzer(self.processed_data)
        self.processed_data = person1.analyze_and_clean()
        self.merge_log(person1.get_log())
        
        # Person 2: Lighting System (lights)
        st.subheader("Person 2: Lighting System Analysis")
        person2 = LightingAnalyzer(self.processed_data)
        self.processed_data = person2.analyze_and_clean()
        self.merge_log(person2.get_log())
        
        # Person 3: Internal Temperature Sensors (T1-T9)
        st.subheader("Person 3: Internal Temperature Sensors Analysis")
        person3 = TempSensorAnalyzer(self.processed_data)
        self.processed_data = person3.analyze_and_clean()
        self.merge_log(person3.get_log())
        
        # Person 4: Internal Humidity Sensors (RH_1-RH_9)
        st.subheader("Person 4: Internal Humidity Sensors Analysis")
        person4 = HumiditySensorAnalyzer(self.processed_data)
        self.processed_data = person4.analyze_and_clean()
        self.merge_log(person4.get_log())
        
        # Person 5: External Weather Part 1 (T_out, RH_out, Tdewpoint)
        st.subheader("Person 5: External Weather Data - Part 1")
        person5 = WeatherPart1Analyzer(self.processed_data)
        self.processed_data = person5.analyze_and_clean()
        self.merge_log(person5.get_log())
        
        # Person 6: External Weather Part 2 (Pressure, Wind speed, Visibility)
        st.subheader("Person 6: External Weather Data - Part 2")
        person6 = WeatherPart2Analyzer(self.processed_data)
        self.processed_data = person6.analyze_and_clean()
        self.merge_log(person6.get_log())
    
    def integrate_external_data(self):
        """
        Integrate external data sources if available.
        This section can be expanded based on available external datasets.
        """
        st.subheader("External Data Integration")
        
        # Placeholder for external data integration
        # This could include weather station data, energy price data, etc.
        self.log_action("External data integration - No additional sources specified")
        
        # Add temporal features from existing date column
        if 'date' in self.processed_data.columns:
            self.processed_data['date'] = pd.to_datetime(self.processed_data['date'], 
                                                       format='%d-%m-%Y %H:%M')
            
            # Extract temporal features
            self.processed_data['hour'] = self.processed_data['date'].dt.hour
            self.processed_data['day_of_week'] = self.processed_data['date'].dt.dayofweek
            self.processed_data['month'] = self.processed_data['date'].dt.month
            self.processed_data['is_weekend'] = (self.processed_data['day_of_week'] >= 5).astype(int)
            
            self.log_action("Added temporal features: hour, day_of_week, month, is_weekend")
    
    def final_quality_check(self):
        """
        Perform final data quality checks on the processed dataset.
        """
        st.subheader("Final Data Quality Assessment")
        
        # Check for remaining null values
        null_counts = self.processed_data.isnull().sum()
        if null_counts.sum() > 0:
            st.warning(f"Remaining null values found in: {null_counts[null_counts > 0].to_dict()}")
        else:
            st.success("No null values in final dataset")
        
        # Check for duplicates
        duplicates = self.processed_data.duplicated().sum()
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate rows")
        else:
            st.success("No duplicate rows found")
        
        # Data type validation
        st.write("Final data types:")
        st.write(self.processed_data.dtypes)
        
        # Basic statistics
        st.write("Final dataset statistics:")
        st.write(self.processed_data.describe())
        
        self.log_action(f"Final dataset shape: {self.processed_data.shape}")
    
    def save_final_dataset(self, output_path='dataset_final_cleaned.csv'):
        """
        Save the final cleaned dataset.
        
        Args:
            output_path (str): Path for the output file
        """
        try:
            self.processed_data.to_csv(output_path, index=False)
            self.log_action(f"Final dataset saved to: {output_path}")
            st.success(f"Dataset saved successfully to {output_path}")
            return True
        except Exception as e:
            self.log_action(f"Error saving dataset: {str(e)}", "ERROR")
            st.error(f"Error saving dataset: {str(e)}")
            return False
    
    def generate_report(self):
        """
        Generate a comprehensive ETL report.
        """
        st.header("ETL Process Report")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Original Records", len(self.original_data))
            st.metric("Original Columns", len(self.original_data.columns))
            
        with col2:
            st.metric("Final Records", len(self.processed_data))
            st.metric("Final Columns", len(self.processed_data.columns))
        
        # Processing log
        st.subheader("Processing Log")
        for entry in self.cleaning_log:
            if entry.get('level') == 'ERROR':
                st.error(entry['message'])
            elif entry.get('level') == 'WARNING':
                st.warning(entry['message'])
            else:
                st.info(entry['message'])
    
    def log_action(self, message, level="INFO"):
        """Log an action with timestamp."""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'level': level,
            'message': message
        }
        self.cleaning_log.append(log_entry)
    
    def merge_log(self, external_log):
        """Merge log from individual analyzers."""
        self.cleaning_log.extend(external_log)
    
    def run_complete_etl(self):
        """
        Run the complete ETL pipeline.
        """
        st.title("ETL Pipeline - Data Cleaning and Preparation")
        st.write("Assignment 2: Complete ETL Process")
        
        # Load data
        if not self.load_data():
            st.error("Failed to load dataset")
            return False
        
        # Show original data overview
        st.header("Original Dataset Overview")
        st.write(f"Shape: {self.original_data.shape}")
        st.write("First 5 rows:")
        st.dataframe(self.original_data.head())
        
        # Run individual analyses
        self.run_individual_analyses()
        
        # Integrate external data
        self.integrate_external_data()
        
        # Final quality check
        self.final_quality_check()
        
        # Save final dataset
        self.save_final_dataset()
        
        # Generate report
        self.generate_report()
        
        return True

def main():
    """Main function to run the ETL pipeline in Streamlit."""
    etl = ETLPipeline()
    etl.run_complete_etl()

if __name__ == "__main__":
    # For Streamlit app
    if 'streamlit' in locals():
        main()
    else:
        # For direct execution
        etl = ETLPipeline()
        etl.run_complete_etl()