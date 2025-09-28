"""
Simple ETL Runner Script
Assignment 2: Data Cleaning and Preparation

Run this script directly with Python if you don't want to use Streamlit:
python run_etl.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Set matplotlib backend for better compatibility
import matplotlib
matplotlib.use('Agg')

def print_separator(title="", char="=", width=80):
    """Print a formatted separator with optional title."""
    if title:
        title_str = f" {title} "
        padding = (width - len(title_str)) // 2
        line = char * padding + title_str + char * padding
        if len(line) < width:
            line += char
    else:
        line = char * width
    print(line)

def run_etl_pipeline():
    """Run the complete ETL pipeline without Streamlit."""
    
    print_separator("ETL PIPELINE - DATA CLEANING AND PREPARATION")
    print("Assignment 2: Complete ETL Process")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Check if dataset exists
    dataset_path = 'dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset '{dataset_path}' not found!")
        print("Please ensure the dataset.csv file is in the current directory.")
        return False
    
    try:
        # Import ETL modules
        from etl_person1_temporal import TemporalDataAnalyzer
        from etl_person2_lighting import LightingAnalyzer
        from etl_person3_temp_sensors import TempSensorAnalyzer
        from etl_person4_humidity_sensors import HumiditySensorAnalyzer
        from etl_person5_weather_part1 import WeatherPart1Analyzer
        from etl_person6_weather_part2 import WeatherPart2Analyzer
        
        print("✅ All ETL modules imported successfully")
        
    except ImportError as e:
        print(f"❌ Error importing ETL modules: {e}")
        print("Please ensure all ETL module files are in the current directory.")
        return False
    
    # Load original dataset
    print_separator("LOADING DATASET")
    print(f"Loading dataset: {dataset_path}")
    
    try:
        original_data = pd.read_csv(dataset_path)
        processed_data = original_data.copy()
        
        print(f"✅ Dataset loaded successfully")
        print(f"   - Shape: {original_data.shape} (rows, columns)")
        print(f"   - Size: {original_data.shape[0] * original_data.shape[1]:,} total cells")
        print(f"   - Memory usage: {original_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Show column info
        print(f"\n📊 Dataset Overview:")
        print(f"   - Columns: {list(original_data.columns)}")
        
        # Check for missing values
        missing_values = original_data.isnull().sum().sum()
        if missing_values > 0:
            print(f"   - Missing values: {missing_values:,}")
        else:
            print(f"   - Missing values: None ✅")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Initialize processing log
    processing_log = []
    
    def log_action(message, level="INFO"):
        """Log processing actions."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        processing_log.append(log_entry)
        print(f"   {log_entry}")
    
    # Run individual analyses
    print_separator("INDIVIDUAL ANALYSES")
    
    # Person 1: Temporal Data
    print_separator("Person 1: Temporal Data Analysis", "-")
    print("📅 Analyzing temporal data (Appliances, date)...")
    
    try:
        person1 = TemporalDataAnalyzer(processed_data)
        processed_data = person1.analyze_and_clean()
        
        # Extract and display key findings
        for entry in person1.get_log():
            log_action(entry['message'], entry['level'])
            
        print(f"✅ Person 1 analysis completed")
        
    except Exception as e:
        print(f"❌ Error in Person 1 analysis: {e}")
        log_action(f"Person 1 analysis failed: {e}", "ERROR")
    
    # Person 2: Lighting System
    print_separator("Person 2: Lighting System Analysis", "-")
    print("💡 Analyzing lighting system data...")
    
    try:
        person2 = LightingAnalyzer(processed_data)
        processed_data = person2.analyze_and_clean()
        
        for entry in person2.get_log():
            log_action(entry['message'], entry['level'])
            
        print(f"✅ Person 2 analysis completed")
        
    except Exception as e:
        print(f"❌ Error in Person 2 analysis: {e}")
        log_action(f"Person 2 analysis failed: {e}", "ERROR")
    
    # Person 3: Temperature Sensors
    print_separator("Person 3: Internal Temperature Sensors", "-")
    print("🌡️ Analyzing internal temperature sensors (T1-T9)...")
    
    try:
        person3 = TempSensorAnalyzer(processed_data)
        processed_data = person3.analyze_and_clean()
        
        for entry in person3.get_log():
            log_action(entry['message'], entry['level'])
            
        print(f"✅ Person 3 analysis completed")
        
    except Exception as e:
        print(f"❌ Error in Person 3 analysis: {e}")
        log_action(f"Person 3 analysis failed: {e}", "ERROR")
    
    # Person 4: Humidity Sensors
    print_separator("Person 4: Internal Humidity Sensors", "-")
    print("💧 Analyzing internal humidity sensors (RH_1-RH_9)...")
    
    try:
        person4 = HumiditySensorAnalyzer(processed_data)
        processed_data = person4.analyze_and_clean()
        
        for entry in person4.get_log():
            log_action(entry['message'], entry['level'])
            
        print(f"✅ Person 4 analysis completed")
        
    except Exception as e:
        print(f"❌ Error in Person 4 analysis: {e}")
        log_action(f"Person 4 analysis failed: {e}", "ERROR")
    
    # Person 5: External Weather Part 1
    print_separator("Person 5: External Weather Data (Part 1)", "-")
    print("🌤️ Analyzing external weather data part 1 (T_out, RH_out, Tdewpoint)...")
    
    try:
        person5 = WeatherPart1Analyzer(processed_data)
        processed_data = person5.analyze_and_clean()
        
        for entry in person5.get_log():
            log_action(entry['message'], entry['level'])
            
        print(f"✅ Person 5 analysis completed")
        
    except Exception as e:
        print(f"❌ Error in Person 5 analysis: {e}")
        log_action(f"Person 5 analysis failed: {e}", "ERROR")
    
    # Person 6: External Weather Part 2
    print_separator("Person 6: External Weather Data (Part 2)", "-")
    print("🌪️ Analyzing external weather data part 2 (Pressure, Wind speed, Visibility)...")
    
    try:
        person6 = WeatherPart2Analyzer(processed_data)
        processed_data = person6.analyze_and_clean()
        
        for entry in person6.get_log():
            log_action(entry['message'], entry['level'])
            
        print(f"✅ Person 6 analysis completed")
        
    except Exception as e:
        print(f"❌ Error in Person 6 analysis: {e}")
        log_action(f"Person 6 analysis failed: {e}", "ERROR")
    
    # External data integration
    print_separator("EXTERNAL DATA INTEGRATION", "-")
    print("🔗 Integrating external data sources...")
    
    # Add temporal features from existing date column
    if 'date' in processed_data.columns:
        try:
            processed_data['date'] = pd.to_datetime(processed_data['date'], format='%d-%m-%Y %H:%M')
            
            # Extract temporal features
            processed_data['hour'] = processed_data['date'].dt.hour
            processed_data['day_of_week'] = processed_data['date'].dt.dayofweek
            processed_data['month'] = processed_data['date'].dt.month
            processed_data['is_weekend'] = (processed_data['day_of_week'] >= 5).astype(int)
            
            log_action("Added temporal features: hour, day_of_week, month, is_weekend")
            print("✅ External data integration completed")
            
        except Exception as e:
            log_action(f"External data integration failed: {e}", "ERROR")
            print(f"❌ Error in external data integration: {e}")
    
    # Final quality assessment
    print_separator("FINAL QUALITY ASSESSMENT")
    print("🔍 Performing final data quality checks...")
    
    # Check for remaining null values
    null_counts = processed_data.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        print(f"⚠️ Remaining null values: {total_nulls:,}")
        null_columns = null_counts[null_counts > 0]
        for col, count in null_columns.items():
            print(f"   - {col}: {count} nulls")
    else:
        print("✅ No null values in final dataset")
    
    # Check for duplicates
    duplicates = processed_data.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ Duplicate rows: {duplicates:,}")
    else:
        print("✅ No duplicate rows found")
    
    # Final statistics
    print(f"\n📊 Final Dataset Statistics:")
    print(f"   - Original shape: {original_data.shape}")
    print(f"   - Final shape: {processed_data.shape}")
    print(f"   - Rows change: {processed_data.shape[0] - original_data.shape[0]:+d}")
    print(f"   - Columns change: {processed_data.shape[1] - original_data.shape[1]:+d}")
    
    # New columns created
    new_columns = set(processed_data.columns) - set(original_data.columns)
    if new_columns:
        print(f"   - New columns added: {len(new_columns)}")
        for col in sorted(new_columns):
            print(f"     • {col}")
    
    # Save final dataset
    print_separator("SAVING FINAL DATASET")
    output_filename = 'dataset_final_cleaned.csv'
    
    try:
        processed_data.to_csv(output_filename, index=False)
        file_size = os.path.getsize(output_filename) / (1024 * 1024)
        print(f"✅ Final dataset saved successfully")
        print(f"   - Filename: {output_filename}")
        print(f"   - File size: {file_size:.2f} MB")
        print(f"   - Records: {len(processed_data):,}")
        print(f"   - Columns: {len(processed_data.columns)}")
        
        log_action(f"Final dataset saved: {output_filename}")
        
    except Exception as e:
        print(f"❌ Error saving final dataset: {e}")
        log_action(f"Failed to save dataset: {e}", "ERROR")
        return False
    
    # Generate processing report
    print_separator("PROCESSING REPORT")
    
    # Save processing log
    log_filename = 'etl_processing_log.txt'
    try:
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write("ETL PROCESSING LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Original shape: {original_data.shape}\n")
            f.write(f"Final shape: {processed_data.shape}\n")
            f.write("\nPROCESSING STEPS:\n")
            f.write("-" * 30 + "\n")
            
            for log_entry in processing_log:
                f.write(log_entry + "\n")
        
        print(f"📋 Processing log saved: {log_filename}")
        
    except Exception as e:
        print(f"⚠️ Could not save processing log: {e}")
    
    # Summary
    print_separator()
    print("🎉 ETL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files:")
    print(f"  - Cleaned dataset: {output_filename}")
    print(f"  - Processing log: {log_filename}")
    print_separator()
    
    return True

def main():
    """Main function."""
    try:
        success = run_etl_pipeline()
        
        if success:
            print("\n✅ All processing completed successfully!")
            print("You can now use the cleaned dataset for your analysis and modeling.")
        else:
            print("\n❌ ETL pipeline failed!")
            print("Please check the error messages above and try again.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()