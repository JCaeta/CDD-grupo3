"""
Streamlit App for ETL Pipeline
Assignment 2: Data Cleaning and Preparation

Run this file with: streamlit run streamlit_etl_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(__file__))

try:
    from etl_main import ETLPipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all ETL modules are in the same directory as this app")
    st.stop()

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="ETL Pipeline - Data Cleaning & Preparation",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .person-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-container {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<h1 class="main-header">🔧 ETL Pipeline - Data Cleaning & Preparation</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.title("ETL Configuration")
    st.sidebar.markdown("---")
    
    # File upload or path selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Use default dataset.csv", "Upload custom file"]
    )
    
    dataset_path = None
    
    if data_source == "Upload custom file":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your dataset CSV file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            dataset_path = "temp_dataset.csv"
            with open(dataset_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success("File uploaded successfully!")
        else:
            st.sidebar.warning("Please upload a CSV file to proceed")
            return
    else:
        dataset_path = "dataset.csv"
        if not os.path.exists(dataset_path):
            st.sidebar.error(f"Default dataset '{dataset_path}' not found in current directory")
            return
    
    # ETL Options
    st.sidebar.markdown("### ETL Options")
    
    run_individual_analyses = st.sidebar.checkbox("Run Individual Analyses", value=True)
    save_intermediate_results = st.sidebar.checkbox("Save Intermediate Results", value=False)
    show_detailed_logs = st.sidebar.checkbox("Show Detailed Logs", value=True)
    
    output_filename = st.sidebar.text_input(
        "Output Filename", 
        value="dataset_final_cleaned.csv",
        help="Name for the final cleaned dataset"
    )
    
    # Assignment information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Assignment Information")
    st.sidebar.markdown("""
    **Individual Responsibilities:**
    
    - **Person 1:** Temporal Data (Appliances, date)
    - **Person 2:** Lighting System (lights)
    - **Person 3:** Internal Temperature Sensors (T1-T9)
    - **Person 4:** Internal Humidity Sensors (RH_1-RH_9)
    - **Person 5:** External Weather Part 1 (T_out, RH_out, Tdewpoint)
    - **Person 6:** External Weather Part 2 (Pressure, Wind speed, Visibility)
    """)
    
    # Main content area
    if st.sidebar.button("🚀 Run ETL Pipeline", type="primary"):
        run_etl_pipeline(dataset_path, output_filename, run_individual_analyses, 
                        save_intermediate_results, show_detailed_logs)
    
    # Information tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Data Info", "🔍 Preview", "📚 Documentation"])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_data_info(dataset_path)
    
    with tab3:
        show_data_preview(dataset_path)
    
    with tab4:
        show_documentation()

def run_etl_pipeline(dataset_path, output_filename, run_individual, save_intermediate, show_logs):
    """Run the complete ETL pipeline."""
    
    st.markdown('<h2 class="section-header">🔄 Running ETL Pipeline</h2>', unsafe_allow_html=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize ETL Pipeline
        status_text.text("Initializing ETL Pipeline...")
        progress_bar.progress(10)
        
        etl = ETLPipeline(dataset_path)
        
        # Load data
        status_text.text("Loading dataset...")
        progress_bar.progress(20)
        
        if not etl.load_data():
            st.error("Failed to load dataset")
            return
        
        # Show original data info
        st.success(f"✅ Dataset loaded successfully: {etl.original_data.shape}")
        
        with st.expander("📊 Original Dataset Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{len(etl.original_data):,}")
            with col2:
                st.metric("Total Columns", len(etl.original_data.columns))
            with col3:
                st.metric("Total Cells", f"{len(etl.original_data) * len(etl.original_data.columns):,}")
            
            st.write("**Dataset Preview:**")
            st.dataframe(etl.original_data.head(), use_container_width=True)
            
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': etl.original_data.columns,
                'Data Type': etl.original_data.dtypes,
                'Non-Null Count': etl.original_data.count(),
                'Null Count': etl.original_data.isnull().sum(),
                'Null Percentage': (etl.original_data.isnull().sum() / len(etl.original_data) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
        
        if run_individual:
            # Run individual analyses
            status_text.text("Running individual analyses...")
            progress_bar.progress(30)
            
            st.markdown('<h3 class="section-header">👥 Individual Analyses</h3>', unsafe_allow_html=True)
            
            # Import individual analyzers
            from etl_person1_temporal import TemporalDataAnalyzer
            from etl_person2_lighting import LightingAnalyzer
            from etl_person3_temp_sensors import TempSensorAnalyzer
            from etl_person4_humidity_sensors import HumiditySensorAnalyzer
            from etl_person5_weather_part1 import WeatherPart1Analyzer
            from etl_person6_weather_part2 import WeatherPart2Analyzer
            
            # Person 1: Temporal Data
            status_text.text("Running Person 1 analysis (Temporal Data)...")
            progress_bar.progress(35)
            
            with st.expander("👤 Person 1: Temporal Data Analysis (Appliances, date)", expanded=False):
                person1 = TemporalDataAnalyzer(etl.processed_data)
                etl.processed_data = person1.analyze_and_clean()
                etl.merge_log(person1.get_log())
                
                if save_intermediate:
                    temp_file = "intermediate_person1.csv"
                    etl.processed_data.to_csv(temp_file, index=False)
                    st.success(f"Intermediate results saved to {temp_file}")
            
            # Person 2: Lighting
            status_text.text("Running Person 2 analysis (Lighting System)...")
            progress_bar.progress(45)
            
            with st.expander("👤 Person 2: Lighting System Analysis (lights)", expanded=False):
                person2 = LightingAnalyzer(etl.processed_data)
                etl.processed_data = person2.analyze_and_clean()
                etl.merge_log(person2.get_log())
            
            # Person 3: Temperature Sensors
            status_text.text("Running Person 3 analysis (Temperature Sensors)...")
            progress_bar.progress(55)
            
            with st.expander("👤 Person 3: Internal Temperature Sensors (T1-T9)", expanded=False):
                person3 = TempSensorAnalyzer(etl.processed_data)
                etl.processed_data = person3.analyze_and_clean()
                etl.merge_log(person3.get_log())
            
            # Person 4: Humidity Sensors
            status_text.text("Running Person 4 analysis (Humidity Sensors)...")
            progress_bar.progress(65)
            
            with st.expander("👤 Person 4: Internal Humidity Sensors (RH_1-RH_9)", expanded=False):
                person4 = HumiditySensorAnalyzer(etl.processed_data)
                etl.processed_data = person4.analyze_and_clean()
                etl.merge_log(person4.get_log())
            
            # Person 5: Weather Part 1
            status_text.text("Running Person 5 analysis (Weather Part 1)...")
            progress_bar.progress(75)
            
            with st.expander("👤 Person 5: External Weather Part 1 (T_out, RH_out, Tdewpoint)", expanded=False):
                person5 = WeatherPart1Analyzer(etl.processed_data)
                etl.processed_data = person5.analyze_and_clean()
                etl.merge_log(person5.get_log())
            
            # Person 6: Weather Part 2
            status_text.text("Running Person 6 analysis (Weather Part 2)...")
            progress_bar.progress(85)
            
            with st.expander("👤 Person 6: External Weather Part 2 (Pressure, Wind speed, Visibility)", expanded=False):
                person6 = WeatherPart2Analyzer(etl.processed_data)
                etl.processed_data = person6.analyze_and_clean()
                etl.merge_log(person6.get_log())
        
        # Integrate external data
        status_text.text("Integrating external data...")
        progress_bar.progress(90)
        etl.integrate_external_data()
        
        # Final quality check
        status_text.text("Performing final quality checks...")
        progress_bar.progress(95)
        
        st.markdown('<h3 class="section-header">🔍 Final Quality Assessment</h3>', unsafe_allow_html=True)
        
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Records", f"{len(etl.processed_data):,}", 
                     delta=len(etl.processed_data) - len(etl.original_data))
        with col2:
            st.metric("Final Columns", len(etl.processed_data.columns),
                     delta=len(etl.processed_data.columns) - len(etl.original_data.columns))
        with col3:
            null_count = etl.processed_data.isnull().sum().sum()
            st.metric("Null Values", f"{null_count:,}")
        with col4:
            duplicate_count = etl.processed_data.duplicated().sum()
            st.metric("Duplicates", f"{duplicate_count:,}")
        
        # Data quality summary
        if null_count == 0:
            st.success("✅ No null values in final dataset")
        else:
            st.warning(f"⚠️ {null_count:,} null values remaining")
        
        if duplicate_count == 0:
            st.success("✅ No duplicate rows found")
        else:
            st.warning(f"⚠️ {duplicate_count:,} duplicate rows found")
        
        # Save final dataset
        status_text.text("Saving final dataset...")
        progress_bar.progress(98)
        
        if etl.save_final_dataset(output_filename):
            st.success(f"✅ Final dataset saved as: {output_filename}")
            
            # Download button
            with open(output_filename, 'rb') as f:
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=f,
                    file_name=output_filename,
                    mime="text/csv"
                )
        
        # Show processing log
        if show_logs:
            st.markdown('<h3 class="section-header">📋 Processing Log</h3>', unsafe_allow_html=True)
            
            with st.expander("View Detailed Processing Log", expanded=False):
                log_df = pd.DataFrame(etl.cleaning_log)
                if not log_df.empty:
                    # Color code by level
                    for _, entry in log_df.iterrows():
                        level = entry.get('level', 'INFO')
                        message = entry.get('message', '')
                        timestamp = entry.get('timestamp', '')
                        
                        if level == 'ERROR':
                            st.error(f"[{timestamp}] {message}")
                        elif level == 'WARNING':
                            st.warning(f"[{timestamp}] {message}")
                        else:
                            st.info(f"[{timestamp}] {message}")
                else:
                    st.info("No log entries found")
        
        # Final summary
        progress_bar.progress(100)
        status_text.text("ETL Pipeline completed successfully!")
        
        st.balloons()
        st.success("🎉 ETL Pipeline completed successfully!")
        
        # Show final dataset preview
        with st.expander("📊 Final Dataset Preview", expanded=True):
            st.write("**Final Dataset Statistics:**")
            st.dataframe(etl.processed_data.describe(), use_container_width=True)
            
            st.write("**Final Dataset Preview:**")
            st.dataframe(etl.processed_data.head(10), use_container_width=True)
            
            st.write("**New Columns Created:**")
            new_columns = set(etl.processed_data.columns) - set(etl.original_data.columns)
            if new_columns:
                st.write(f"Added {len(new_columns)} new columns:")
                for col in sorted(new_columns):
                    st.write(f"- {col}")
            else:
                st.write("No new columns were created")
    
    except Exception as e:
        st.error(f"❌ Error during ETL pipeline execution: {str(e)}")
        st.error("Please check your data and try again")
        if show_logs:
            st.exception(e)

def show_overview():
    """Show ETL overview information."""
    
    st.markdown("## 📋 ETL Process Overview")
    
    st.markdown("""
    This ETL (Extract, Transform, Load) pipeline is designed for **Assignment 2: Data Cleaning and Preparation**.
    
    ### 🎯 Objectives
    - Create a clean and model-ready dataset
    - Perform comprehensive data quality analysis
    - Handle missing values, outliers, and inconsistencies
    - Generate meaningful features from existing data
    - Provide detailed documentation of all cleaning decisions
    
    ### 👥 Team Structure
    The project is divided into **6 individual responsibilities**, each focusing on specific data columns:
    """)
    
    # Create columns for team structure
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Person 1: Temporal Data**
        - Appliances energy consumption
        - Date/time information
        
        **Person 2: Lighting System**
        - Lights energy consumption
        
        **Person 3: Internal Temperature**
        - Temperature sensors T1 through T9
        """)
    
    with col2:
        st.markdown("""
        **Person 4: Internal Humidity**
        - Humidity sensors RH_1 through RH_9
        
        **Person 5: External Weather (Part 1)**
        - Outdoor temperature, humidity, dew point
        
        **Person 6: External Weather (Part 2)**
        - Atmospheric pressure, wind speed, visibility
        """)
    
    st.markdown("""
    ### 🔄 ETL Process Steps
    
    1. **Extract**: Load and validate the original dataset
    2. **Transform**: 
       - Individual analysis and cleaning by each team member
       - Data quality assessment and validation
       - Missing value imputation and outlier handling
       - Feature engineering and data enrichment
    3. **Load**: Save the final cleaned dataset with comprehensive documentation
    
    ### 📊 Expected Outputs
    - Clean, model-ready dataset
    - Comprehensive data quality report
    - Feature documentation
    - Processing log with all decisions made
    """)

def show_data_info(dataset_path):
    """Show dataset information."""
    
    st.markdown("## 📊 Dataset Information")
    
    if dataset_path and os.path.exists(dataset_path):
        try:
            # Load just the first few rows for quick info
            df_sample = pd.read_csv(dataset_path, nrows=1000)
            
            st.success(f"✅ Dataset found: {dataset_path}")
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sample Rows", len(df_sample))
            with col2:
                st.metric("Total Columns", len(df_sample.columns))
            with col3:
                file_size = os.path.getsize(dataset_path) / (1024 * 1024)  # MB
                st.metric("File Size", f"{file_size:.2f} MB")
            
            # Column information
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column Name': df_sample.columns,
                'Data Type': df_sample.dtypes,
                'Sample Value': [df_sample[col].iloc[0] if not pd.isna(df_sample[col].iloc[0]) else 'NaN' 
                               for col in df_sample.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Expected columns by person
            st.write("**Expected Column Assignment:**")
            
            expected_assignments = {
                'Person 1 (Temporal)': ['date', 'Appliances'],
                'Person 2 (Lighting)': ['lights'],
                'Person 3 (Temperature)': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9'],
                'Person 4 (Humidity)': ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9'],
                'Person 5 (Weather P1)': ['T_out', 'RH_out', 'Tdewpoint'],
                'Person 6 (Weather P2)': ['Pressure', 'Press_mm_hg', 'Windspeed', 'Wind speed', 'Visibility']
            }
            
            available_columns = set(df_sample.columns)
            
            for person, expected_cols in expected_assignments.items():
                st.write(f"**{person}:**")
                found_cols = []
                missing_cols = []
                
                for col in expected_cols:
                    if col in available_columns:
                        found_cols.append(col)
                    else:
                        # Check for similar column names
                        similar_cols = [c for c in available_columns if col.lower() in c.lower() or c.lower() in col.lower()]
                        if similar_cols:
                            found_cols.extend(similar_cols)
                        else:
                            missing_cols.append(col)
                
                if found_cols:
                    st.success(f"✅ Found: {', '.join(found_cols)}")
                if missing_cols:
                    st.warning(f"⚠️ Missing: {', '.join(missing_cols)}")
            
        except Exception as e:
            st.error(f"❌ Error reading dataset: {str(e)}")
    else:
        st.warning("⚠️ Dataset not found. Please upload a file or ensure 'dataset.csv' exists in the current directory.")

def show_data_preview(dataset_path):
    """Show dataset preview."""
    
    st.markdown("## 🔍 Dataset Preview")
    
    if dataset_path and os.path.exists(dataset_path):
        try:
            # Load sample data
            sample_size = st.slider("Sample Size", min_value=5, max_value=1000, value=100)
            df_sample = pd.read_csv(dataset_path, nrows=sample_size)
            
            # Show data
            st.write(f"**First {len(df_sample)} rows:**")
            st.dataframe(df_sample, use_container_width=True)
            
            # Basic statistics
            st.write("**Basic Statistics:**")
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df_sample[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
            
            # Missing values
            st.write("**Missing Values:**")
            missing_data = df_sample.isnull().sum()
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': (missing_data.values / len(df_sample) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values in the sample")
                
        except Exception as e:
            st.error(f"❌ Error loading dataset preview: {str(e)}")
    else:
        st.warning("⚠️ No dataset available for preview")

def show_documentation():
    """Show documentation and help."""
    
    st.markdown("## 📚 Documentation")
    
    st.markdown("""
    ### 🚀 How to Use This ETL Pipeline
    
    1. **Select Data Source**: Choose to use the default `dataset.csv` or upload your own file
    2. **Configure Options**: 
       - Enable/disable individual analyses
       - Choose to save intermediate results
       - Control log verbosity
    3. **Set Output Filename**: Specify the name for your cleaned dataset
    4. **Run Pipeline**: Click "Run ETL Pipeline" to start the process
    
    ### 📋 What Each Person's Analysis Does
    
    #### Person 1: Temporal Data Analysis
    - **Columns**: `Appliances`, `date`
    - **Tasks**:
      - Convert date strings to proper datetime format
      - Validate temporal consistency and detect gaps
      - Handle negative energy values
      - Create temporal features (hour, day of week, season, etc.)
      - Detect and handle outliers in energy consumption
    
    #### Person 2: Lighting System Analysis
    - **Columns**: `lights`
    - **Tasks**:
      - Validate lighting energy consumption values
      - Handle missing values using forward fill and statistical imputation
      - Create lighting categories (Off, Low, Medium, High)
      - Calculate lighting efficiency ratios
      - Generate moving averages for trend analysis
    
    #### Person 3: Internal Temperature Sensors
    - **Columns**: `T1`, `T2`, `T3`, `T4`, `T5`, `T6`, `T7`, `T8`, `T9`
    - **Tasks**:
      - Validate temperature ranges (reasonable indoor temperatures)
      - Detect sensor correlations and potential redundancies
      - Use KNN imputation for missing values based on other sensors
      - Create aggregate features (average, range, standard deviation)
      - Calculate comfort indices and thermal zones
    
    #### Person 4: Internal Humidity Sensors
    - **Columns**: `RH_1` through `RH_9`
    - **Tasks**:
      - Validate humidity ranges (0-100%)
      - Handle impossible values and sensor malfunctions
      - Create humidity comfort categories
      - Calculate mold risk indicators
      - Generate humidity uniformity metrics
    
    #### Person 5: External Weather (Part 1)
    - **Columns**: `T_out`, `RH_out`, `Tdewpoint`
    - **Tasks**:
      - Validate meteorological relationships (dewpoint ≤ temperature)
      - Calculate heat index and apparent temperature
      - Create weather comfort categories
      - Handle seasonal patterns in missing data
      - Generate weather-based features for energy modeling
    
    #### Person 6: External Weather (Part 2)
    - **Columns**: `Pressure`, `Wind speed`, `Visibility`
    - **Tasks**:
      - Validate atmospheric pressure ranges and units
      - Handle wind speed (must be non-negative)
      - Create weather system indicators (storms, high pressure systems)
      - Generate wind categories based on Beaufort scale
      - Calculate weather stability indices
    
    ### 🔧 Data Cleaning Strategies
    
    #### Missing Value Handling
    1. **Short gaps** (≤3 values): Forward fill
    2. **Medium gaps**: KNN imputation using correlated variables
    3. **Long gaps**: Seasonal median or statistical imputation
    
    #### Outlier Treatment
    1. **Invalid values**: Set to NaN (e.g., negative energy, humidity > 100%)
    2. **Extreme outliers**: Cap at reasonable bounds or remove
    3. **Physical impossibilities**: Correct or remove (e.g., dewpoint > temperature)
    
    #### Feature Engineering
    - **Temporal features**: Hour, day of week, season, holidays
    - **Aggregated features**: Averages, ranges, standard deviations
    - **Categorical features**: Comfort levels, weather categories
    - **Derived features**: Efficiency ratios, comfort indices, trend indicators
    
    ### 📊 Expected Output Quality
    
    The final dataset should have:
    - ✅ No missing values (or properly documented decisions)
    - ✅ No invalid values (within physical/logical bounds)
    - ✅ Consistent data types and formats
    - ✅ Comprehensive feature set for modeling
    - ✅ Detailed processing log with all decisions documented
    
    ### 🎯 Best Practices Implemented
    
    1. **Preserve Original Data**: Never modify source data directly
    2. **Document Decisions**: Log all cleaning and transformation steps  
    3. **Validate Relationships**: Check physical and logical constraints
    4. **Conservative Approach**: Prefer data retention over removal when uncertain
    5. **Domain Knowledge**: Apply understanding of energy and weather systems
    6. **Reproducibility**: Ensure all steps can be repeated and verified
    
    ### 🔍 Quality Assurance
    
    - Cross-validation between related sensors
    - Physical constraint checking
    - Statistical outlier detection with domain context
    - Temporal consistency validation
    - Comprehensive logging and audit trail
    """)

if __name__ == "__main__":
    main()