# ETL Pipeline - Data Cleaning and Preparation
## Assignment 2: Complete ETL Process

This repository contains a comprehensive ETL (Extract, Transform, Load) pipeline designed for **Assignment 2: Data Cleaning and Preparation**. The project implements a collaborative data cleaning process with individual responsibilities for different data aspects.

## 🎯 Project Overview

The goal is to create a clean and model-ready dataset from energy consumption and meteorological data. The project is structured to simulate a team environment where each person is responsible for specific data columns.

### Team Structure

| Person | Responsibility | Columns |
|--------|----------------|---------|
| **Person 1** | Temporal Data | `Appliances`, `date` |
| **Person 2** | Lighting System | `lights` |
| **Person 3** | Internal Temperature Sensors | `T1`, `T2`, `T3`, `T4`, `T5`, `T6`, `T7`, `T8`, `T9` |
| **Person 4** | Internal Humidity Sensors | `RH_1`, `RH_2`, `RH_3`, `RH_4`, `RH_5`, `RH_6`, `RH_7`, `RH_8`, `RH_9` |
| **Person 5** | External Weather (Part 1) | `T_out`, `RH_out`, `Tdewpoint` |
| **Person 6** | External Weather (Part 2) | `Pressure`, `Wind speed`, `Visibility` |

## 📁 File Structure

```
├── dataset.csv                           # Original dataset (required)
├── etl_main.py                          # Main ETL pipeline orchestrator
├── etl_person1_temporal.py              # Person 1: Temporal data analysis
├── etl_person2_lighting.py              # Person 2: Lighting system analysis
├── etl_person3_temp_sensors.py          # Person 3: Temperature sensor analysis
├── etl_person4_humidity_sensors.py      # Person 4: Humidity sensor analysis
├── etl_person5_weather_part1.py         # Person 5: Weather analysis part 1
├── etl_person6_weather_part2.py         # Person 6: Weather analysis part 2
├── streamlit_etl_app.py                 # Streamlit web interface
├── run_etl.py                           # Simple command-line runner
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8 or higher**
2. **Required Python packages** (install via requirements.txt)

### Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure your dataset** is named `dataset.csv` and placed in the project directory

### Required Dataset Format

Your `dataset.csv` should contain the following columns:
- `date` - Date and time information
- `Appliances` - Appliance energy consumption
- `lights` - Lighting energy consumption  
- `T1` through `T9` - Temperature sensor readings
- `RH_1` through `RH_9` - Humidity sensor readings
- `T_out`, `RH_out`, `Tdewpoint` - External weather data part 1
- `Pressure`, `Windspeed` (or `Wind speed`), `Visibility` - External weather data part 2

## 🖥️ Running the ETL Pipeline

You have **three options** to run the ETL pipeline:

### Option 1: Streamlit Web Interface (Recommended)

Launch the interactive web interface:

```bash
streamlit run streamlit_etl_app.py
```

**Features:**
- Interactive web interface
- Real-time progress tracking
- Data visualization
- Individual analysis sections
- Download cleaned dataset
- Comprehensive reporting

**Benefits:**
- User-friendly interface
- Visual data exploration
- Step-by-step progress monitoring
- Easy configuration options

### Option 2: Command Line (Simple)

Run the ETL pipeline directly:

```bash
python run_etl.py
```

**Features:**
- Command-line output
- Progress tracking
- Automatic report generation
- Lightweight execution

**Benefits:**
- Fast execution
- No web browser required
- Suitable for automation
- Clear console output

### Option 3: Python Script (Advanced)

Import and use in your own Python code:

```python
from etl_main import ETLPipeline

# Initialize pipeline
etl = ETLPipeline('dataset.csv')

# Run complete ETL process
etl.load_data()
etl.run_individual_analyses()
etl.integrate_external_data()
etl.save_final_dataset('cleaned_data.csv')

# Generate report
etl.generate_report()
```

## 📊 What Each Analysis Does

### Person 1: Temporal Data Analysis
- **Date Processing**: Convert date strings to proper datetime format
- **Temporal Validation**: Check for gaps and inconsistencies in timeline
- **Energy Validation**: Handle negative values and outliers in appliance consumption
- **Feature Engineering**: Create hour, day of week, season, weekend indicators
- **Trend Analysis**: Generate moving averages and change rates

### Person 2: Lighting System Analysis  
- **Value Validation**: Check for negative or impossible lighting values
- **Missing Value Imputation**: Use forward fill and statistical methods
- **Categorization**: Create lighting intensity categories (Off, Low, Medium, High)
- **Efficiency Metrics**: Calculate lighting-to-appliance ratios
- **Pattern Analysis**: Identify usage patterns and anomalies

### Person 3: Internal Temperature Sensors
- **Range Validation**: Ensure temperatures are within reasonable indoor ranges (-10°C to 50°C)
- **Sensor Correlation**: Analyze relationships between different sensors
- **Advanced Imputation**: Use KNN imputation based on correlated sensors
- **Aggregation**: Create average internal temperature, range, and uniformity metrics
- **Comfort Analysis**: Generate thermal comfort indices and zone classifications

### Person 4: Internal Humidity Sensors
- **Humidity Validation**: Ensure values are within 0-100% range
- **Correlation Analysis**: Identify redundant or faulty sensors
- **Comfort Assessment**: Categorize humidity levels (Too Dry, Comfortable, Too Humid)
- **Health Indicators**: Calculate mold risk scores for high humidity areas
- **Uniformity Metrics**: Assess humidity distribution consistency

### Person 5: External Weather (Part 1)
- **Meteorological Validation**: Ensure dewpoint ≤ temperature relationship
- **Physical Constraints**: Validate temperature and humidity ranges
- **Derived Metrics**: Calculate heat index, apparent temperature, comfort indices
- **Seasonal Imputation**: Use seasonal patterns for missing value filling
- **Weather Categorization**: Create temperature and humidity comfort categories

### Person 6: External Weather (Part 2)  
- **Pressure Validation**: Handle different units (hPa, mmHg) and validate ranges
- **Wind Speed Processing**: Ensure non-negative values and handle extreme speeds
- **Visibility Processing**: Validate visibility ranges and detect fog conditions
- **Weather Systems**: Identify storm conditions, high-pressure systems
- **Stability Indices**: Calculate weather stability and variability metrics

## 🔧 Data Cleaning Strategies

### Missing Value Handling
1. **Short gaps (≤3 values)**: Forward fill for temporal continuity
2. **Medium gaps**: KNN imputation using correlated variables
3. **Long gaps**: Seasonal median or domain-specific imputation
4. **Systematic missing**: Domain knowledge-based strategies

### Outlier Treatment
1. **Invalid values**: Remove physically impossible values (negative energy, humidity > 100%)
2. **Extreme outliers**: Cap at reasonable bounds or investigate further
3. **Physical constraints**: Enforce meteorological and physical relationships
4. **Conservative approach**: Prefer data retention over removal when uncertain

### Feature Engineering
- **Temporal features**: Hour, day of week, season, holiday indicators
- **Aggregated features**: Means, ranges, standard deviations across sensors
- **Categorical features**: Comfort levels, weather categories, system states
- **Derived metrics**: Efficiency ratios, comfort indices, stability measures
- **Interaction features**: Cross-variable relationships and combinations

## 📈 Expected Outputs

### Primary Outputs
1. **`dataset_final_cleaned.csv`** - Clean, model-ready dataset
2. **`etl_processing_log.txt`** - Comprehensive processing log
3. **Quality assessment report** - Data quality metrics and validation results

### Dataset Improvements
- ✅ **No missing values** (or properly documented decisions)
- ✅ **No invalid values** (within physical/logical bounds)  
- ✅ **Consistent data types** and proper formatting
- ✅ **Enhanced feature set** for modeling and analysis
- ✅ **Temporal consistency** and proper date handling
- ✅ **Physical constraint compliance** for all measurements

### New Features Created
- **Temporal**: Hour, day of week, season, weekend indicators
- **Aggregated**: Average temperatures/humidity, ranges, uniformity measures
- **Comfort**: Thermal comfort, humidity comfort, weather comfort indices
- **Efficiency**: Lighting efficiency, energy utilization ratios
- **Weather**: Heat index, apparent temperature, weather system indicators
- **Trends**: Moving averages, change rates, stability measures

## 🔍 Quality Assurance

### Validation Checks
- **Cross-sensor validation**: Consistency between related measurements
- **Physical constraint checking**: Meteorological and energy system rules
- **Statistical outlier detection**: Context-aware anomaly identification
- **Temporal consistency**: Timeline gaps and sequence validation
- **Data type validation**: Proper formatting and type consistency

### Documentation Standards
- **Decision logging**: All cleaning decisions documented with reasoning
- **Traceability**: Full audit trail from original to final data
- **Methodology documentation**: Detailed explanation of all techniques used
- **Quality metrics**: Comprehensive data quality assessment results

## 🛠️ Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Ensure all `.py` files are in the same directory
   - Check that `requirements.txt` packages are installed
   - Verify Python version compatibility (3.8+)

2. **"Dataset not found" error**
   - Ensure `dataset.csv` exists in the project directory
   - Check file name spelling and case sensitivity
   - Verify file is not corrupted and is valid CSV format

3. **Memory issues with large datasets**
   - Use sampling for initial exploration
   - Process data in chunks if necessary
   - Close unused applications to free memory

4. **Streamlit connection issues**
   - Check if port 8501 is available
   - Try different port: `streamlit run streamlit_etl_app.py --server.port 8502`
   - Verify firewall settings if accessing remotely

### Performance Optimization

- **Large datasets**: Consider sampling for initial analysis
- **Memory usage**: Process data in chunks if memory is limited
- **Visualization**: Sample data for plotting if dataset is very large
- **Processing speed**: Run individual analyses separately for debugging

## 📚 Dependencies

### Core Libraries
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Basic plotting and visualization
- `seaborn` - Statistical data visualization
- `streamlit` - Web application framework

### Advanced Libraries
- `scikit-learn` - KNN imputation and advanced analytics
- `scipy` - Statistical functions and tests
- `plotly` - Interactive visualizations (optional)

### Installation
```bash
pip install pandas numpy matplotlib seaborn streamlit scikit-learn scipy plotly
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

## 🎓 Educational Value

This project demonstrates:
- **Collaborative data science** workflows
- **Comprehensive data quality assessment** techniques
- **Domain-specific data cleaning** strategies
- **Feature engineering** for energy and weather data
- **Documentation and reproducibility** best practices
- **Real-world data challenges** and solutions

## 📋 Assignment Deliverables

1. **Clean Dataset** (`dataset_final_cleaned.csv`)
2. **Processing Documentation** (processing log and reports)
3. **Individual Analysis Reports** (embedded in each module)
4. **Quality Assessment** (validation results and metrics)
5. **Feature Documentation** (new variables created and their meanings)
6. **Methodology Documentation** (this README and code comments)

## 🤝 Contributing

Each team member should:
1. **Focus on their assigned columns** but understand the overall process
2. **Document all decisions** made during cleaning and transformation
3. **Validate their work** against domain knowledge and data relationships
4. **Communicate findings** that might affect other team members' work
5. **Review and test** the integrated pipeline

## 📞 Support

If you encounter issues:
1. **Check the troubleshooting section** above
2. **Review the processing logs** for error details  
3. **Examine individual module outputs** to isolate problems
4. **Verify data format and requirements** match expectations
5. **Test with smaller data samples** to identify bottlenecks

---

**Good luck with your ETL pipeline and data cleaning assignment!** 🚀

Remember: The goal is not just to clean the data, but to **understand it**, **document your decisions**, and **create a robust, reproducible process** that others can follow and verify.