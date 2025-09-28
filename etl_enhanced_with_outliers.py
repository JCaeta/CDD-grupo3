"""
ETL Pipeline with Integrated Outlier Treatment
=============================================

Enhanced version of the main ETL pipeline that includes comprehensive
outlier detection and treatment capabilities.

Author: ETL Team  
Date: 28 de septiembre de 2025
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

# Import outlier treatment modules
from outlier_treatment import OutlierDetector, OutlierTreatmentPipeline

class EnhancedETLPipeline:
    """
    Enhanced ETL Pipeline class with integrated outlier treatment.
    """
    
    def __init__(self, dataset_path='dataset.csv'):
        """
        Initialize the enhanced ETL pipeline.
        
        Args:
            dataset_path (str): Path to the input dataset
        """
        self.dataset_path = dataset_path
        self.original_data = None
        self.processed_data = None
        self.outlier_treated_data = None
        self.cleaning_log = []
        self.outlier_treatment_log = []
        
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
    
    def log_action(self, message, level="INFO"):
        """Log actions for tracking."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.cleaning_log.append(log_entry)
        print(log_entry)
    
    def outlier_analysis_phase(self):
        """
        Comprehensive outlier analysis phase before main ETL processing.
        """
        st.header("🔍 Fase de Análisis de Outliers")
        
        self.log_action("Starting comprehensive outlier analysis")
        
        # Create outlier detector
        detector = OutlierDetector(self.processed_data)
        
        # Define column groups for analysis
        column_groups = {
            'Consumo Energético': ['Appliances', 'lights'],
            'Temperatura Interna': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9'],
            'Humedad Interna': ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9'],
            'Meteorología Externa': ['T_out', 'RH_out', 'Tdewpoint', 'Pressure', 'Wind speed', 'Windspeed', 'Visibility']
        }
        
        outlier_summary = []
        
        # Analyze each group
        for group_name, columns in column_groups.items():
            st.subheader(f"Análisis de Outliers: {group_name}")
            
            # Filter existing columns
            valid_columns = [col for col in columns if col in self.processed_data.columns]
            
            if not valid_columns:
                continue
            
            group_results = {}
            
            for column in valid_columns:
                if self.processed_data[column].dtype in ['int64', 'float64']:
                    # Analyze outliers
                    results = detector.analyze_column_outliers(
                        column, 
                        methods=['iqr', 'zscore'],
                        iqr_multiplier=1.5,
                        zscore_threshold=3
                    )
                    
                    group_results[column] = results
                    
                    # Display results
                    for method, info in results.items():
                        if info:
                            outlier_count = info['outlier_count']
                            outlier_pct = info['outlier_percentage']
                            
                            outlier_summary.append({
                                'Grupo': group_name,
                                'Columna': column,
                                'Método': method,
                                'Outliers': outlier_count,
                                'Porcentaje': f"{outlier_pct:.2f}%"
                            })
                            
                            if outlier_count > 0:
                                st.warning(f"**{column}** ({method}): {outlier_count} outliers ({outlier_pct:.2f}%)")
                            else:
                                st.success(f"**{column}** ({method}): Sin outliers detectados")
            
            # Visualize outliers for this group
            if valid_columns:
                self._create_group_outlier_visualizations(group_name, valid_columns[:4])  # Limit for performance
        
        # Display summary table
        if outlier_summary:
            st.subheader("📊 Resumen de Outliers Detectados")
            summary_df = pd.DataFrame(outlier_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Key statistics
            total_outliers = summary_df[summary_df['Método'] == 'iqr']['Outliers'].sum()
            affected_columns = len(summary_df[summary_df['Outliers'] > 0]['Columna'].unique())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Outliers (IQR)", total_outliers)
            with col2:
                st.metric("Columnas Afectadas", affected_columns)
            with col3:
                avg_outliers = summary_df[summary_df['Método'] == 'iqr']['Outliers'].mean()
                st.metric("Promedio por Columna", f"{avg_outliers:.1f}")
        
        return detector
    
    def _create_group_outlier_visualizations(self, group_name, columns):
        """Create outlier visualizations for a group of columns."""
        if len(columns) > 0:
            fig, axes = plt.subplots(2, len(columns), figsize=(5*len(columns), 8))
            if len(columns) == 1:
                axes = axes.reshape(2, 1)
            
            for i, column in enumerate(columns):
                # Boxplot
                self.processed_data.boxplot(column=column, ax=axes[0, i])
                axes[0, i].set_title(f'Boxplot - {column}')
                axes[0, i].grid(True)
                
                # Histogram
                axes[1, i].hist(self.processed_data[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[1, i].set_title(f'Distribución - {column}')
                axes[1, i].grid(True)
            
            plt.suptitle(f'Análisis de Outliers - {group_name}', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
    def outlier_treatment_phase(self, treatment_strategy='conservative'):
        """
        Apply outlier treatment based on the selected strategy.
        
        Args:
            treatment_strategy (str): 'conservative', 'moderate', 'aggressive'
        """
        st.header("🔧 Fase de Tratamiento de Outliers")
        
        self.log_action(f"Starting outlier treatment with {treatment_strategy} strategy")
        
        # Treatment parameters based on strategy
        strategies = {
            'conservative': {
                'method': 'iqr',
                'treatment': 'cap',
                'iqr_multiplier': 1.5,
                'description': 'Limitar valores extremos a los límites IQR (conservador)'
            },
            'moderate': {
                'method': 'iqr', 
                'treatment': 'median',
                'iqr_multiplier': 1.2,
                'description': 'Reemplazar outliers con mediana (moderado)'
            },
            'aggressive': {
                'method': 'zscore',
                'treatment': 'remove',
                'zscore_threshold': 2.5,
                'description': 'Eliminar outliers usando Z-score estricto (agresivo)'
            }
        }
        
        strategy_config = strategies[treatment_strategy]
        
        st.info(f"**Estrategia seleccionada**: {treatment_strategy.title()}")
        st.write(f"**Descripción**: {strategy_config['description']}")
        
        # Create treatment pipeline
        pipeline = OutlierTreatmentPipeline(self.processed_data)
        
        # Apply treatment to numeric columns (exclude date)
        exclude_cols = ['date', 'hour', 'day_of_week', 'month', 'is_weekend']
        
        if treatment_strategy == 'conservative':
            pipeline.auto_treat_numeric_columns(
                method='iqr',
                treatment='cap',
                exclude_columns=exclude_cols
            )
        elif treatment_strategy == 'moderate':
            pipeline.auto_treat_numeric_columns(
                method='iqr',
                treatment='median',
                exclude_columns=exclude_cols
            )
        elif treatment_strategy == 'aggressive':
            # More selective aggressive treatment
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            cols_to_treat = [col for col in numeric_cols if col not in exclude_cols]
            
            for col in cols_to_treat:
                detector = OutlierDetector(pipeline.data)
                # Only remove outliers if they represent < 5% of data
                results = detector.analyze_column_outliers(col, methods=['zscore'])
                if results['zscore']['outlier_percentage'] < 5.0:
                    pipeline.data = detector.treat_outliers(col, method='remove', detection_method='zscore')
                else:
                    pipeline.data = detector.treat_outliers(col, method='cap', detection_method='iqr')
        
        # Store treated data
        self.outlier_treated_data = pipeline.data
        self.processed_data = pipeline.data  # Use treated data for subsequent processing
        
        # Display treatment summary
        treatment_summary = pipeline.get_summary_report()
        
        if isinstance(treatment_summary, pd.DataFrame):
            st.subheader("📈 Resumen del Tratamiento")
            st.dataframe(treatment_summary, use_container_width=True)
        
        # Compare before/after statistics
        self._display_before_after_comparison()
        
        self.log_action(f"Outlier treatment completed using {treatment_strategy} strategy")
        
        return pipeline
    
    def _display_before_after_comparison(self):
        """Display before/after comparison of key statistics."""
        st.subheader("📊 Comparación Antes/Después del Tratamiento")
        
        # Select a few key columns for comparison
        key_columns = ['Appliances', 'lights', 'T_out', 'RH_out']
        key_columns = [col for col in key_columns if col in self.processed_data.columns]
        
        comparison_data = []
        
        for col in key_columns:
            if col in self.original_data.columns:
                original_stats = self.original_data[col].describe()
                treated_stats = self.processed_data[col].describe()
                
                comparison_data.append({
                    'Columna': col,
                    'Métrica': 'Media',
                    'Original': f"{original_stats['mean']:.2f}",
                    'Tratado': f"{treated_stats['mean']:.2f}",
                    'Cambio': f"{((treated_stats['mean'] - original_stats['mean']) / original_stats['mean'] * 100):.1f}%"
                })
                
                comparison_data.append({
                    'Columna': col,
                    'Métrica': 'Std Dev',
                    'Original': f"{original_stats['std']:.2f}",
                    'Tratado': f"{treated_stats['std']:.2f}",
                    'Cambio': f"{((treated_stats['std'] - original_stats['std']) / original_stats['std'] * 100):.1f}%"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    def run_individual_analyses(self):
        """
        Run individual analyses for each person's assigned columns.
        """
        st.header("👥 Análisis Individual por Columnas")
        
        # Person 1: Temporal Data (Appliances, date)
        st.subheader("Person 1: Análisis de Datos Temporales")
        person1 = TemporalDataAnalyzer(self.processed_data)
        self.processed_data = person1.analyze_and_clean()
        self.merge_log(person1.get_log())
        
        # Person 2: Lighting System (lights)
        st.subheader("Person 2: Análisis del Sistema de Iluminación")
        person2 = LightingAnalyzer(self.processed_data)
        self.processed_data = person2.analyze_and_clean()
        self.merge_log(person2.get_log())
        
        # Person 3: Internal Temperature Sensors (T1-T9)
        st.subheader("Person 3: Análisis de Sensores de Temperatura Interna")
        person3 = TempSensorAnalyzer(self.processed_data)
        self.processed_data = person3.analyze_and_clean()
        self.merge_log(person3.get_log())
        
        # Person 4: Internal Humidity Sensors (RH_1-RH_9)
        st.subheader("Person 4: Análisis de Sensores de Humedad Interna")
        person4 = HumiditySensorAnalyzer(self.processed_data)
        self.processed_data = person4.analyze_and_clean()
        self.merge_log(person4.get_log())
        
        # Person 5: External Weather Part 1 (T_out, RH_out, Tdewpoint)
        st.subheader("Person 5: Análisis de Datos Meteorológicos - Parte 1")
        person5 = WeatherPart1Analyzer(self.processed_data)
        self.processed_data = person5.analyze_and_clean()
        self.merge_log(person5.get_log())
        
        # Person 6: External Weather Part 2 (Pressure, Wind speed, Visibility)
        st.subheader("Person 6: Análisis de Datos Meteorológicos - Parte 2")
        person6 = WeatherPart2Analyzer(self.processed_data)
        self.processed_data = person6.analyze_and_clean()
        self.merge_log(person6.get_log())
    
    def integrate_external_data(self):
        """
        Integrate external data sources if available.
        """
        st.subheader("🔗 Integración de Datos Externos")
        
        self.log_action("External data integration - No additional sources specified")
        
        # Add temporal features from existing date column
        if 'date' in self.processed_data.columns:
            try:
                self.processed_data['date'] = pd.to_datetime(self.processed_data['date'], 
                                                           format='%d-%m-%Y %H:%M')
                
                # Extract temporal features
                self.processed_data['hour'] = self.processed_data['date'].dt.hour
                self.processed_data['day_of_week'] = self.processed_data['date'].dt.dayofweek
                self.processed_data['month'] = self.processed_data['date'].dt.month
                self.processed_data['is_weekend'] = (self.processed_data['day_of_week'] >= 5).astype(int)
                
                st.success("Características temporales agregadas: hour, day_of_week, month, is_weekend")
                self.log_action("Added temporal features: hour, day_of_week, month, is_weekend")
            except Exception as e:
                st.warning(f"Error procesando columna de fecha: {str(e)}")
                self.log_action(f"Error processing date column: {str(e)}", "WARNING")
    
    def final_quality_check(self):
        """
        Perform final data quality checks on the processed dataset.
        """
        st.subheader("✅ Evaluación Final de Calidad de Datos")
        
        # Check for remaining null values
        null_counts = self.processed_data.isnull().sum()
        if null_counts.sum() > 0:
            st.warning(f"Valores nulos restantes: {null_counts[null_counts > 0].to_dict()}")
            self.log_action(f"Remaining null values: {null_counts[null_counts > 0].to_dict()}", "WARNING")
        else:
            st.success("✅ Sin valores nulos en el dataset final")
            self.log_action("No null values in final dataset")
        
        # Check for duplicates
        duplicates = self.processed_data.duplicated().sum()
        if duplicates > 0:
            st.warning(f"Se encontraron {duplicates} filas duplicadas")
            self.log_action(f"Found {duplicates} duplicate rows", "WARNING")
        else:
            st.success("✅ Sin filas duplicadas")
            self.log_action("No duplicate rows found")
        
        # Final dataset info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filas Originales", len(self.original_data))
        with col2:
            st.metric("Filas Finales", len(self.processed_data))
        with col3:
            retention_rate = (len(self.processed_data) / len(self.original_data)) * 100
            st.metric("Retención de Datos", f"{retention_rate:.1f}%")
        
        # Data type validation
        st.write("**Tipos de datos finales:**")
        st.write(self.processed_data.dtypes)
        
        # Basic statistics
        st.write("**Estadísticas del dataset final:**")
        st.write(self.processed_data.describe())
        
        self.log_action(f"Final quality check completed. Dataset shape: {self.processed_data.shape}")
    
    def save_final_dataset(self, filename='dataset_final_cleaned_with_outlier_treatment.csv'):
        """
        Save the final processed dataset.
        
        Args:
            filename (str): Output filename
        """
        try:
            self.processed_data.to_csv(filename, index=False)
            st.success(f"✅ Dataset final guardado como: {filename}")
            self.log_action(f"Final dataset saved as: {filename}")
            
            # Also save outlier-only treated version if different
            if self.outlier_treated_data is not None:
                outlier_filename = 'dataset_outlier_treated_only.csv'
                self.outlier_treated_data.to_csv(outlier_filename, index=False)
                st.info(f"📊 Dataset con solo tratamiento de outliers guardado como: {outlier_filename}")
            
            return True
        except Exception as e:
            st.error(f"Error guardando dataset: {str(e)}")
            self.log_action(f"Error saving dataset: {str(e)}", "ERROR")
            return False
    
    def save_processing_log(self, filename='etl_processing_log_with_outliers.txt'):
        """Save the complete processing log."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("ETL PROCESSING LOG WITH OUTLIER TREATMENT\n")
                f.write("=" * 50 + "\n\n")
                
                for log_entry in self.cleaning_log:
                    f.write(log_entry + "\n")
                
                if self.outlier_treatment_log:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("OUTLIER TREATMENT LOG\n")
                    f.write("=" * 50 + "\n")
                    
                    for log_entry in self.outlier_treatment_log:
                        f.write(log_entry + "\n")
            
            st.success(f"✅ Log de procesamiento guardado como: {filename}")
            self.log_action(f"Processing log saved as: {filename}")
            return True
        except Exception as e:
            st.error(f"Error guardando log: {str(e)}")
            return False
    
    def merge_log(self, external_log):
        """Merge external log entries."""
        if isinstance(external_log, list):
            self.cleaning_log.extend(external_log)
        else:
            self.cleaning_log.append(str(external_log))
    
    def get_log(self):
        """Return the processing log."""
        return self.cleaning_log
    
    def run_complete_pipeline_with_outlier_treatment(self, treatment_strategy='conservative'):
        """
        Run the complete enhanced pipeline including outlier treatment.
        
        Args:
            treatment_strategy (str): Outlier treatment strategy
        """
        st.title("🔄 Pipeline ETL Completo con Tratamiento de Outliers")
        
        # Step 1: Load Data
        if not self.load_data():
            st.error("❌ Error cargando datos. Deteniendo pipeline.")
            return False
        
        # Step 2: Outlier Analysis
        detector = self.outlier_analysis_phase()
        
        # Step 3: Outlier Treatment
        treatment_pipeline = self.outlier_treatment_phase(treatment_strategy)
        
        # Step 4: Individual Analyses
        self.run_individual_analyses()
        
        # Step 5: External Data Integration
        self.integrate_external_data()
        
        # Step 6: Final Quality Check
        self.final_quality_check()
        
        # Step 7: Save Results
        self.save_final_dataset()
        self.save_processing_log()
        
        st.success("🎉 ¡Pipeline ETL con tratamiento de outliers completado exitosamente!")
        
        return True


# Enhanced Streamlit app integration would go here
def create_enhanced_streamlit_app():
    """Create enhanced Streamlit app with outlier treatment options."""
    
    st.set_page_config(
        page_title="ETL Pipeline con Tratamiento de Outliers",
        page_icon="🔄",
        layout="wide"
    )
    
    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuración del Pipeline")
    
    # Dataset upload
    uploaded_file = st.sidebar.file_uploader("Cargar Dataset", type=['csv'])
    
    if uploaded_file:
        # Save uploaded file temporarily
        with open('dataset.csv', 'wb') as f:
            f.write(uploaded_file.getbuffer())
    
    # Outlier treatment strategy selection
    strategy = st.sidebar.selectbox(
        "Estrategia de Tratamiento de Outliers",
        ['conservative', 'moderate', 'aggressive'],
        help="Conservative: Limitar valores, Moderate: Reemplazar con mediana, Aggressive: Eliminar outliers"
    )
    
    # Run pipeline button
    if st.sidebar.button("🚀 Ejecutar Pipeline Completo"):
        pipeline = EnhancedETLPipeline('dataset.csv')
        pipeline.run_complete_pipeline_with_outlier_treatment(strategy)


if __name__ == "__main__":
    create_enhanced_streamlit_app()