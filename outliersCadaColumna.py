"""
Análisis Comprehensivo de Outliers por Columna
==============================================

Este script realiza un análisis detallado de outliers para cada columna
del dataset, implementando múltiples métodos de detección y tratamiento.

Basado en el trabajo previo de análisis meteorológico y extendido para
todo el dataset.

Autor: Equipo ETL
Fecha: 28 de septiembre de 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew, kurtosis
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Importar nuestro módulo de tratamiento de outliers
from outlier_treatment import OutlierDetector, OutlierTreatmentPipeline

# Colores para output
BLUE = '\033[94m'
RESET = '\033[0m'
GRAY = '\033[90m'
MAGENTA = '\033[95m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'

class ComprehensiveOutlierAnalysis:
    """
    Análisis comprehensivo de outliers para todo el dataset.
    """
    
    def __init__(self, dataset_path='dataset.csv'):
        """
        Inicializar el análisis.
        
        Args:
            dataset_path (str): Ruta al dataset
        """
        self.dataset_path = dataset_path
        self.df = None
        self.numeric_columns = None
        self.outlier_results = {}
        self.load_data()
    
    def load_data(self):
        """Cargar los datos."""
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"{GREEN}Dataset cargado exitosamente: {self.df.shape}{RESET}")
            return True
        except Exception as e:
            print(f"{RED}Error cargando dataset: {str(e)}{RESET}")
            return False
    
    def print_dataset_info(self):
        """Mostrar información básica del dataset."""
        print('='*80)
        print(f'{MAGENTA}INFORMACIÓN DEL DATASET{RESET}')
        print('='*80)
        
        print(f'{BLUE}Forma del dataset:{RESET} {self.df.shape}')
        print(f'{BLUE}Columnas totales:{RESET} {len(self.df.columns)}')
        print(f'{BLUE}Columnas numéricas:{RESET} {len(self.numeric_columns)}')
        print()
        
        print(f'{BLUE}TODAS LAS COLUMNAS:{RESET}')
        for i, col in enumerate(self.df.columns, 1):
            col_type = "numérica" if col in self.numeric_columns else "categórica"
            print(f'{GRAY}  {i:2d}. {col} ({col_type}){RESET}')
        print()
    
    def analyze_column_group(self, group_name, columns, description=""):
        """
        Analizar un grupo específico de columnas.
        
        Args:
            group_name (str): Nombre del grupo
            columns (list): Lista de columnas
            description (str): Descripción del grupo
        """
        print('='*80)
        print(f'{MAGENTA}{group_name.upper()}{RESET}')
        if description:
            print(f'{GRAY}{description}{RESET}')
        print('='*80)
        
        # Filtrar solo columnas numéricas que existen
        valid_columns = [col for col in columns if col in self.df.columns and col in self.numeric_columns]
        
        if not valid_columns:
            print(f"{YELLOW}No se encontraron columnas numéricas válidas en este grupo{RESET}")
            return
        
        # Crear detector de outliers
        detector = OutlierDetector(self.df)
        group_results = {}
        
        # Análisis por columna
        for column in valid_columns:
            print(f'\n{BLUE}--- Análisis de {column} ---{RESET}')
            
            # Descripción estadística básica
            print(f'{GRAY}Estadísticas básicas:{RESET}')
            basic_stats = self.df[column].describe()
            print(basic_stats)
            
            # Detectar outliers con múltiples métodos
            results = detector.analyze_column_outliers(
                column, 
                methods=['iqr', 'zscore', 'modified_zscore'],
                iqr_multiplier=1.5,
                zscore_threshold=3,
                mod_zscore_threshold=3.5
            )
            
            group_results[column] = results
            
            # Mostrar resultados de detección
            print(f'\n{GREEN}Resultados de detección de outliers:{RESET}')
            for method, info in results.items():
                if info:
                    print(f'  {method:15s}: {info["outlier_count"]:4d} outliers ({info["outlier_percentage"]:5.2f}%)')
            
            # Calcular métricas adicionales
            skewness = skew(self.df[column].dropna())
            kurt = kurtosis(self.df[column].dropna())
            
            print(f'\n{BLUE}Métricas de distribución:{RESET}')
            print(f'  Asimetría (Skewness): {skewness:6.3f}')
            print(f'  Curtosis (Kurtosis):  {kurt:6.3f}')
            
            # Interpretación
            self._interpret_distribution(skewness, kurt)
        
        # Resumen del grupo
        self._print_group_summary(group_name, group_results)
        
        # Guardar resultados
        self.outlier_results[group_name] = group_results
    
    def _interpret_distribution(self, skewness, kurtosis_val):
        """Interpretar las métricas de distribución."""
        print(f'  {GRAY}Interpretación:{RESET}')
        
        # Interpretación de skewness
        if abs(skewness) < 0.5:
            skew_interp = "aproximadamente simétrica"
        elif skewness > 0.5:
            skew_interp = "sesgada hacia la derecha (cola derecha)"
        else:
            skew_interp = "sesgada hacia la izquierda (cola izquierda)"
        
        # Interpretación de kurtosis
        if abs(kurtosis_val) < 0.5:
            kurt_interp = "distribución normal (mesocúrtica)"
        elif kurtosis_val > 0.5:
            kurt_interp = "colas pesadas (leptocúrtica) - más valores extremos"
        else:
            kurt_interp = "colas ligeras (platicúrtica) - menos valores extremos"
        
        print(f'    - Distribución {skew_interp}')
        print(f'    - {kurt_interp.capitalize()}')
    
    def _print_group_summary(self, group_name, results):
        """Imprimir resumen del grupo."""
        print(f'\n{GREEN}=== RESUMEN - {group_name} ==={RESET}')
        
        summary_data = []
        for column, methods in results.items():
            for method, info in methods.items():
                if info:
                    summary_data.append([
                        column,
                        info['method'],
                        info['outlier_count'],
                        f"{info['outlier_percentage']:.2f}%"
                    ])
        
        if summary_data:
            headers = ['Columna', 'Método', 'Outliers', 'Porcentaje']
            print(tabulate(summary_data, headers=headers, tablefmt='grid'))
    
    def analyze_all_groups(self):
        """Analizar todos los grupos de columnas del dataset."""
        
        # Información general
        self.print_dataset_info()
        
        # Definir grupos de columnas según la estructura del proyecto ETL
        groups = {
            'Datos Temporales y Consumo': {
                'columns': ['Appliances', 'lights'],
                'description': 'Consumo energético de electrodomésticos e iluminación'
            },
            'Sensores de Temperatura Interna': {
                'columns': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9'],
                'description': 'Lecturas de 9 sensores de temperatura internos (°C)'
            },
            'Sensores de Humedad Interna': {
                'columns': ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9'],
                'description': 'Lecturas de 9 sensores de humedad internos (%)'
            },
            'Datos Meteorológicos Externos - Parte 1': {
                'columns': ['T_out', 'RH_out', 'Tdewpoint'],
                'description': 'Temperatura, humedad y punto de rocío externos'
            },
            'Datos Meteorológicos Externos - Parte 2': {
                'columns': ['Pressure', 'Wind speed', 'Windspeed', 'Visibility'],
                'description': 'Presión, velocidad del viento y visibilidad'
            }
        }
        
        # Analizar cada grupo
        for group_name, group_info in groups.items():
            self.analyze_column_group(
                group_name, 
                group_info['columns'], 
                group_info['description']
            )
    
    def create_visualization_dashboard(self, columns=None, figsize=(20, 15)):
        """
        Crear un dashboard de visualizaciones para outliers.
        
        Args:
            columns (list): Columnas a visualizar (None para todas las numéricas)
            figsize (tuple): Tamaño de la figura
        """
        if columns is None:
            columns = self.numeric_columns[:8]  # Limitar para visualización
        
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
        axes = axes.reshape(n_rows * 2, n_cols)
        
        for i, column in enumerate(columns):
            row = i // n_cols
            col = i % n_cols
            
            # Boxplot
            self.df.boxplot(column=column, ax=axes[row * 2, col])
            axes[row * 2, col].set_title(f'Boxplot - {column}')
            axes[row * 2, col].grid(True)
            
            # Histogram
            axes[row * 2 + 1, col].hist(self.df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[row * 2 + 1, col].set_title(f'Histogram - {column}')
            axes[row * 2 + 1, col].grid(True)
        
        # Ocultar axes vacíos
        for i in range(len(columns), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row * 2, col].set_visible(False)
            axes[row * 2 + 1, col].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Dashboard de Análisis de Outliers', fontsize=16, y=0.98)
        plt.show()
    
    def generate_comprehensive_report(self):
        """Generar reporte comprehensivo de outliers."""
        print('\n' + '='*80)
        print(f'{MAGENTA}REPORTE COMPREHENSIVO DE OUTLIERS{RESET}')
        print('='*80)
        
        all_results = []
        
        # Recopilar todos los resultados
        for group_name, group_results in self.outlier_results.items():
            for column, methods in group_results.items():
                for method, info in methods.items():
                    if info:
                        all_results.append({
                            'Grupo': group_name,
                            'Columna': column,
                            'Método': info['method'],
                            'Total_Outliers': info['outlier_count'],
                            'Porcentaje': f"{info['outlier_percentage']:.2f}%",
                            'Detalles': self._format_method_details(info)
                        })
        
        # Crear DataFrame del reporte
        report_df = pd.DataFrame(all_results)
        
        if len(report_df) > 0:
            print('\n' + tabulate(report_df, headers='keys', tablefmt='grid', showindex=False))
            
            # Estadísticas del reporte
            print(f'\n{GREEN}ESTADÍSTICAS DEL ANÁLISIS:{RESET}')
            print(f'  Columnas analizadas: {len(report_df["Columna"].unique())}')
            print(f'  Métodos aplicados: {len(report_df["Método"].unique())}')
            print(f'  Promedio de outliers por columna: {report_df["Total_Outliers"].mean():.1f}')
            print(f'  Columna con más outliers: {report_df.loc[report_df["Total_Outliers"].idxmax(), "Columna"]}')
        
        return report_df
    
    def _format_method_details(self, info):
        """Formatear detalles específicos del método."""
        if info['method'] == 'IQR':
            return f"Q1: {info['Q1']:.2f}, Q3: {info['Q3']:.2f}"
        elif info['method'] == 'Z-Score':
            return f"Threshold: {info['threshold']}, σ: {info['std']:.2f}"
        elif info['method'] == 'Modified Z-Score':
            return f"Threshold: {info['threshold']}, MAD: {info['mad']:.2f}"
        return ""
    
    def apply_treatment_recommendations(self):
        """Aplicar recomendaciones de tratamiento basadas en el análisis."""
        print('\n' + '='*80)
        print(f'{MAGENTA}APLICANDO TRATAMIENTOS RECOMENDADOS{RESET}')
        print('='*80)
        
        # Crear pipeline de tratamiento
        pipeline = OutlierTreatmentPipeline(self.df)
        
        # Aplicar tratamiento automático conservador
        pipeline.auto_treat_numeric_columns(
            method='iqr',
            treatment='cap',  # Más conservador que eliminar
            exclude_columns=['date']  # Excluir columnas no numéricas
        )
        
        # Generar reporte de tratamiento
        summary = pipeline.get_summary_report()
        
        return pipeline.data


def main():
    """Función principal para ejecutar el análisis completo."""
    print(f"{GREEN}{'='*80}")
    print("    ANÁLISIS COMPREHENSIVO DE OUTLIERS POR COLUMNA")
    print("         Procesamiento Automático del Dataset ETL")
    print(f"{'='*80}{RESET}")
    
    # Inicializar análisis
    analyzer = ComprehensiveOutlierAnalysis('dataset.csv')
    
    # Ejecutar análisis por grupos
    analyzer.analyze_all_groups()
    
    # Generar reporte final
    report_df = analyzer.generate_comprehensive_report()
    
    # Crear visualizaciones
    print(f"\n{BLUE}Generando visualizaciones...{RESET}")
    analyzer.create_visualization_dashboard()
    
    # Aplicar tratamientos recomendados
    treated_data = analyzer.apply_treatment_recommendations()
    
    # Guardar dataset tratado
    output_path = 'dataset_sin_outliers.csv'
    treated_data.to_csv(output_path, index=False)
    print(f"\n{GREEN}Dataset sin outliers guardado en: {output_path}{RESET}")
    
    print(f"\n{GREEN}¡Análisis de outliers completado!{RESET}")
    
    return analyzer, treated_data, report_df


if __name__ == "__main__":
    analyzer, treated_data, report = main()