#!/usr/bin/env python3
"""
Ejecutor Rápido de Análisis de Outliers
=======================================

Script simple para ejecutar análisis de outliers desde línea de comandos.

Usage:
    python run_outlier_analysis.py [dataset_path] [--treatment METHOD]

Author: ETL Team
Date: 28 de septiembre de 2025
"""

import sys
import argparse
import pandas as pd
from outlier_treatment import OutlierDetector, OutlierTreatmentPipeline


def quick_outlier_analysis(dataset_path='dataset.csv', treatment_method=None):
    """
    Ejecutar análisis rápido de outliers.
    
    Args:
        dataset_path (str): Ruta al dataset
        treatment_method (str): Método de tratamiento ('cap', 'remove', 'median', None)
    """
    
    print("=" * 80)
    print("    ANÁLISIS RÁPIDO DE OUTLIERS - PIPELINE ETL")
    print("=" * 80)
    
    # Cargar datos
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset cargado: {df.shape}")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return
    
    # Información básica
    print(f"\n📊 INFORMACIÓN DEL DATASET:")
    print(f"   Filas: {len(df):,}")
    print(f"   Columnas: {len(df.columns)}")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"   Columnas numéricas: {len(numeric_cols)}")
    
    # Analizar outliers por columna
    print(f"\n🔍 ANÁLISIS DE OUTLIERS:")
    print("-" * 60)
    
    detector = OutlierDetector(df)
    summary_results = []
    
    for col in numeric_cols[:10]:  # Limitar a primeras 10 columnas para velocidad
        print(f"\n📈 Analizando: {col}")
        
        # Análisis con método IQR
        results = detector.analyze_column_outliers(col, methods=['iqr'])
        
        if 'iqr' in results and results['iqr']:
            info = results['iqr']
            outlier_count = info['outlier_count']
            outlier_pct = info['outlier_percentage']
            
            print(f"   Outliers detectados: {outlier_count} ({outlier_pct:.2f}%)")
            print(f"   Rango normal: [{info['lower_whisker']:.2f}, {info['upper_whisker']:.2f}]")
            
            summary_results.append({
                'Columna': col,
                'Outliers': outlier_count,
                'Porcentaje': f"{outlier_pct:.2f}%",
                'Límite_Inferior': f"{info['lower_bound']:.2f}",
                'Límite_Superior': f"{info['upper_bound']:.2f}"
            })
        else:
            print(f"   ⚠️  No se pudo analizar la columna {col}")
    
    # Resumen
    if summary_results:
        print(f"\n📋 RESUMEN DE OUTLIERS:")
        print("-" * 60)
        
        summary_df = pd.DataFrame(summary_results)
        print(summary_df.to_string(index=False))
        
        total_outliers = summary_df['Outliers'].sum()
        avg_outliers = summary_df['Outliers'].mean()
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   Total de outliers: {total_outliers}")
        print(f"   Promedio por columna: {avg_outliers:.1f}")
        print(f"   Columnas con outliers: {len(summary_df[summary_df['Outliers'] > 0])}")
    
    # Aplicar tratamiento si se especifica
    if treatment_method:
        print(f"\n🔧 APLICANDO TRATAMIENTO: {treatment_method.upper()}")
        print("-" * 60)
        
        pipeline = OutlierTreatmentPipeline(df)
        
        if treatment_method in ['cap', 'remove', 'median']:
            pipeline.auto_treat_numeric_columns(
                method='iqr',
                treatment=treatment_method,
                exclude_columns=['date']
            )
            
            # Mostrar resultado del tratamiento
            treated_summary = pipeline.get_summary_report()
            
            # Guardar dataset tratado
            output_path = f'dataset_treated_{treatment_method}.csv'
            pipeline.data.to_csv(output_path, index=False)
            
            print(f"✅ Dataset tratado guardado: {output_path}")
            print(f"   Filas originales: {len(df)}")
            print(f"   Filas finales: {len(pipeline.data)}")
            print(f"   Retención: {(len(pipeline.data)/len(df)*100):.1f}%")
        else:
            print(f"❌ Método de tratamiento inválido: {treatment_method}")
            print("   Métodos válidos: cap, remove, median")
    
    print(f"\n🎉 Análisis de outliers completado!")
    return summary_results


def main():
    """Función principal con argumentos de línea de comandos."""
    
    parser = argparse.ArgumentParser(
        description="Análisis rápido de outliers para dataset ETL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    python run_outlier_analysis.py
    python run_outlier_analysis.py dataset.csv
    python run_outlier_analysis.py dataset.csv --treatment cap
    python run_outlier_analysis.py --treatment remove
        """
    )
    
    parser.add_argument(
        'dataset', 
        nargs='?', 
        default='dataset.csv',
        help='Ruta al archivo CSV del dataset (default: dataset.csv)'
    )
    
    parser.add_argument(
        '--treatment', 
        choices=['cap', 'remove', 'median'],
        help='Método de tratamiento de outliers'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Análisis rápido (solo primeras 5 columnas)'
    )
    
    args = parser.parse_args()
    
    # Ejecutar análisis
    try:
        results = quick_outlier_analysis(args.dataset, args.treatment)
        
        if results:
            print(f"\n💾 Resultados disponibles para procesamiento adicional")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()