import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración para mejor visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cargar el dataset
df = pd.read_csv('dataset.csv')

# Mostrar información básica
print("=== INFORMACIÓN BÁSICA DEL DATASET ===")
print(f"Dimensiones: {df.shape} (filas, columnas)")
print(f"Total de celdas: {df.shape[0] * df.shape[1]}")
print("\nPrimeras 5 filas:")
print(df.head())

print("=== ANÁLISIS DE VALORES NULOS ===")

# Verificar valores nulos por columna
null_analysis = df.isnull().sum()
null_percentage = (df.isnull().sum() / len(df)) * 100

# Crear DataFrame con el análisis
null_summary = pd.DataFrame({
    'Valores_Nulos': null_analysis,
    'Porcentaje_Nulos': null_percentage
})

# Filtrar solo columnas con valores nulos
null_summary = null_summary[null_summary['Valores_Nulos'] > 0]

if null_summary.empty:
    print("✅ No hay valores nulos en el dataset")
else:
    print("❌ Se encontraron valores nulos:")
    print(null_summary)
    print(f"\nTotal de valores nulos: {df.isnull().sum().sum()}")

print("\n=== ANÁLISIS DE VALORES INCONSISTENTES ===")

# Función para detectar valores potencialmente problemáticos
def check_inconsistent_values(series, series_name):
    issues = []
    
    # Verificar valores infinitos
    if np.any(np.isinf(series)):
        issues.append(f"Valores infinitos encontrados")
    
    # Verificar valores extremos (fuera de rangos esperados)
    if series.dtype in ['float64', 'int64']:
        # Para temperaturas (esperamos valores entre -50 y 50°C)
        if 'T' in series_name and not series_name.startswith('RH'):
            if series.min() < -50 or series.max() > 50:
                issues.append(f"Temperaturas fuera de rango razonable: [{series.min()}, {series.max()}]")
        
        # Para humedad relativa (esperamos 0-100%)
        if series_name.startswith('RH'):
            if series.min() < 0 or series.max() > 100:
                issues.append(f"Humedad fuera de rango 0-100%: [{series.min()}, {series.max()}]")
        
        # Para presión atmosférica (esperamos valores razonables)
        if series_name == 'Press_mm_hg':
            if series.min() < 600 or series.max() > 800:
                issues.append(f"Presión fuera de rango razonable: [{series.min()}, {series.max()}]")
    
    return issues

# Aplicar la función a cada columna
inconsistent_issues = {}
for column in df.columns:
    if column != 'date':  # Excluir la columna de fecha
        issues = check_inconsistent_values(df[column], column)
        if issues:
            inconsistent_issues[column] = issues

if inconsistent_issues:
    print("❌ Se encontraron valores inconsistentes:")
    for col, issues in inconsistent_issues.items():
        print(f"  {col}: {', '.join(issues)}")
else:
    print("✅ No se encontraron valores inconsistentes evidentes")



print("\n=== ANÁLISIS DE LA COLUMNA DATE ===")

# Convertir a datetime y verificar consistencia
try:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
    print("✅ Formato de fecha correcto")
    
    # Verificar si hay duplicados temporales
    duplicate_dates = df['date'].duplicated().sum()
    if duplicate_dates > 0:
        print(f"❌ Se encontraron {duplicate_dates} fechas duplicadas")
    else:
        print("✅ No hay fechas duplicadas")
        
    # Verificar continuidad temporal
    time_diff = df['date'].diff().value_counts()
    expected_interval = pd.Timedelta(minutes=10)  # Asumiendo mediciones cada 10 min
    most_common_interval = time_diff.index[0]
    
    if most_common_interval == expected_interval:
        print("✅ Intervalo temporal consistente (10 minutos)")
    else:
        print(f"⚠️  Intervalo temporal inusual: {most_common_interval}")
        
except Exception as e:
    print(f"❌ Error en formato de fecha: {e}")


print("\n" + "="*50)
print("RESUMEN EJECUTIVO DE LA LIMPIEZA")
print("="*50)

# Resumen de problemas encontrados
issues_found = []

if not null_summary.empty:
    issues_found.append(f"Valores nulos: {df.isnull().sum().sum()} celdas")

if inconsistent_issues:
    issues_found.append(f"Valores inconsistentes: {len(inconsistent_issues)} columnas")


if issues_found:
    print("❌ PROBLEMAS ENCONTRADOS:")
    for issue in issues_found:
        print(f"  - {issue}")
else:
    print("✅ DATASET LIMPIO - No se encontraron problemas críticos")

print(f"\nTotal de filas: {df.shape[0]}")
print(f"Total de columnas: {df.shape[1]}")
print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")