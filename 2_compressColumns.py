import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Cargar el dataset original
df = pd.read_csv('dataset.csv')

# Convertir fecha (según tu código anterior)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')

# =============================================================================
# ESTRATEGIA DE COMPRESIÓN HÍBRIDA
# =============================================================================

print("=== INICIO COMPRESIÓN DE VARIABLES ===")

# Crear nueva versión del dataset
df_compressed = df.copy()

# 1. COMPRESIÓN DE TEMPERATURAS (9 → 2 variables)
df_compressed['T_avg'] = df[['T1','T2','T3','T4','T5','T6','T7','T8','T9']].mean(axis=1)
df_compressed['T_range'] = df[['T1','T2','T3','T4','T5','T6','T7','T8','T9']].max(axis=1) - df[['T1','T2','T3','T4','T5','T6','T7','T8','T9']].min(axis=1)

# 2. COMPRESIÓN DE HUMEDADES (9 → 2 variables)
df_compressed['RH_avg'] = df[['RH_1','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7','RH_8','RH_9']].mean(axis=1)
df_compressed['RH_range'] = df[['RH_1','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7','RH_8','RH_9']].max(axis=1) - df[['RH_1','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7','RH_8','RH_9']].min(axis=1)

# 3. ELIMINAR COLUMNAS ORIGINALES COMPRIMIDAS
columnas_a_eliminar = [
    'T1','T2','T3','T4','T5','T6','T7','T8','T9',
    'RH_1','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7','RH_8','RH_9'
]

# Eliminar columnas originales comprimidas
df_compressed_final = df_compressed.drop(columns=columnas_a_eliminar)

# =============================================================================
# ELIMINACIÓN DE VARIABLES ALEATORIAS (rv1, rv2)
# =============================================================================

print("\n=== ANÁLISIS DE VARIABLES ALEATORIAS ===")

# Verificar si existen las variables aleatorias
if all(col in df_compressed_final.columns for col in ['rv1', 'rv2']):
    # Análisis de correlación con el target
    correlacion_rv = df_compressed_final[['Appliances', 'rv1', 'rv2']].corr()['Appliances']
    print("📊 CORRELACIÓN CON APPLIANCES:")
    print(f"rv1: {correlacion_rv['rv1']:.4f}")
    print(f"rv2: {correlacion_rv['rv2']:.4f}")
    
    # Estadísticas básicas
    print("\n📈 ESTADÍSTICAS DE VARIABLES ALEATORIAS:")
    print(f"rv1 - Media: {df_compressed_final['rv1'].mean():.3f}, Desvío: {df_compressed_final['rv1'].std():.3f}")
    print(f"rv2 - Media: {df_compressed_final['rv2'].mean():.3f}, Desvío: {df_compressed_final['rv2'].std():.3f}")
    
    # Eliminar variables aleatorias
    df_final_limpio = df_compressed_final.drop(columns=['rv1', 'rv2'])
    print(f"\n✅ ELIMINADAS VARIABLES ALEATORIAS: rv1, rv2")
    
else:
    df_final_limpio = df_compressed_final.copy()
    print("ℹ️  Variables aleatorias no encontradas en el dataset")

# =============================================================================
# VALIDACIÓN DE LA COMPRESIÓN
# =============================================================================

print("\n=== VALIDACIÓN DE LA COMPRESIÓN ===")

# 1. Estadísticas de las nuevas variables
print("📊 ESTADÍSTICAS DE NUEVAS VARIABLES:")
print(f"T_avg: {df_compressed['T_avg'].mean():.2f}°C (rango: {df_compressed['T_avg'].min():.1f}-{df_compressed['T_avg'].max():.1f}°C)")
print(f"T_range: {df_compressed['T_range'].mean():.2f}°C (rango: {df_compressed['T_range'].min():.1f}-{df_compressed['T_range'].max():.1f}°C)")
print(f"RH_avg: {df_compressed['RH_avg'].mean():.2f}% (rango: {df_compressed['RH_avg'].min():.1f}-{df_compressed['RH_avg'].max():.1f}%)")
print(f"RH_range: {df_compressed['RH_range'].mean():.2f}% (rango: {df_compressed['RH_range'].min():.1f}-{df_compressed['RH_range'].max():.1f}%)")

# 2. Correlación con el target (Appliances)
print("\n🔗 CORRELACIÓN CON CONSUMO ENERGETICO (APPLIANCES):")
correlaciones = df_compressed[['Appliances', 'T_avg', 'T_range', 'RH_avg', 'RH_range']].corr()['Appliances']
for var, corr in correlaciones.items():
    if var != 'Appliances':
        print(f"{var}: {corr:.3f}")

# 3. Reducción de dimensionalidad
print(f"\n📉 REDUCCIÓN DE DIMENSIONALIDAD:")
print(f"Columnas originales: {df.shape[1]}")
print(f"Columnas después de compresión: {df_final_limpio.shape[1]}")
print(f"Reducción: {((df.shape[1] - df_final_limpio.shape[1]) / df.shape[1]) * 100:.1f}%")

# 4. Verificar que no se pierde información crítica
print(f"\n✅ CONSISTENCIA DE DATOS:")
print(f"Filas originales: {df.shape[0]}")
print(f"Filas después de compresión: {df_final_limpio.shape[0]}")
print(f"Valores nulos introducidos: {df_final_limpio.isnull().sum().sum()}")

# =============================================================================
# VISUALIZACIÓN DE LA COMPRESIÓN
# =============================================================================

# Gráfico de comparación
plt.figure(figsize=(12, 8))

# Comparación de temperaturas
plt.subplot(2, 2, 1)
plt.scatter(df['T1'], df_compressed['T_avg'], alpha=0.5)
plt.xlabel('T1 (Sensor individual)')
plt.ylabel('T_avg (Promedio 9 sensores)')
plt.title('Relación T1 vs T_avg')

# Distribución del rango de temperaturas
plt.subplot(2, 2, 2)
plt.hist(df_compressed['T_range'], bins=30, alpha=0.7)
plt.xlabel('Rango de Temperatura (°C)')
plt.ylabel('Frecuencia')
plt.title('Distribución del Rango de Temperaturas')

# Correlación con appliances
plt.subplot(2, 2, 3)
corr_matrix = df_compressed[['Appliances', 'T_avg', 'RH_avg']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlación con Consumo Energético')

# Serie temporal de temperatura promedio
plt.subplot(2, 2, 4)
plt.plot(df_compressed['date'], df_compressed['T_avg'], alpha=0.7)
plt.xlabel('Fecha')
plt.ylabel('Temperatura Promedio (°C)')
plt.title('Evolución Temporal - Temperatura Promedio')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================
# GUARDAR NUEVO DATASET
# =============================================================================

# Guardar dataset final limpio
df_final_limpio.to_csv('dataset_final_limpio.csv', index=False)

print(f"\n💾 DATASET FINAL GUARDADO:")
print(f"Archivo: dataset_final_limpio.csv")
print(f"Dimensiones: {df_final_limpio.shape}")
print(f"Memoria: {df_final_limpio.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# =============================================================================
# RESUMEN FINAL DEL ETL
# =============================================================================

print("\n" + "="*60)
print("🎯 RESUMEN FINAL DEL PROCESO ETL")
print("="*60)

print(f"📊 DATASET ORIGINAL:")
print(f"   - Filas: {df.shape[0]}")
print(f"   - Columnas: {df.shape[1]}")
print(f"   - Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n🔄 TRANSFORMACIONES APLICADAS:")
print(f"   1. ✅ Compresión temperaturas: 9 variables → 2 variables (T_avg, T_range)")
print(f"   2. ✅ Compresión humedades: 9 variables → 2 variables (RH_avg, RH_range)") 
print(f"   3. ✅ Eliminación variables aleatorias: rv1, rv2")

print(f"\n📈 DATASET FINAL:")
print(f"   - Filas: {df_final_limpio.shape[0]}")
print(f"   - Columnas: {df_final_limpio.shape[1]}")
print(f"   - Reducción dimensionalidad: {((df.shape[1] - df_final_limpio.shape[1]) / df.shape[1]) * 100:.1f}%")
print(f"   - Memoria: {df_final_limpio.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n🎯 VARIABLES FINALES ({df_final_limpio.shape[1]} columnas):")
for i, col in enumerate(df_final_limpio.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n🎯 ETL FINALIZADO EXITOSAMENTE!")