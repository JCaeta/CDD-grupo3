import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset original
df = pd.read_csv('dataset.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')

print("=== ESTRATEGIA DE COMPRESIÓN SIMPLIFICADA ===")

# Crear nueva versión del dataset
df_compressed = df.copy()

# =============================================================================
# COMPRESIÓN SIMPLIFICADA - SOLO PROMEDIOS, SIN RANGE
# =============================================================================

# 1. COMPRESIÓN DE TEMPERATURAS INTERIORES (7 variables → 1 variable)
temperaturas_interiores = ['T1','T2','T3','T4','T5','T7','T8']  # Excluyendo T6, T9 (exteriores)
df_compressed['T_int_avg'] = df[temperaturas_interiores].mean(axis=1)
# ❌ ELIMINADO: T_int_range

# 2. COMPRESIÓN DE HUMEDADES INTERIORES (7 variables → 1 variable)  
humedades_interiores = ['RH_1','RH_2','RH_3','RH_4','RH_5','RH_7','RH_8']  # Excluyendo RH_6, RH_9 (exteriores)
df_compressed['RH_int_avg'] = df[humedades_interiores].mean(axis=1)
# ❌ ELIMINADO: RH_int_range

# =============================================================================
# ELIMINACIÓN DE COLUMNAS ORIGINALES
# =============================================================================

columnas_a_eliminar = [
    # Temperaturas y humedades interiores (individuales)
    'T1','T2','T3','T4','T5','T7','T8',
    'RH_1','RH_2','RH_3','RH_4','RH_5','RH_7','RH_8',
    
    # Temperaturas y humedades exteriores redundantes
    'T6','T9','RH_6','RH_9',  # Estas son redundantes con T_out y RH_out
    
    # Variables aleatorias
    'rv1', 'rv2'
]

# Eliminar columnas (manteniendo T_out, RH_out y los promedios)
df_final_limpio = df_compressed.drop(columns=[col for col in columnas_a_eliminar if col in df_compressed.columns])

# =============================================================================
# VALIDACIÓN DE LA COMPRESIÓN
# =============================================================================

print("\n=== VALIDACIÓN DE LA COMPRESIÓN ===")

# 1. Estadísticas de las variables finales
print("📈 ESTADÍSTICAS DE VARIABLES FINALES:")
print(f"T_int_avg: {df_final_limpio['T_int_avg'].mean():.2f}°C (rango: {df_final_limpio['T_int_avg'].min():.1f}-{df_final_limpio['T_int_avg'].max():.1f}°C)")
print(f"T_out: {df_final_limpio['T_out'].mean():.2f}°C (rango: {df_final_limpio['T_out'].min():.1f}-{df_final_limpio['T_out'].max():.1f}°C)")
print(f"RH_int_avg: {df_final_limpio['RH_int_avg'].mean():.2f}% (rango: {df_final_limpio['RH_int_avg'].min():.1f}-{df_final_limpio['RH_int_avg'].max():.1f}%)")
print(f"RH_out: {df_final_limpio['RH_out'].mean():.2f}% (rango: {df_final_limpio['RH_out'].min():.1f}-{df_final_limpio['RH_out'].max():.1f}%)")

# 2. Correlación con el target
print("\n🔗 CORRELACIÓN CON APPLIANCES:")
correlaciones = df_final_limpio[['Appliances', 'T_int_avg', 'T_out', 'RH_int_avg', 'RH_out', 'lights']].corr()['Appliances']
for var, corr in correlaciones.items():
    if var != 'Appliances':
        importancia = "🔥 ALTA" if abs(corr) > 0.15 else "📈 MEDIA" if abs(corr) > 0.05 else "📊 BAJA"
        print(f"{var}: {corr:.3f} ({importancia})")

# =============================================================================
# VISUALIZACIÓN SIMPLIFICADA
# =============================================================================

# plt.figure(figsize=(12, 8))
# grid_cols = 2
# grid_rows = 2

# # 1. Comparación temperaturas interiores vs exteriores
# plt.subplot(grid_rows, grid_cols, 1)
# plt.scatter(df_final_limpio['T_out'], df_final_limpio['T_int_avg'], alpha=0.5, c=df_final_limpio['Appliances'], cmap='viridis')
# plt.colorbar(label='Consumo Energético')
# plt.xlabel('Temperatura Externa T_out (°C)')
# plt.ylabel('Temperatura Interna Promedio (°C)')
# plt.title('Relación Temperaturas Interna/Externa')

# # 2. Serie temporal comparativa
# plt.subplot(grid_rows, grid_cols, 2)
# plt.plot(df_final_limpio['date'], df_final_limpio['T_int_avg'], label='Interior', alpha=0.7, linewidth=1)
# plt.plot(df_final_limpio['date'], df_final_limpio['T_out'], label='Exterior T_out', alpha=0.7, linewidth=1)
# plt.xlabel('Fecha')
# plt.ylabel('Temperatura (°C)')
# plt.title('Evolución Temporal')
# plt.legend()
# plt.xticks(rotation=45)

# # 3. Correlación con appliances (heatmap)
# plt.subplot(grid_rows, grid_cols, 3)
# corr_vars = ['Appliances', 'T_int_avg', 'T_out', 'RH_int_avg', 'RH_out', 'lights']
# corr_matrix = df_final_limpio[corr_vars].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
# plt.title('Correlación con Consumo Energético')

# # 4. Comportamiento por hora del día
# plt.subplot(grid_rows, grid_cols, 4)
# df_final_limpio['hora'] = df_final_limpio['date'].dt.hour
# consumo_por_hora = df_final_limpio.groupby('hora')['Appliances'].mean()
# plt.plot(consumo_por_hora.index, consumo_por_hora.values, marker='o', color='green')
# plt.xlabel('Hora del Día')
# plt.ylabel('Consumo Promedio (Wh)')
# plt.title('Consumo Energético por Hora del Día')
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(20, 20))
grid_cols = 2
grid_rows = 3
grid_height_ratios = [1] * grid_rows
grid_height_ratios[0] = 0.75

# Define the grid layout
gs = fig.add_gridspec(grid_rows, grid_cols, height_ratios=grid_height_ratios)

# Row 1: Line chart of T_out and Tdewpoint (span all columns)
ax1 = fig.add_subplot(gs[0, :])
# Row 2: Line chart of RH_out (span all columns)
ax2 = fig.add_subplot(gs[1, 0]) # Hist appliances
ax3 = fig.add_subplot(gs[1, 1])  # Hist lights
# ax4 = fig.add_subplot(gs[1, 2])  # Hist T_out
# ax5 = fig.add_subplot(gs[1, 3])  # Hist RH_out

ax6 = fig.add_subplot(gs[2, 0])  # Boxplot Appliances
ax7 = fig.add_subplot(gs[2, 1])  # Boxplot lights

# 1. Serie temporal comparativa
ax1.plot(df_final_limpio['date'], df_final_limpio['T_int_avg'], label='Interior', alpha=0.7, linewidth=1, color='orange')
ax1.plot(df_final_limpio['date'], df_final_limpio['T_out'], label='Exterior T_out', alpha=0.7, linewidth=1, color='red')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Temperatura (°C)')
ax1.set_title('Evolución Temporal')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)


# 3. Histogram for lights
n, bins, patches = ax2.hist(df_final_limpio['Appliances'], bins=30, density=False, alpha=0.7, color='green', edgecolor='black')
ax2.set_title('Consumo de electrodomésticos (Wh)')
ax2.set_xlabel('Wh')
ax2.set_ylabel('Frecuencia')
ax2.grid(True, alpha=0.3)


# 3. Histogram for lights
n, bins, patches = ax3.hist(df_final_limpio['lights'], bins=30, density=False, alpha=0.7, color='red', edgecolor='black')
ax3.set_title('Histograma de Luces')
ax3.set_xlabel('Luces (Wh)')
ax3.set_ylabel('Frecuencia')
ax3.grid(True, alpha=0.3)


sns.boxplot(data=df_final_limpio['Appliances'], ax=ax6, orient="h", color='green')
ax6.set_title('Consumo de electrodomésticos (Wh)')

sns.boxplot(data=df_final_limpio['lights'], ax=ax7, orient="h", color='green')
ax7.set_title('Luces (Wh)')

plt.tight_layout()

## Adjust propeties
wspace = 0.179
hspace = 0.61
right = 0.983
left = 0.093
top = 0.957
bottom = 0.076

plt.subplots_adjust(wspace=wspace, hspace=hspace, right=right, left=left, top=top, bottom=bottom)
plt.show()







# =============================================================================
# RESUMEN FINAL SIMPLIFICADO
# =============================================================================

print("\n" + "="*70)
print("🎯 RESUMEN FINAL DEL ETL SIMPLIFICADO")
print("="*70)

print(f"📊 ESTRATEGIA DE COMPRESIÓN:")
print(f"   - Temperaturas INTERIORES: 7 sensores → 1 variable (T_int_avg)")
print(f"   - Humedades INTERIORES: 7 sensores → 1 variable (RH_int_avg)")
print(f"   - Temperatura EXTERIOR: Mantenemos T_out")
print(f"   - Humedad EXTERIOR: Mantenemos RH_out")
print(f"   - ❌ ELIMINADAS: Variables range (según solicitud)")

print(f"\n📈 DATASET FINAL - VARIABLES ({df_final_limpio.shape[1]} columnas):")
for i, col in enumerate(df_final_limpio.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n📉 REDUCCIÓN DE DIMENSIONALIDAD:")
print(f"   - Columnas originales: {df.shape[1]}")
print(f"   - Columnas finales: {df_final_limpio.shape[1]}")
print(f"   - Reducción: {((df.shape[1] - df_final_limpio.shape[1]) / df.shape[1]) * 100:.1f}%")

print(f"\n💡 CARACTERÍSTICAS:")
print(f"   - Dataset más compacto y simple")
print(f"   - Sin variables redundantes")
print(f"   - Ideal para modelos de machine learning")

# Guardar dataset simplificado
df_final_limpio.to_csv('dataset_final_simplificado.csv', index=False)
print(f"\n💾 DATASET SIMPLIFICADO GUARDADO: dataset_final_simplificado.csv")