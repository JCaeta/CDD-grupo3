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

fig = plt.figure(figsize=(20, 20))
grid_cols = 3
grid_rows = 3
grid_height_ratios = [1] * grid_rows
# grid_height_ratios[0] = 0.7

# Define the grid layout
gs = fig.add_gridspec(grid_rows, grid_cols, height_ratios=grid_height_ratios)

# Row 1: Line chart of T_out and Tdewpoint (span all columns)
ax1 = fig.add_subplot(gs[0, 0]) # Hist T_out
ax2 = fig.add_subplot(gs[0, 1]) # Hist T_int
ax3 = fig.add_subplot(gs[0, 2])  # Hist RH_out

ax4 = fig.add_subplot(gs[1, 0])  # Boxplot T_out
ax5 = fig.add_subplot(gs[1, 1])  # Boxplot T_int
ax6 = fig.add_subplot(gs[1, 2])  # Boxplot RH_out

ax7 = fig.add_subplot(gs[2, 1])  # Correlación

ax1.hist(df_final_limpio['T_out'], bins=30, density=False, alpha=0.7, color='red', edgecolor='black')
ax1.set_title('Temperatura exterior')
ax1.set_xlabel('Temperatura (°C)')
ax1.set_ylabel('Frecuencia')
ax1.grid(True, alpha=0.3)

ax2.hist(df_final_limpio['T_int_avg'], bins=30, density=False, alpha=0.7, color='orange', edgecolor='black')
ax2.set_title('Temperatura interior (Promedio)')
ax2.set_xlabel('Temperatura (°C)')
ax2.set_ylabel('Frecuencia')
ax2.grid(True, alpha=0.3)

ax3.hist(df_final_limpio['RH_out'], bins=30, density=False, alpha=0.7, color='blue', edgecolor='black')
ax3.set_title('Humedad exterior')
ax3.set_xlabel('Humedad (%)')
ax3.set_ylabel('Frecuencia')
ax3.grid(True, alpha=0.3)

sns.boxplot(data=df_final_limpio['T_out'], ax=ax4, orient="h", color='red')
ax4.set_title('Temperatura exterior')
ax4.set_xlabel('Temperatura (°C)')

sns.boxplot(data=df_final_limpio['T_int_avg'], ax=ax5, orient="h", color='orange')
ax5.set_title('Temperatura interior (Promedio)')
ax5.set_xlabel('Temperatura (°C)')

sns.boxplot(data=df_final_limpio['RH_out'], ax=ax6, orient="h", color='blue')
ax6.set_title('Humedad exterior')
ax6.set_xlabel('Humedad (%)')

# 2. Correlación
corr_vars = ['Appliances', 'T_int_avg', 'T_out', 'RH_int_avg', 'RH_out', 'lights']
corr_matrix = df_final_limpio[corr_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax7)
ax7.set_title('Correlación con Consumo Energético')
ax7.tick_params(axis='y', rotation=0)

plt.tight_layout()

## Adjust propeties
wspace = 0.179
hspace = 0.467
right = 0.983
left = 0.093
top = 0.957
bottom = 0.11

plt.subplots_adjust(wspace=wspace, hspace=hspace, right=right, left=left, top=top, bottom=bottom)
plt.show()

# Guardar dataset simplificado
df_final_limpio.to_csv('dataset_paso_3.csv', index=False)
print(f"\n💾 DATASET SIMPLIFICADO GUARDADO: dataset_paso_3.csv")
