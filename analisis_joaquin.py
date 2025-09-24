import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skew, kurtosis

BLUE = '\033[94m'
RESET = '\033[0m'
GRAY = '\033[90m'
MAGENTA = '\033[95m'

path = './dataset.csv'
df = pd.read_csv(path)

print('='*60)
print(f'{MAGENTA}COLUMNAS{RESET}')
print(' ')
for col in df.columns:
    print(f'{GRAY}{col}{RESET}')

columns= [
    'date',
    'T_out',
    'RH_out',
    'Tdewpoint'
]
print('='*60)
print(f'{MAGENTA}DATOS METEOROLÓGICOS (PARTE 1){RESET}')
df_meteorologicos_pt1 = df[columns]
print(f'{GRAY}{df_meteorologicos_pt1}{RESET}')

print('='*60)
print(f'{MAGENTA}MÉTRICAS ESTADÍSTICAS{RESET}')
print(f'{BLUE}T_out{RESET}: Temperatura exterior en grados Celsius °C.\n')
print(f'{BLUE}RH_out{RESET}: Humedad exterior en porcentaje %.\n')
print(f'{BLUE}Tdewpoint{RESET}: Punto de rocío en grados Celsius °C. Temperatura a la cual el aire debe enfriarse para que el vapor de agua contenido en él se condense formando rocío o escarcha.\n')
columns_describe = [
    'T_out',
    'RH_out',
    'Tdewpoint'
]
df_stats = df[columns_describe]
df_meteorologicos_pt1_describe = df_stats.describe(include="all")

# Calculat límites inferior y superior
lower_whisker_series = []
upper_whisker_series = []
skewness_series = []
kurtosis_series = []

for col in ['T_out', 'RH_out', 'Tdewpoint']:
    Q1 = df_stats[col].quantile(0.25)
    Q3 = df_stats[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR

    # Whiskers reales: extremos dentro del rango
    lower_whisker_val = df_stats[col][df_stats[col] >= lower_limit].min()
    upper_whisker_val = df_stats[col][df_stats[col] <= upper_limit].max()

    # Calcular skewness y kurtosis
    skewness_val = skew(df_stats[col].dropna())
    kurtosis_val = kurtosis(df_stats[col].dropna())

    lower_whisker_series.append(lower_whisker_val)
    upper_whisker_series.append(upper_whisker_val)
    skewness_series.append(skewness_val)
    kurtosis_series.append(kurtosis_val)

# Agregar a la tabla describe
df_meteorologicos_pt1_describe.loc['Lower whisker'] = lower_whisker_series
df_meteorologicos_pt1_describe.loc['Upper whisker'] = upper_whisker_series
df_meteorologicos_pt1_describe.loc['Skewness'] = skewness_series
df_meteorologicos_pt1_describe.loc['Kurtosis'] = kurtosis_series

df_meteorologicos_pt1_describe = df_meteorologicos_pt1_describe.reindex(index={
    "count": "Cantidad",
    "mean": "Media",
    "std": "Desvío estándar",
    "min": "Mínimo",
    "max": "Máximo",
    "Lower whisker": "Límite inferior",
    "Upper whisker": "Límite superior",
    "25%": "Cuartil Q1 (25%)",
    "50%": "Mediana Q2 (50%)",
    "75%": "Cuartil Q3 (75%)",
    "Skewness": "Asimetría (Skewness)",
    "Kurtosis": "Curtosis"
})

df_meteorologicos_pt1_describe = df_meteorologicos_pt1_describe.rename(index={
    "count": "Cantidad",
    "mean": "Media",
    "std": "Desvío estándar",
    "min": "Mínimo",
    "max": "Máximo",
    "Lower whisker": "Límite inferior",
    "Upper whisker": "Límite superior",
    "25%": "Cuartil Q1 (25%)",
    "50%": "Mediana Q2 (50%)",
    "75%": "Cuartil Q3 (75%)",
    "Skewness": "Asimetría (Skewness)",
    "Kurtosis": "Curtosis"
})

print(tabulate(df_meteorologicos_pt1_describe, headers='keys', tablefmt='fancy_grid'))

print(f'{BLUE}Límite inferior{RESET}: valor mínimo dentro del rango aceptable. '
      'Se calcula como el menor valor que sigue siendo mayor o igual a Q1 - 1.5 * IQR. '
      'Cualquier dato por debajo de este valor se considera extremo.\n')

print(f'{BLUE}Límite superior{RESET}: valor máximo dentro del rango aceptable. '
      'Se calcula como el mayor valor que sigue siendo menor o igual a Q3 + 1.5 * IQR. '
      'Cualquier dato por encima de este valor se considera extremo.\n')

print(f'{GRAY}Referencias: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html{RESET}')

print(f'{BLUE}Asimetría (Skewness){RESET}:')
print(f'  • ≈ 0: Distribución simétrica')
print(f'  • > 0: Sesgo positivo (cola a la derecha)')
print(f'  • < 0: Sesgo negativo (cola a la izquierda)')
print(f'  • |skewness| > 1: Considerada altamente sesgada\n')

print(f'{BLUE}Curtosis{RESET}:')
print(f'  • ≈ 0: Distribución normal (mesocúrtica)')
print(f'  • > 0: Colas pesadas (leptocúrtica) - más valores extremos')
print(f'  • < 0: Colas ligeras (platicúrtica) - menos valores extremos\n')

print('='*60)

# Interpretación adicional de skewness y kurtosis
print(f'{MAGENTA}INTERPRETACIÓN DE SKEWNESS Y KURTOSIS{RESET}')
for i, col in enumerate(['T_out', 'RH_out', 'Tdewpoint']):
    skew_val = skewness_series[i]
    kurt_val = kurtosis_series[i]
    
    print(f'\n{BLUE}{col}{RESET}:')
    
    # Interpretación skewness
    if abs(skew_val) < 0.5:
        skew_interpretation = "Distribución aproximadamente simétrica"
    elif 0.5 <= abs(skew_val) < 1:
        skew_interpretation = "Distribución moderadamente sesgada"
    else:
        skew_interpretation = "Distribución altamente sesgada"
    
    if skew_val > 0:
        skew_interpretation += " hacia la derecha"
    elif skew_val < 0:
        skew_interpretation += " hacia la izquierda"
    
    # Interpretación kurtosis
    if abs(kurt_val) < 0.5:
        kurt_interpretation = "Distribución normal (mesocúrtica)"
    elif kurt_val > 0.5:
        kurt_interpretation = "Colas pesadas (leptocúrtica) - más propensa a outliers"
    else:
        kurt_interpretation = "Colas ligeras (platicúrtica) - menos propensa a outliers"
    
    print(f'  Skewness: {skew_val:.3f} - {skew_interpretation}')
    print(f'  Kurtosis: {kurt_val:.3f} - {kurt_interpretation}')
print('='*60)
# --- Grilla de gráficos ---
df_meteorologicos_pt1 = df[columns].copy()
df_meteorologicos_pt1['date'] = pd.to_datetime(df_meteorologicos_pt1['date'], dayfirst=True)

# Extraer la hora del día
df_meteorologicos_pt1['hour'] = df_meteorologicos_pt1['date'].dt.hour

# Create a 5x3 grid of subplots
fig = plt.figure(figsize=(20, 20))

# Define the grid layout
gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1])

# Row 1: Line chart of T_out and Tdewpoint (span all columns)
ax1 = fig.add_subplot(gs[0, :])
# Row 2: Line chart of RH_out (span all columns)
ax2 = fig.add_subplot(gs[1, :])
# Row 3: Histograms
ax3 = fig.add_subplot(gs[2, 0])  # T_out histogram
ax4 = fig.add_subplot(gs[2, 1])  # Tdewpoint histogram
ax5 = fig.add_subplot(gs[2, 2])  # RH_out histogram
# Row 4: Boxplots
ax6 = fig.add_subplot(gs[3, 0:2])  # Combined boxplot for T_out and Tdewpoint
ax7 = fig.add_subplot(gs[3, 2])    # Boxplot for RH_out
# Row 5: Hourly patterns with error bars
ax8 = fig.add_subplot(gs[4, 0])    # T_out hourly pattern
ax9 = fig.add_subplot(gs[4, 1])    # Tdewpoint hourly pattern
ax10 = fig.add_subplot(gs[4, 2])   # RH_out hourly pattern

# 1. Line chart of T_out and Tdewpoint
ax1.plot(df_meteorologicos_pt1['date'], df_meteorologicos_pt1['T_out'], label='T_out (°C)', color='red', linewidth=1)
ax1.plot(df_meteorologicos_pt1['date'], df_meteorologicos_pt1['Tdewpoint'], label='Tdewpoint (°C)', color='green', linewidth=1)
ax1.set_title("Temperaturas a lo largo del tiempo")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Grados Celsius °C")
ax1.legend()
ax1.grid(True)

# 2. Line chart of RH_out
ax2.plot(df_meteorologicos_pt1['date'], 
    df_meteorologicos_pt1['RH_out'], 
    label='RH_out (%)', 
    color='blue',
    linewidth=1)
ax2.set_title("Humedad exterior a lo largo del tiempo")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Humedad (%)")
ax2.legend()
ax2.grid(True)

# 3. Histogram for T_out
mean_t_out = df_stats['T_out'].mean()
median_t_out = df_stats['T_out'].median()
std_t_out = df_stats['T_out'].std()
n, bins, patches = ax3.hist(df_stats['T_out'], bins=30, density=False, alpha=0.7, color='red', edgecolor='black')
x_t_out = np.linspace(mean_t_out - 3*std_t_out, mean_t_out + 3*std_t_out, 100)
y_t_out = norm.pdf(x_t_out, mean_t_out, std_t_out)
# ax3.plot(x_t_out, y_t_out, 'k-', linewidth=2)

# Agregar líneas verticales para media y mediana
ax3.axvline(mean_t_out, color='blue', linestyle='--', linewidth=1, label=f'Media: {mean_t_out:.2f}°C')
ax3.axvline(median_t_out, color='green', linestyle='--', linewidth=1, label=f'Mediana: {median_t_out:.2f}°C')
ax3.legend()

ax3.set_title(f'Distribución de T_out\n(μ={mean_t_out:.2f}, σ={std_t_out:.2f})')
ax3.set_xlabel('Temperatura exterior (°C)')
ax3.set_ylabel('Frecuencia')
ax3.grid(True, alpha=0.3)

# 4. Histogram for Tdewpoint
mean_tdew = df_stats['Tdewpoint'].mean()
median_tdew = df_stats['Tdewpoint'].median()
std_tdew = df_stats['Tdewpoint'].std()
n, bins, patches = ax4.hist(df_stats['Tdewpoint'], bins=30, density=False, alpha=0.7, color='green', edgecolor='black')
x_tdew = np.linspace(mean_tdew - 3*std_tdew, mean_tdew + 3*std_tdew, 100)
y_tdew = norm.pdf(x_tdew, mean_tdew, std_tdew)
# ax4.plot(x_tdew, y_tdew, 'k-', linewidth=2)

# Agregar líneas verticales para media y mediana
ax4.axvline(mean_tdew, color='blue', linestyle='--', linewidth=1, label=f'Media: {mean_tdew:.2f}°C')
ax4.axvline(median_tdew, color='red', linestyle='--', linewidth=1, label=f'Mediana: {median_tdew:.2f}°C')
ax4.legend()

ax4.set_title(f'Distribución de Tdewpoint\n(μ={mean_tdew:.2f}, σ={std_tdew:.2f})')
ax4.set_xlabel('Temperatura de punto de rocío (°C)')
ax4.set_ylabel('Frecuencia')
ax4.grid(True, alpha=0.3)

# 5. Histogram for RH_out
mean_rh = df_stats['RH_out'].mean()
median_rh = df_stats['RH_out'].median()
std_rh = df_stats['RH_out'].std()
n, bins, patches = ax5.hist(df_stats['RH_out'], bins=30, density=False, alpha=0.7, color='blue', edgecolor='black')
x_rh = np.linspace(mean_rh - 3*std_rh, mean_rh + 3*std_rh, 100)
y_rh = norm.pdf(x_rh, mean_rh, std_rh)
# ax5.plot(x_rh, y_rh, 'k-', linewidth=2)

# Agregar líneas verticales para media y mediana
ax5.axvline(mean_rh, color='red', linestyle='--', linewidth=1, label=f'Media: {mean_rh:.2f}%')
ax5.axvline(median_rh, color='green', linestyle='--', linewidth=1, label=f'Mediana: {median_rh:.2f}%')
ax5.legend()

ax5.set_title(f'Distribución de RH_out\n(μ={mean_rh:.2f}, σ={std_rh:.2f})')
ax5.set_xlabel('Porcentaje de humedad (%)')
ax5.set_ylabel('Frecuencia')
ax5.grid(True, alpha=0.3)

# 6. Combined boxplot for T_out and Tdewpoint
sns.boxplot(data=df_stats[['T_out','Tdewpoint']], ax=ax6, 
            palette=['red', 'green'], orient="h")
ax6.set_title("Diagrama de cajas para T_out y Tdewpoint")
ax6.set_xlabel("Temperatura (°C)")
ax6.set_ylabel("Variables")

# 7. Boxplot for RH_out
sns.boxplot(data=df_stats['RH_out'], ax=ax7, color='blue', orient="h")
ax7.set_title("Diagrama de cajas para RH_out")
ax7.set_xlabel("Porcentaje de humedad (%)")
ax7.set_ylabel("RH_out (%)")

# 8. Hourly pattern for T_out
hourly_t_out = df_meteorologicos_pt1.groupby('hour')['T_out'].agg(['mean', 'std']).reset_index()
ax8.errorbar(hourly_t_out['hour'], hourly_t_out['mean'], yerr=hourly_t_out['std'], 
             fmt='-o', alpha=0.7, capsize=5, color='red')
ax8.set_title('T_out por Hora del Día (± desviación estándar)')
ax8.set_xlabel('Hora del Día')
ax8.set_ylabel('T_out Promedio (°C)')
ax8.grid(True, alpha=0.3)
# Formatear el eje x como horas
ax8.set_xticks(range(0, 24, 2))  # Mostrar marcas cada 2 horas
ax8.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)  # Formato HH:00

# 9. Hourly pattern for Tdewpoint
hourly_tdew = df_meteorologicos_pt1.groupby('hour')['Tdewpoint'].agg(['mean', 'std']).reset_index()
ax9.errorbar(hourly_tdew['hour'], hourly_tdew['mean'], yerr=hourly_tdew['std'], 
             fmt='-o', alpha=0.7, capsize=5, color='green')
ax9.set_title('Tdewpoint por Hora del Día (± desviación estándar)')
ax9.set_xlabel('Hora del Día')
ax9.set_ylabel('Tdewpoint Promedio (°C)')
ax9.grid(True, alpha=0.3)
# Formatear el eje x como horas
ax9.set_xticks(range(0, 24, 2))  # Mostrar marcas cada 2 horas
ax9.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)  # Formato HH:00

# 10. Hourly pattern for RH_out
hourly_rh = df_meteorologicos_pt1.groupby('hour')['RH_out'].agg(['mean', 'std']).reset_index()
ax10.errorbar(hourly_rh['hour'], hourly_rh['mean'], yerr=hourly_rh['std'], 
              fmt='-o', alpha=0.7, capsize=5, color='blue')
ax10.set_title('RH_out por Hora del Día (± desviación estándar)')
ax10.set_xlabel('Hora del Día')
ax10.set_ylabel('RH_out Promedio (%)')
ax10.grid(True, alpha=0.3)
# Formatear el eje x como horas
ax10.set_xticks(range(0, 24, 2))  # Mostrar marcas cada 2 horas
ax10.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)  # Formato HH:00

plt.tight_layout()
plt.subplots_adjust(wspace=0.229, hspace=0.995, bottom=0.088, top=0.967)
plt.show()