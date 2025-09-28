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
    'Press_mm_hg',
    'Windspeed',
    'Visibility'
]
print('='*60)
print(f'{MAGENTA}DATOS METEOROLÓGICOS (PARTE 2){RESET}')
df_meteorologicos_pt2 = df[columns]
print(f'{GRAY}{df_meteorologicos_pt2}{RESET}')

print('='*60)
print(f'{MAGENTA}MÉTRICAS ESTADÍSTICAS{RESET}')
print(f'{BLUE}Press_mm_hg{RESET}: Presión atmosférica en milímetros de mercurio (mmHg).\n')
print(f'{BLUE}Windspeed{RESET}: Velocidad del viento en metros por segundo (m/s).\n')
print(f'{BLUE}Visibility{RESET}: Visibilidad atmosférica en kilómetros (km).\n')
columns_describe = [
    'Press_mm_hg',
    'Windspeed',
    'Visibility'
]
df_stats = df[columns_describe]
df_meteorologicos_pt2_describe = df_stats.describe(include="all")

# Calculat límites inferior y superior
lower_whisker_series = []
upper_whisker_series = []
skewness_series = []
kurtosis_series = []

for col in ['Press_mm_hg', 'Windspeed', 'Visibility']:
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
df_meteorologicos_pt2_describe.loc['Lower whisker'] = lower_whisker_series
df_meteorologicos_pt2_describe.loc['Upper whisker'] = upper_whisker_series
df_meteorologicos_pt2_describe.loc['Skewness'] = skewness_series
df_meteorologicos_pt2_describe.loc['Kurtosis'] = kurtosis_series

df_meteorologicos_pt2_describe = df_meteorologicos_pt2_describe.reindex(index={
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

df_meteorologicos_pt2_describe = df_meteorologicos_pt2_describe.rename(index={
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

print(tabulate(df_meteorologicos_pt2_describe, headers='keys', tablefmt='fancy_grid'))

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
for i, col in enumerate(['Press_mm_hg', 'Windspeed', 'Visibility']):
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
df_meteorologicos_pt2 = df[columns].copy()
df_meteorologicos_pt2['date'] = pd.to_datetime(df_meteorologicos_pt2['date'], dayfirst=True)

# Extraer la hora del día
df_meteorologicos_pt2['hour'] = df_meteorologicos_pt2['date'].dt.hour

# Create a 5x3 grid of subplots
fig = plt.figure(figsize=(20, 20))

# Define the grid layout
gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1])

# Row 1: Line chart of Press_mm_hg (span all columns)
ax1 = fig.add_subplot(gs[0, :])
# Row 2: Line charts of Windspeed and Visibility (span all columns)
ax2 = fig.add_subplot(gs[1, :])
# Row 3: Histograms
ax3 = fig.add_subplot(gs[2, 0])  # Press_mm_hg histogram
ax4 = fig.add_subplot(gs[2, 1])  # Windspeed histogram
ax5 = fig.add_subplot(gs[2, 2])  # Visibility histogram
# Row 4: Boxplots
ax6 = fig.add_subplot(gs[3, 0])    # Press_mm_hg boxplot
ax7 = fig.add_subplot(gs[3, 1])    # Windspeed boxplot
ax8 = fig.add_subplot(gs[3, 2])    # Visibility boxplot
# Row 5: Hourly patterns with error bars
ax9 = fig.add_subplot(gs[4, 0])    # Press_mm_hg hourly pattern
ax10 = fig.add_subplot(gs[4, 1])   # Windspeed hourly pattern
ax11 = fig.add_subplot(gs[4, 2])   # Visibility hourly pattern

# 1. Line chart of Press_mm_hg
ax1.plot(df_meteorologicos_pt2['date'], df_meteorologicos_pt2['Press_mm_hg'], label='Press_mm_hg (mmHg)', color='purple', linewidth=1)
ax1.set_title("Presión atmosférica a lo largo del tiempo")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Presión (mmHg)")
ax1.legend()
ax1.grid(True)

# 2. Line charts of Windspeed and Visibility
ax2_twin = ax2.twinx()  # Create second y-axis
ax2.plot(df_meteorologicos_pt2['date'], df_meteorologicos_pt2['Windspeed'], label='Windspeed (m/s)', color='orange', linewidth=1)
ax2_twin.plot(df_meteorologicos_pt2['date'], df_meteorologicos_pt2['Visibility'], label='Visibility (km)', color='cyan', linewidth=1)
ax2.set_title("Velocidad del viento y Visibilidad a lo largo del tiempo")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Velocidad del viento (m/s)", color='orange')
ax2_twin.set_ylabel("Visibilidad (km)", color='cyan')
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True)

# 3. Histogram for Press_mm_hg
mean_press = df_stats['Press_mm_hg'].mean()
median_press = df_stats['Press_mm_hg'].median()
std_press = df_stats['Press_mm_hg'].std()
n, bins, patches = ax3.hist(df_stats['Press_mm_hg'], bins=30, density=False, alpha=0.7, color='purple', edgecolor='black')

# Agregar líneas verticales para media y mediana
ax3.axvline(mean_press, color='blue', linestyle='--', linewidth=1, label=f'Media: {mean_press:.2f} mmHg')
ax3.axvline(median_press, color='green', linestyle='--', linewidth=1, label=f'Mediana: {median_press:.2f} mmHg')
ax3.legend()

ax3.set_title(f'Distribución de Press_mm_hg\n(μ={mean_press:.2f}, σ={std_press:.2f})')
ax3.set_xlabel('Presión atmosférica (mmHg)')
ax3.set_ylabel('Frecuencia')
ax3.grid(True, alpha=0.3)

# 4. Histogram for Windspeed
mean_wind = df_stats['Windspeed'].mean()
median_wind = df_stats['Windspeed'].median()
std_wind = df_stats['Windspeed'].std()
n, bins, patches = ax4.hist(df_stats['Windspeed'], bins=30, density=False, alpha=0.7, color='orange', edgecolor='black')

# Agregar líneas verticales para media y mediana
ax4.axvline(mean_wind, color='blue', linestyle='--', linewidth=1, label=f'Media: {mean_wind:.2f} m/s')
ax4.axvline(median_wind, color='red', linestyle='--', linewidth=1, label=f'Mediana: {median_wind:.2f} m/s')
ax4.legend()

ax4.set_title(f'Distribución de Windspeed\n(μ={mean_wind:.2f}, σ={std_wind:.2f})')
ax4.set_xlabel('Velocidad del viento (m/s)')
ax4.set_ylabel('Frecuencia')
ax4.grid(True, alpha=0.3)

# 5. Histogram for Visibility
mean_vis = df_stats['Visibility'].mean()
median_vis = df_stats['Visibility'].median()
std_vis = df_stats['Visibility'].std()
n, bins, patches = ax5.hist(df_stats['Visibility'], bins=30, density=False, alpha=0.7, color='cyan', edgecolor='black')

# Agregar líneas verticales para media y mediana
ax5.axvline(mean_vis, color='red', linestyle='--', linewidth=1, label=f'Media: {mean_vis:.2f} km')
ax5.axvline(median_vis, color='green', linestyle='--', linewidth=1, label=f'Mediana: {median_vis:.2f} km')
ax5.legend()

ax5.set_title(f'Distribución de Visibility\n(μ={mean_vis:.2f}, σ={std_vis:.2f})')
ax5.set_xlabel('Visibilidad (km)')
ax5.set_ylabel('Frecuencia')
ax5.grid(True, alpha=0.3)

# 6. Boxplot for Press_mm_hg
sns.boxplot(data=df_stats['Press_mm_hg'], ax=ax6, color='purple', orient="v")
ax6.set_title("Diagrama de cajas para Press_mm_hg")
ax6.set_ylabel("Presión atmosférica (mmHg)")

# 7. Boxplot for Windspeed
sns.boxplot(data=df_stats['Windspeed'], ax=ax7, color='orange', orient="v")
ax7.set_title("Diagrama de cajas para Windspeed")
ax7.set_ylabel("Velocidad del viento (m/s)")

# 8. Boxplot for Visibility
sns.boxplot(data=df_stats['Visibility'], ax=ax8, color='cyan', orient="v")
ax8.set_title("Diagrama de cajas para Visibility")
ax8.set_ylabel("Visibilidad (km)")

# 9. Hourly pattern for Press_mm_hg
hourly_press = df_meteorologicos_pt2.groupby('hour')['Press_mm_hg'].agg(['mean', 'std']).reset_index()
ax9.errorbar(hourly_press['hour'], hourly_press['mean'], yerr=hourly_press['std'], 
             fmt='-o', alpha=0.7, capsize=5, color='purple')
ax9.set_title('Press_mm_hg por Hora del Día (± desviación estándar)')
ax9.set_xlabel('Hora del Día')
ax9.set_ylabel('Press_mm_hg Promedio (mmHg)')
ax9.grid(True, alpha=0.3)
# Formatear el eje x como horas
ax9.set_xticks(range(0, 24, 2))  # Mostrar marcas cada 2 horas
ax9.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)  # Formato HH:00

# 10. Hourly pattern for Windspeed
hourly_wind = df_meteorologicos_pt2.groupby('hour')['Windspeed'].agg(['mean', 'std']).reset_index()
ax10.errorbar(hourly_wind['hour'], hourly_wind['mean'], yerr=hourly_wind['std'], 
              fmt='-o', alpha=0.7, capsize=5, color='orange')
ax10.set_title('Windspeed por Hora del Día (± desviación estándar)')
ax10.set_xlabel('Hora del Día')
ax10.set_ylabel('Windspeed Promedio (m/s)')
ax10.grid(True, alpha=0.3)
# Formatear el eje x como horas
ax10.set_xticks(range(0, 24, 2))  # Mostrar marcas cada 2 horas
ax10.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)  # Formato HH:00

# 11. Hourly pattern for Visibility
hourly_vis = df_meteorologicos_pt2.groupby('hour')['Visibility'].agg(['mean', 'std']).reset_index()
ax11.errorbar(hourly_vis['hour'], hourly_vis['mean'], yerr=hourly_vis['std'], 
              fmt='-o', alpha=0.7, capsize=5, color='cyan')
ax11.set_title('Visibility por Hora del Día (± desviación estándar)')
ax11.set_xlabel('Hora del Día')
ax11.set_ylabel('Visibility Promedio (km)')
ax11.grid(True, alpha=0.3)
# Formatear el eje x como horas
ax11.set_xticks(range(0, 24, 2))  # Mostrar marcas cada 2 horas
ax11.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)  # Formato HH:00

plt.tight_layout()
plt.subplots_adjust(wspace=0.229, hspace=0.995, bottom=0.088, top=0.967)
plt.show()