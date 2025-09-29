import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
path = './dataset_final_simplificado.csv'
df = pd.read_csv(path)

start_len = len(df)
q1 = 0.35
q3 = 0.68


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(q1)
    Q3 = df[column].quantile(q3)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_lights(df, column):
    value = 25
    return df[(df[column] <= value)]

def remove_outliers_rh_out(df, column):
    value = 48
    return df[(df[column] >= value)]

# Apply for each column
for col in ['T_out','T_int_avg']:
    df = remove_outliers_iqr(df, col)

df = remove_outliers_lights(df, 'lights')
df = remove_outliers_rh_out(df, 'RH_out')

final_len = len(df)
outliers_removed = start_len - final_len
print(f"Outliers removed: {outliers_removed}")
print(f"Percentage of outliers removed: {outliers_removed/start_len*100:.2f}%")

# Save cleaned dataset
df.to_csv('./dataset_cleaned.csv', index=False)
print("Outliers removed. Cleaned dataset saved as dataset_cleaned.csv")

# ---------------------------
# Visualization (without outliers)
# ---------------------------
fig = plt.figure(figsize=(20, 20))
grid_cols = 2
grid_rows = 3
grid_height_ratios = [1] * grid_rows

gs = fig.add_gridspec(grid_rows, grid_cols, height_ratios=grid_height_ratios)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])

def plot_distribution(df, col, ax, color, title):
    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()
    
    Q1 = df[col].quantile(q1)
    Q3 = df[col].quantile(q3)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Histogram
    ax.hist(df[col], bins=30, alpha=0.7, edgecolor='black', color=color)
    ax.axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'Media: {mean:.2f}')
    ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Mediana: {median:.2f}')

    ax.set_title(f"{title} distribución (μ={mean:.2f}, σ={std:.2f})")
    ax.set_xlabel(title)
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot all three
plot_distribution(df, 'T_out', ax1, 'red', 'Temperatura exterior (°C)')
plot_distribution(df, 'T_int_avg', ax2, 'orange', 'Temperatura interior (promedio) (°C)')
plot_distribution(df, 'RH_out', ax3, 'blue', 'Humedad exterior (%)')
plot_distribution(df, 'lights', ax4, 'green', 'Luces (Wh)')
plot_distribution(df, 'Appliances', ax5, 'green', 'Consumo de electrodomésticos (Wh)')

plt.tight_layout()

left = 0.064
wspace = 0.179
hspace = 0.333
right = 0.971
top = 0.943
bottom = 0.067

plt.subplots_adjust(wspace=wspace, hspace=hspace, right=right, left=left, top=top, bottom=bottom)

plt.show()
