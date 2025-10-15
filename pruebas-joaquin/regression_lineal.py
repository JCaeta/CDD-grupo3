import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

# === Cargar dataset ===
df = pd.read_csv('dataset_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour

# === Seleccionar predictores ===
predictors = [
    'lights',
    'T_out',
    'Press_mm_hg',
    'RH_out',
    'Windspeed',
    'Visibility',
    'Tdewpoint',
    'T_int_avg',
    'RH_int_avg',
    'hour',
]

# === Variables predictoras y objetivo ===
X = df[predictors]
y = df['Appliances']

# === Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Estandarización ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Modelo de Regresión Lineal ===
lr_model = LinearRegression(fit_intercept=True)
lr_model.fit(X_train_scaled, y_train)

# === Predicciones ===
y_pred = lr_model.predict(X_test_scaled)

# === Evaluación (regresión) ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# === Clasificación artificial ===
# threshold = y.median()
# print(f"Threshold: {threshold:.2f}")

# y_test_class = (y_test > threshold).astype(int)
# y_pred_class = (y_pred > threshold).astype(int)

# precision = precision_score(y_test_class, y_pred_class)
# recall = recall_score(y_test_class, y_pred_class)

# print(f"Precision: {precision:.3f}")
# print(f"Recall: {recall:.3f}")

# === Visualización: Real vs Predicho ===
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real Appliances")
plt.ylabel("Predicted Appliances")
plt.title("Linear Regression: Real vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# # === Importancia (coeficientes del modelo) ===
# feature_importance = pd.Series(lr_model.coef_, index=X.columns)
# feature_importance = feature_importance.sort_values(ascending=True)

# plt.figure(figsize=(8,6))
# feature_importance.plot(kind='barh', color='lightcoral')
# plt.title("Linear Regression Coefficients")
# plt.xlabel("Coefficient Value")
# plt.ylabel("Features")
# plt.tight_layout()
# plt.show()
