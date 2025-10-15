import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from xgboost import XGBRegressor

# === Cargar dataset ===
df = pd.read_csv('dataset_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour

print('Columns')

for col in df.columns:
    pass
    print(col)

# === Seleccionar predictores ===
predictors = [
    # 'date',
    # 'Appliances',
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# === Estandarización (opcional pero ayuda un poco) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Modelo XGBoost ===
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    # random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# === Predicciones ===
y_pred = xgb_model.predict(X_test_scaled)

# === Evaluación (regresión) ===
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# === Clasificación artificial ===
threshold = y.median()

y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)

precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# === Visualización: Real vs Predicho ===
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real Appliances")
plt.ylabel("Predicted Appliances")
plt.title("XGBoost Regression: Real vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# === Importancia de variables ===
feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=True)  # ascendente para que la más importante quede arriba

plt.figure(figsize=(8,6))
feature_importance.plot(kind='barh', color='skyblue')
plt.title("XGBoost Feature Importance", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.show()