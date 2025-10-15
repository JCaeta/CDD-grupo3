import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

df = pd.read_csv('dataset_cleaned.csv')

predictors = [
    'hour',
    'T_int_avg',
    'Press_mm_hg',
    'RH_int_avg',
    'Tdewpoint',
    'T_out'
]

df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour

print(df)

print('Columns')

for col in df.columns:
    pass
    print(col)


# Variables predictoras (todas menos 'date' y 'Appliances')
# X = df.drop(columns=['date', 'Appliances'])
X = df[predictors]
y = df['Appliances']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lasso = Lasso(alpha=0.05, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Predicciones
y_pred = lasso.predict(X_test_scaled)


# Evaluar
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")




# --- CLASIFICACIÓN ARTIFICIAL ---
# Definir un umbral: por ejemplo, valores por encima de la mediana son “altos consumos”
threshold = y.median()

y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)

precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real Appliances")
plt.ylabel("Predicted Appliances")
plt.title("Lasso Regression: Real vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

feature_importance = pd.Series(lasso.coef_, index=X.columns)
feature_importance = feature_importance.sort_values(key=abs, ascending=False)

plt.figure(figsize=(10,5))
feature_importance.plot(kind='bar')
plt.title("Lasso Coefficients (Feature Importance)")
plt.ylabel("Coefficient value")
plt.show()

