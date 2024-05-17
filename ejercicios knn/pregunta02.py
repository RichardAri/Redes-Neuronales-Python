import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import drive

# montar Google Drive
drive.mount('/content/drive', force_remount=True)

# leer el archivo CSV
data = pd.read_csv('/content/drive/My Drive/practica_knn.csv')

# corregir los nombres de las columnas eliminando espacios iniciales
data.columns = data.columns.str.strip()

# separar caracteristicas (X) y etiquetas (y)
X = data[['WhiteMeat', 'Eggs', 'Milk', 'Fish', 'Cereals', 'Starch', 'Nuts', 'Fr&Veg']]
y = data['RedMeat']

# dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# normalizar las caracteristicas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# lista de valores de k para probar
k_values = range(1, 21)  # Probando k desde 1 hasta 20

# listas para almacenar los resultados
mse_values = []
r2_values = []

# evaluar el modelo con diferentes valores de k
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_values.append(mse)
    r2_values.append(r2)

# graficar los resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, mse_values, marker='o')
plt.title('MSE vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(k_values, r2_values, marker='o', color='red')
plt.title('R^2 Score vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('R^2 Score')
plt.show()

