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

# entrenar el modelo KNN para regresion
k = 3  # Número de vecinos
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train, y_train)

# predecir las etiquetas para el conjunto de prueba
y_pred = knn.predict(X_test)

# evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# resultados
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Reales vs. Valores Predichos')
plt.show()
