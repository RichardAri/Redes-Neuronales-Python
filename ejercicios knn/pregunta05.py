from google.colab import drive
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# montar Google Drive
drive.mount('/content/drive', force_remount=True)

# leer el archivo CSV
data = pd.read_csv('/content/drive/My Drive/practica_knn.csv')

# verificar los nombres de las columnas y corregir posibles espacios
print("Nombres de columnas originales:", data.columns.tolist())
data.columns = data.columns.str.strip()
print("Nombres de columnas corregidos:", data.columns.tolist())

# asumiendo que queremos categorizar el consumo de 'RedMeat' en alto o bajo
median_red_meat = data['RedMeat'].median()
data['RedMeatHigh'] = (data['RedMeat'] > median_red_meat).astype(int)  # 1 si alto, 0 si bajo

# preparar los datos para el modelo
X = data.drop(['Country', 'RedMeat', 'RedMeatHigh'], axis=1)
y = data['RedMeatHigh']

# dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# escalar las caracteristicas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# realizar predicciones
predictions = knn.predict(X_test_scaled)

# evaluar el modelo
print("Matriz de Confusión:")
print(confusion_matrix(y_test, predictions))
print("\nReporte de Clasificación:")
print(classification_report(y_test, predictions))

# aplicar PCA para visualizacion en dos dimensiones
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

# visualizacion usando matplotlib
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=predictions, cmap='viridis', alpha=0.7)
plt.title('PCA of Food Consumption Patterns (KNN Predictions)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Predicted Red Meat Consumption High (1) or Low (0)')
plt.show()

