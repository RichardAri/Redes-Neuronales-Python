import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from google.colab import drive

# montar Google Drive
drive.mount('/content/drive', force_remount=True)

# leer el archivo CSV
data = pd.read_csv('/content/drive/My Drive/practica_knn.csv')

# corregir los nombres de las columnas eliminando espacios iniciales
data.columns = data.columns.str.strip()

# separar caracteristicas (X) y etiquetas (y)
X = data.drop(['Country'], axis=1)
countries = data['Country']

# normalizar las caracteristicas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# aplicar PCA para reducir la dimensionalidad (opcional, pero util para visualizacion)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# determinar el numero optimo de clusters
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# implementar el algoritmo K-Means con el numero optimo de clusters determinado
k_opt = 3  # Asumiendo que 3 es optimo basado en el metodo del codo
kmeans = KMeans(n_clusters=k_opt, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.labels_

# visualizar los clusters
plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
plt.title('Visualization of Country Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
for i, country in enumerate(countries):
    plt.annotate(country, (X_pca[i, 0], X_pca[i, 1]))
plt.show()
