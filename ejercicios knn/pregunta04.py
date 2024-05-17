import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive

# montar Google Drive
drive.mount('/content/drive', force_remount=True)

# leer el archivo CSV
data = pd.read_csv('/content/drive/My Drive/practica_knn.csv')

# corregir los nombres de las columnas eliminando espacios iniciales
data.columns = data.columns.str.strip()

numeric_data = data.select_dtypes(include=[np.number])  # Esto seleccionara solo las columnas con datos numericos

# analisis de correlacion
correlation_matrix = numeric_data.corr()  # Obtener la matriz de correlacion de todas las variables numericas

# visualizacion de la correlación de 'RedMeat' con otros alimentos
red_meat_correlation = correlation_matrix['RedMeat'].drop('RedMeat')  # Eliminar la correlación de 'RedMeat' consigo mismo
red_meat_correlation.plot(kind='bar', color='red')
plt.title('Correlation of Red Meat with Other Food Types')
plt.xlabel('Food Types')
plt.ylabel('Correlation Coefficient')
plt.show()

# visualizacion mas detallada con scatter plots para algunos alimentos seleccionados
selected_foods = ['WhiteMeat', 'Fish', 'Milk', 'Eggs']  # seleccionar algunos tipos de alimentos
for food in selected_foods:
    sns.scatterplot(x='RedMeat', y=food, data=numeric_data)
    plt.title(f'Relationship between Red Meat and {food}')
    plt.xlabel('Red Meat Consumption')
    plt.ylabel(f'{food} Consumption')
    plt.show()
