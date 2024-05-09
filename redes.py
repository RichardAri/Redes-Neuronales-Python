# Importar TensorFlow y NumPy
import tensorflow as tf
import numpy as np

# Datos de entrada y salida para la conversion de Celsius a Fahrenheit
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

# Definir las capas de la red neuronal
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])  # Capa oculta 1 con 3 unidades, espera una entrada de 1 dimensión
oculta2 = tf.keras.layers.Dense(units=3)  # Capa oculta 2 con 3 unidades
salida = tf.keras.layers.Dense(units=1)   # Capa de salida con 1 unidad

# Crear el modelo secuencial y agregar las capas definidas
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

# Compilar el modelo especificando el optimizador y la funcion de perdida
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenamiento del modelo con los datos de entrada y salida
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)  # Entrenamiento durante 1000 epocas

# Visualizacion del historial de perdida durante el entrenamiento
import matplotlib.pyplot as plt

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

# Realizar una predicción con el modelo entrenado (convertir 100 grados Celsius a Fahrenheit)
resultado = modelo.predict([100.0])

# Imprimir el resultado de la predicción
print("Hagamos una prediccion!")
print("El resultado es " + str(resultado) + " Fahrenheit!")

# Mostrar las variables internas (pesos) de las capas ocultas y de salida del modelo
print("Variables internas del modelo")
print(oculta1.get_weights()) 
print(oculta2.get_weights()) 
print(salida.get_weights())


#* RESULTADO
"""
El resultado es [[211.74742]] fahrenheit!
Variables internas del modelo
[array([[-0.6033124 ,  0.21874487, -0.57152843]], dtype=float32), array([-4.711251,  2.093381, -4.820127], dtype=float32)]
[array([[-0.53705776,  0.11381449,  1.030693  ],
       [-0.27494565,  0.6410036 , -0.03102221],
       [-1.4078707 , -0.09891377, -0.16513045]], dtype=float32), array([ 4.762473 ,  4.0995235, -4.69153  ], dtype=float32)]
[array([[ 1.1460562 ],
       [ 0.80103356],
       [-0.8812149 ]], dtype=float32), array([4.396808], dtype=float32)]
"""

#* Cambiar las epocas no tiene casi ningun efecto