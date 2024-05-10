import tensorflow as tf
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr

# Obtener datos
yf.pdr_override()

tickers = ['BTC-USD','ETH-USD']
start = '2023-05-08'
end = '2024-05-09'

btc_data = pdr.get_data_yahoo(tickers[0], start, end)['Close'].values.astype(float)
eth_data = pdr.get_data_yahoo(tickers[1], start, end)['Close'].values.astype(float)

# Escalar datos 
btc_min = np.min(btc_data) # obtenemos valores minimos
btc_max = np.max(btc_data) # y valores maximos
eth_min = np.min(eth_data)
eth_max = np.max(eth_data)

# formula de escalado min/max
btc_scaled = (btc_data - btc_min) / (btc_max - btc_min)
eth_scaled = (eth_data - eth_min) / (eth_max - eth_min)

bitcoin = eth_scaled[:-1]
ethereum = btc_scaled[1:]

# definir modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=30, input_shape=[1]), #! 30 y 60 el mejor tiempo hasta ahora
    tf.keras.layers.Dense(units=60),
    tf.keras.layers.Dense(units=1)
])

# compilar modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error') #adam learning rate 

# entrenar modelo
historial = modelo.fit(bitcoin, ethereum, epochs=1000, verbose=False) # historial guarda la perdida en cada epoca

# realizar predicción
result = modelo.predict([ethereum[-1]])

# desescalamos la predicción
predicted_btc = result * (btc_max - btc_min) + btc_min

print("Hagamos una predicción!")
print("El próximo precio de Bitcoin es aproximadamente " + str(predicted_btc[0][0]) + " USD!")

#print("Variables internas del modelo")
#print(oculta1.get_weights()) 
#print(oculta2.get_weights()) 
#print(salida.get_weights())
