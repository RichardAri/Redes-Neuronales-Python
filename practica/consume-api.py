#* ------------------------------------------ datos en crudo 
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

tickers = ['BTC-USD','ETH-USD']

start = datetime(2022,12,1)
end = datetime(2023,12,1)

data = pdr.get_data_yahoo(tickers, start, end)

data

#* ------------------------------------------ datos con grafico 

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

yf.pdr_override()

tickers = ['BTC-USD','ETH-USD']

start = '2022-12-01'
end = '2023-12-01'

data = pdr.get_data_yahoo(tickers, start, end)['Close']

data.plot(figsize=(10, 6))
plt.title('Precios históricos de Bitcoin y Ethereum')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre (USD)')
plt.grid(True)
plt.legend(tickers)
plt.show()

