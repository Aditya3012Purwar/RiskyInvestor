from keras.models import Sequential
from keras.layers import Dense,LSTM,BatchNormalization,Flatten
from matplotlib import pyplot as plt
import tensorflow as tf
import requests
import numpy as np
import yfinance
df = yfinance.download('AAPL','2020-1-1','2021-01-27')
close = df['Close']
print(close)
plt.plot(close)
def SMA(prices,value):
    means = []
    count = 0
    while value+count <= len(prices):
        pre_val = prices[count:value+count]
        count +=1
        means.append(np.mean(pre_val))
    return means
sma_1 = SMA(close,20)
sma_2 = SMA(close,4)
sma_1 = np.array(sma_1).reshape(len(sma_1),1)
sma_2 = np.array(sma_2).reshape(len(sma_2),1)
sma_2 = sma_2[:-20]
print(len(sma_1),len(sma_2))
plt.plot(sma_1)
plt.plot(sma_2)
smas = np.concatenate((sma_1,sma_2))
smas.shape
X = np.array(smas)
y = np.array(close[:-19])
print(X.shape,y.shape)
model = Sequential()
model.add(Dense(100,input_shape = (None,2)))
model.add(Dense(128))
model.add(Dense(1))
print(model.summary())
model.compile(optimizer = 'adam', loss = 'mse')
data_list = tf.stack(X)
y = tf.stack(y)
def estimate_profits(pred,y):
    profits = 0
    investment = 1000
    log = []
    for i in range(len(y)-1):
        if pred[i][0] > y[i]:
            trade = 'buy'
        elif pred[i][0] < y[i]:
            trade = 'sell'
        else:
            trade = None
        if y[i+1] > y[i]:
            ctrade = 'buy'
        elif y[i+1] < y[i]:
            ctrade = 'sell'
        else:
            ctrade = None
        if ctrade:
            if trade == ctrade:
                profits += 0.8*investment
                log.append(1)
            else:
                profits -= investment
                log.append(0)
    accuracy = round(log.count(1)/len(log)*100)
    return profits,accuracy
profit,acc = estimate_profits(X,y)
profit /= 100
print(acc,profit)
