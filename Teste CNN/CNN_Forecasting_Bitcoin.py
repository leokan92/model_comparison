# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:19:07 2018

@author: leona
"""
import pandas as pd
import matplotlib.pylab as plt
import os
directory = os.getcwd()
os.chdir(directory)
from utils import *
from sklearn.metrics import *
from sklearn.metrics import mean_squared_error
from math import sqrt



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *

import seaborn as sns
sns.despine()

#from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred): 
    #, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean((y_true - y_pred) / y_true) * 100


# data_chng = data_original.ix[:, 'Adj Close'].pct_change().dropna().tolist()


STEP = 1
FORECAST = 1

#Number of historical data for INPUT
back_hist = 20
number_neurons = 100
drop_out_rate=0.5
num_epochs = 500
num_batches = 120
filter_num = 200
ajuste_base = 3
n_atraso = 10
file = 'BTCBRL_final.csv'
        
#*************************************************************************************
# Part 1 - Extraction and Data Preparation
#*************************************************************************************
directory = os.getcwd()
os.chdir(directory)
data = pd.read_csv(file,sep = ';')
dataset_total = pd.DataFrame(data)
dataset_total = dataset_total[20:int(round(len(dataset_total)/ajuste_base))]
dataset_total = dataset_total.reset_index()
dataset_total = dataset_total.drop(["index"],axis=1)
test_size = int(round(len(dataset_total.index)*0.3,0))
num_columns = len(dataset_total.columns)
dataset_train = dataset_total.iloc[test_size:,:]

#O -1 é devido à última coluna - os dados precisam estar no formato    
training_x = dataset_train.iloc[n_atraso:, 2:num_columns-1].values
training_y = dataset_train.iloc[:len(dataset_train)-n_atraso, num_columns-1:num_columns].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_y = sc.fit_transform(training_y)

num_columns = training_x.shape[1]
# Creating a data structure with 60 timesteps and 1 output
X_train = []   
y_train = []
for i in range(back_hist, len(training_x)):
	X_train.append(training_x[i-back_hist:i, 0: num_columns])
	y_train.append(training_y[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
    
    
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_columns))


#For Testing:

dataset_test = dataset_total.iloc[0:test_size+back_hist,:]
    
real_stock_price = dataset_test.iloc[:len(dataset_test)-back_hist-n_atraso, dataset_test.shape[1]-1:dataset_test.shape[1]].values
test_x = dataset_test.iloc[n_atraso:, 2:dataset_test.shape[1]-1].values
#subtrai um pois o último dado contem o preço
# Getting the predicted stock price of 2017
    
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - back_hist:].values
#inputs = inputs.reshape(-1, num_columns)
X_test = []
    
for i in range(back_hist, len(test_x)):
	X_test.append(test_x[i- back_hist:i, 0:num_columns])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_columns))


#============================================
#=============Model Creation=================
#============================================
model = Sequential()
model.add(Convolution1D(input_shape = (X_train.shape[1], num_columns),
                        nb_filter=number_neurons,
                        filter_length=filter_num,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(drop_out_rate))

model.add(Convolution1D(nb_filter=int(round(number_neurons/2)),
                        filter_length=int(round(filter_num*0.8)),
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(drop_out_rate))

model.add(Convolution1D(nb_filter=int(round(number_neurons/4)),
                        filter_length=int(round(filter_num*0.4)),
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(drop_out_rate))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())


model.add(Dense(1))

#model.add(Activation('softmax'))
#
#opt = Nadam(lr=0.002)
#
#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
#checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


model.compile(optimizer='adam', 
              loss='mean_squared_error',
              )

history = model.fit(X_train, y_train, 
          epochs = num_epochs, 
          batch_size = num_batches, 
          verbose=1, 
          validation_data=(X_test, real_stock_price),
          shuffle=True)

#model.load_weights("lolkek.hdf5")

#================================
#=====Previsão do Modelo=========
#================================

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

mae = mean_absolute_error(real_stock_price, predicted_stock_price)
mse = mean_squared_error(real_stock_price, predicted_stock_price)
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(real_stock_price,predicted_stock_price)
mpe = mean_percentage_error(real_stock_price,predicted_stock_price)

print("===========================================")
print("=               Resultados                =")
print("===========================================")
print("MAE: %.5f" %mae)
print("MSE: %.5f" %mse)
print("RMSE: %.5f" %rmse)
print("MAPE: %.5f" %mape)
print("MPE: %.5f" %mpe)

plt.plot(predicted_stock_price)
plt.plot(real_stock_price)
plt.savefig('Resultado1.png',dpi=200)
plt.show()


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())





