# Recurrent Neural Network - Using Cryptocompy API



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import os


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

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

#Number of historical data for INPUT
back_hist = 5
number_neurons = 600
drop_out_rate=0.3
num_epochs = 5
num_batches = 20
ajuste_base = 3
n_atraso = 1

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
test_size = 60

#int(round(len(dataset_total.index)*0.3,0))
num_columns = len(dataset_total.columns)
dataset_train = dataset_total.iloc[test_size:,:]

#O -1 é devido à última coluna - os dados precisam estar no formato    
training_x = dataset_train.iloc[n_atraso:, 2:num_columns-9].values
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
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    
#*************************************************************************************
# Part 2 - Building the RNN
#*************************************************************************************

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
    
# Initialising the RNN
regressor = Sequential()
    
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = number_neurons, return_sequences = True, input_shape = (X_train.shape[1], num_columns)))
regressor.add(Dropout(drop_out_rate))
    
# Adding a LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = number_neurons, return_sequences = True))
regressor.add(Dropout(drop_out_rate))
    
# Adding a LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = number_neurons, return_sequences = True))
regressor.add(Dropout(drop_out_rate))
    
# Adding a LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = number_neurons))
regressor.add(Dropout(drop_out_rate))
    
# Adding the output layer
regressor.add(Dense(units = 1))
        
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

     
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = num_epochs, batch_size = num_batches)
    
#Saving the Model
    
    
#*************************************************************************************
# Part 3 - Making the predictions and visualising the results
#*************************************************************************************
    
    
# Getting the Bitcoin price of 2017
#dataset_test = pd.read_csv('Bitcoin Price - Coindesk_Test.csv')
    
dataset_test = dataset_total.iloc[0:test_size+back_hist+n_atraso:]
    
real_stock_price = dataset_test.iloc[:len(dataset_test)-back_hist-n_atraso, dataset_test.shape[1]-1:dataset_test.shape[1]].values
test_x = dataset_test.iloc[n_atraso:, 2:dataset_test.shape[1]-1+n_atraso].values
#subtrai um pois o último dado contem o preço
# Getting the predicted stock price of 2017
    
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - back_hist:].values
#inputs = inputs.reshape(-1, num_columns)
X_test = []
    
for i in range(back_hist, len(test_x)):
	X_test.append(test_x[i- back_hist:i, 0:num_columns])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_columns))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price_train = sc.inverse_transform(training_y)
predicted_stock_price_train = regressor.predict(y_train)


#*************************************************************************************
# Part 4 - Results Visualization
#*************************************************************************************
   

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
plt.legend(['Previsto','Real'])
plt.savefig('Resultado1.png',dpi=200)
plt.show()
#previnir vazamento de memória da GPU



