
def LSTM_Model(data,back_hist = 10, drop_out_rate = 0.2, number_neurons= 600,num_epochs = 50,num_batches = 200,n_atraso = 1):
#    
#    import tensorflow as tf
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth=True
#    sess = tf.Session(config=config)
    
    
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
    import keras
    
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
    
    print("Janela: %.0f" %back_hist)
    print("Drouout: %.1f" %drop_out_rate)
    print("N de Neuronios: %.0f" %number_neurons)
    print("N de Epocas: %.0f" %num_epochs)
    print("N de Batches: %.0f" %num_batches)
    print("Dias de Previsao: %.0f" %n_atraso)  
    
    #*************************************************************************************
    # Part 1 - Extraction and Data Preparation
    #*************************************************************************************
    dataset_total = data
    test_size = 60
    dataset_total_sem_teste = dataset_total[test_size:len(dataset_total)]
    dataset_total_sem_teste = dataset_total_sem_teste.reset_index()
    dataset_total_sem_teste = dataset_total_sem_teste.drop(["index"],axis=1)
    
    #int(round(len(dataset_total.index)*0.3,0))
    num_columns = len(dataset_total_sem_teste.columns)
    dataset_train = dataset_total_sem_teste.iloc[:,:]
    
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
        
    dataset_test = dataset_total.iloc[:test_size+back_hist+n_atraso,:]
        
    real_stock_price = dataset_test.iloc[:len(dataset_test)-back_hist-n_atraso, dataset_test.shape[1]-9:dataset_test.shape[1]-8].values
    #preciso arrumar sempre que muda o formato da base.
    test_x = dataset_test.iloc[n_atraso:, 2:dataset_test.shape[1]-9].values
    X_test = []
        
    for i in range(back_hist, len(test_x)):
    	X_test.append(test_x[i- back_hist:i, 0:num_columns])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_columns))
    
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    #Making predictions for the training
    real_stock_price_train = dataset_train.iloc[:len(dataset_train)-back_hist-n_atraso, dataset_test.shape[1]-9:dataset_test.shape[1]-8].values
    predicted_stock_price_train = regressor.predict(X_train)
    predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)
    
    #*************************************************************************************
    # Part 4 - Results Visualization
    #*************************************************************************************
    mae_train = mean_absolute_error(real_stock_price_train, predicted_stock_price_train)
    mse_train = mean_squared_error(real_stock_price_train, predicted_stock_price_train)
    rmse_train = sqrt(mse_train)
    mape_train = mean_absolute_percentage_error(real_stock_price_train,predicted_stock_price_train)
    mpe_train = mean_percentage_error(real_stock_price_train,predicted_stock_price_train)       
    
    mae_test = mean_absolute_error(real_stock_price, predicted_stock_price)
    mse_test = mean_squared_error(real_stock_price, predicted_stock_price)
    rmse_test = sqrt(mse_test)
    mape_test = mean_absolute_percentage_error(real_stock_price,predicted_stock_price)
    mpe_test = mean_percentage_error(real_stock_price,predicted_stock_price)
    
    
    #Evitando o memory leak


    
    print("===========================================")
    print("=               Resultados                =")
    print("===========================================")
    print("MAE: %.5f" %mae_test)
    print("MSE: %.5f" %mse_test)
    print("RMSE: %.5f" %rmse_test)
    print("MAPE: %.5f" %mape_test)
    print("MPE: %.5f" %mpe_test)
    
    print("===========================================")
    print("=               Resultados                =")
    print("===========================================")
    print("MAE: %.5f" %mae_train)
    print("MSE: %.5f" %mse_train)
    print("RMSE: %.5f" %rmse_train)
    print("MAPE: %.5f" %mape_train)
    print("MPE: %.5f" %mpe_train)
    
    plt.plot(predicted_stock_price[::-1],'k')
    plt.plot(real_stock_price[::-1])
    plt.legend(['Previsto','Real'])
    plt.savefig('Resultado_test{}.png'.format(n_atraso),dpi=200)
    plt.show()
    
    plt.plot(predicted_stock_price_train[::-1],'k')
    plt.plot(real_stock_price_train[::-1],'b')
    plt.legend(['Previsto','Real'])
    plt.savefig('Resultado_train{}.png'.format(n_atraso),dpi=200)
    plt.show() 
    
#    from keras import backend as K
#    del regressor   
#    K.clear_session()
#    
    import tensorflow as tf
    if keras.backend.tensorflow_backend._SESSION:
        tf.reset_default_graph() 
        keras.backend.tensorflow_backend._SESSION.close()
        keras.backend.tensorflow_backend._SESSION = None
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    return  mae_test, mse_test, rmse_test, mape_test ,mpe_test, mae_train, mse_train, rmse_train, mape_train, mpe_train,

