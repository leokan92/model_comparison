
def CNN_Model(data,back_hist = 10, drop_out_rate = 0.2, number_neurons= 600,num_epochs = 50,filter_num = 10, num_batches = 200,n_atraso = 1):
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
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.recurrent import LSTM, GRU
    from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
    from keras.layers.wrappers import Bidirectional
    from keras import regularizers
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import LeakyReLU
    from keras.optimizers import RMSprop, Adam, SGD, Nadam
    #from keras.initializers import *
    
    import seaborn as sns
    sns.despine()
    
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
    
    # Generating test samples
       
    dataset_test = dataset_total.iloc[:test_size+back_hist+n_atraso,:]
        
    real_stock_price = dataset_test.iloc[:len(dataset_test)-back_hist-n_atraso, dataset_test.shape[1]-9:dataset_test.shape[1]-8].values
    #preciso arrumar sempre que muda o formato da base.
    test_x = dataset_test.iloc[n_atraso:, 2:dataset_test.shape[1]-9].values
    X_test = []
        
    for i in range(back_hist, len(test_x)):
    	X_test.append(test_x[i- back_hist:i, 0:num_columns])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_columns))
    # Part 2 - Building the RNN
    #*************************************************************************************
    
    # Importing the Keras libraries and packages
        
    model = Sequential()
    model.add(Convolution1D(input_shape = (X_train.shape[1], X_train.shape[2]),
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
        
    #Saving the Model
        
        
    #*************************************************************************************
    # Part 3 - Making the predictions and visualising the results
    #*************************************************************************************
        

    

    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    #Making predictions for the training
    real_stock_price_train = dataset_train.iloc[:len(dataset_train)-back_hist-n_atraso, dataset_test.shape[1]-9:dataset_test.shape[1]-8].values
    predicted_stock_price_train = model.predict(X_train)
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
    
    
#    plt.plot(predicted_stock_price)
#    plt.plot(real_stock_price)
#    plt.legend(['Previsto','Real'])
#    plt.savefig('Resultado_test.png',dpi=200)
#    plt.show()
#    
#    plt.plot(predicted_stock_price_train)
#    plt.plot(real_stock_price_train)
#    plt.legend(['Previsto','Real'])
#    plt.savefig('Resultado_train.png',dpi=200)
#    plt.show() 
    
#    from keras import backend as K
#    del regressor   
#    K.clear_session()
#    
    import tensorflow as tf
    import keras
    if keras.backend.tensorflow_backend._SESSION:
        tf.reset_default_graph() 
        keras.backend.tensorflow_backend._SESSION.close()
        keras.backend.tensorflow_backend._SESSION = None
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    return  mae_test, mse_test, rmse_test, mape_test ,mpe_test, mae_train, mse_train, rmse_train, mape_train, mpe_train,

