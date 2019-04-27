# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:15:30 2018

@author: leona
"""


#=============================================================

# função que tira a média dos valores no arquivo e salva em novo arquivo3
import os
import pandas as pd

def gera_medias(nu_cv,file_2,n_iterações):
    directory = os.getcwd()
    os.chdir(directory) 
    data = pd.read_csv(file_2,sep = ';', engine='python')
    dataset_total = pd.DataFrame(data)
    mean_dataset = []
    mean_dataset = pd.DataFrame(mean_dataset)
    file = open('media_5fold.csv', 'a')
    file.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs, Batche Size, Prediction Window\n')
    file.close()
    for i in range(0,n_iterações):
        dataset_temp = dataset_total.iloc[(nu_cv*4*i):(nu_cv)*4*(i+1),:]
        mean_dataset = mean_dataset.append(pd.DataFrame(dataset_temp.groupby(dataset_temp.index %4).mean()))
    mean_dataset.to_csv('media_5fold.csv',sep=',')    

file_2 = '100_ite_LSTM.csv'
n_cv = 5
n_iterações = 100    
gera_medias(n_cv,file_2,n_iterações)

#=============================================================

