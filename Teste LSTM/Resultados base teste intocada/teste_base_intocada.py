# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:15:30 2018

@author: leona
"""
import random
import time



start = time.time()



def divide_base(nu_cv,file,janela_passado = 40,time_size = 600):
    for n in range (1,nu_cv+1):
        import os
        import pandas as pd
        directory = os.getcwd()
        os.chdir(directory)
        data = pd.read_csv(file,sep = ';')
        #Regra combinada entre os colegas do teste final (sem crosss-validation)
        dataset_total = pd.DataFrame(data)
        data_final = dataset_total[:time_size]
        dfs.extend([data_final])
    return dfs


#Cria o arquivo
arquivo = open('30CV_future_1.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs, Batche Size, Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [10]
drop_out_rate = [0.7]
number_neurons = [100]
num_epochs = [30]
num_batches = [20]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 1
dfs = []
#Declaração do arquivo a ser usado para treinar e testar o modelo
file = 'BTCBRL_final.csv'


for i in range (0,n_iterações):
    
    back_hist_i = random.choice (back_hist)
    drop_out_rate_i = random.choice (drop_out_rate)
    number_neurons_i = random.choice (number_neurons)
    num_epochs_i = random.choice (num_epochs)
    num_batches_i = random.choice (num_batches)
    
    
    
    
    for j in range (0,n_cv):
        #Utiliza o modelo com os hiperparâmetros para gerar as métricas de erro.
        
        from LSTM_Forecasting_Bitcoin_Function_Form_print import LSTM_Model   
        #Para previsão de +1
        n_atraso_i = 1
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = LSTM_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()

#Cria o arquivo
arquivo = open('30CV_future_5.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs, Batche Size, Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [203]
drop_out_rate = [0.1]
number_neurons = [700]
num_epochs = [85]
num_batches = [60]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 1
dfs = []
#Declaração do arquivo a ser usado para treinar e testar o modelo
file = 'BTCBRL_final.csv'


for i in range (0,n_iterações):
    
    back_hist_i = random.choice (back_hist)
    drop_out_rate_i = random.choice (drop_out_rate)
    number_neurons_i = random.choice (number_neurons)
    num_epochs_i = random.choice (num_epochs)
    num_batches_i = random.choice (num_batches)
    
    
    
    
    for j in range (0,n_cv):
        #Utiliza o modelo com os hiperparâmetros para gerar as métricas de erro.
        
        from LSTM_Forecasting_Bitcoin_Function_Form_print import LSTM_Model   
        #Para previsão de +1
        n_atraso_i = 5
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = LSTM_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()


#Cria o arquivo
arquivo = open('30CV_future_10.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs, Batche Size, Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [20]
drop_out_rate = [0.5]
number_neurons = [100]
num_epochs = [150]
num_batches = [20]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 1
dfs = []
#Declaração do arquivo a ser usado para treinar e testar o modelo
file = 'BTCBRL_final.csv'


for i in range (0,n_iterações):
    
    back_hist_i = random.choice (back_hist)
    drop_out_rate_i = random.choice (drop_out_rate)
    number_neurons_i = random.choice (number_neurons)
    num_epochs_i = random.choice (num_epochs)
    num_batches_i = random.choice (num_batches)
    
    
    
    
    for j in range (0,n_cv):
        #Utiliza o modelo com os hiperparâmetros para gerar as métricas de erro.
        
        from LSTM_Forecasting_Bitcoin_Function_Form_print import LSTM_Model   
        #Para previsão de +1
        n_atraso_i = 10
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = LSTM_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()


#Cria o arquivo
arquivo = open('30CV_future_30.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs, Batche Size, Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [1]
drop_out_rate = [0.4]
number_neurons = [400]
num_epochs = [100]
num_batches = [10]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 1
dfs = []
#Declaração do arquivo a ser usado para treinar e testar o modelo
file = 'BTCBRL_final.csv'


for i in range (0,n_iterações):
    
    back_hist_i = random.choice (back_hist)
    drop_out_rate_i = random.choice (drop_out_rate)
    number_neurons_i = random.choice (number_neurons)
    num_epochs_i = random.choice (num_epochs)
    num_batches_i = random.choice (num_batches)
    
    
    
    
    for j in range (0,n_cv):
        #Utiliza o modelo com os hiperparâmetros para gerar as métricas de erro.
        
        from LSTM_Forecasting_Bitcoin_Function_Form_print import LSTM_Model   
        #Para previsão de +1
        n_atraso_i = 30
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = LSTM_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()