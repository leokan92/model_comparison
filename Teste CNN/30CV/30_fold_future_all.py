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
        test_size = 60
        dataset_total = pd.DataFrame(data)
        dataset_len = len(dataset_total)
        dataset_len_m_teste = dataset_len - test_size
        tx_desloc = int(round(((dataset_len_m_teste-janela_passado) - time_size)/nu_cv))
        data_final = dataset_total[n*tx_desloc+test_size:time_size+n*tx_desloc]
        dfs.extend([data_final])
    return dfs


#Cria o arquivo
arquivo = open('30CV_future_1.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Kernels,Number of Epochs Filter Size,Batche Size,Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [40]
drop_out_rate = [0.2]
number_neurons = [80]
num_epochs = [120]
filter_num = [25]
num_batches = [70]


#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 30
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
        
        from CNN_Forecasting_Bitcoin_Function_Form import CNN_Model   
        #Para previsão de +1
        n_atraso_i = 1
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = CNN_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i,filter_num = filter_num_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,filter_num_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()

#Cria o arquivo
arquivo = open('30CV_future_5.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Kernels,Number of Epochs Filter Size,Batche Size,Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [2]
drop_out_rate = [0.4]
number_neurons = [90]
num_epochs = [170]
filter_num = [8]
num_batches = [30]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 30
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
        
        from CNN_Forecasting_Bitcoin_Function_Form import CNN_Model   
        #Para previsão de +1
        n_atraso_i = 5
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = CNN_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i,filter_num = filter_num_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,filter_num_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()


#Cria o arquivo
arquivo = open('30CV_future_10.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Kernels,Number of Epochs Filter Size,Batche Size,Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [1]
drop_out_rate = [0.1]
number_neurons = [20]
num_epochs = [150]
filter_num = [8]
num_batches = [100]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 30
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
        
        from CNN_Forecasting_Bitcoin_Function_Form import CNN_Model   
        #Para previsão de +1
        n_atraso_i = 10
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = CNN_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i,filter_num = filter_num_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,filter_num_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()


#Cria o arquivo
arquivo = open('30CV_future_30.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Kernels,Number of Epochs Filter Size,Batche Size,Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [4]
drop_out_rate = [0.1]
number_neurons = [70]
num_epochs = [150]
filter_num = [40]
num_batches = [50]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 1

#Determinar o número de validações cruzados a serem realizas
n_cv = 30
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
        
        from CNN_Forecasting_Bitcoin_Function_Form import CNN_Model   
        #Para previsão de +1
        n_atraso_i = 30
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = CNN_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i,filter_num = filter_num_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,filter_num_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 1 dia")
        
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()