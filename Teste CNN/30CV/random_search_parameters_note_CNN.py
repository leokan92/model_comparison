# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:15:30 2018

@author: leona
"""
import random
import time
from keras.initializers import *

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
        #Aqui com a soma do test_size ele garante que os útlimos dados não serão usados
        data_final = dataset_total[n*tx_desloc+test_size:time_size+n*tx_desloc]
        dfs.extend([data_final])
    return dfs


#Cria o arquivo
arquivo = open('20_ite_CNN_06.csv', 'a')
arquivo.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs Filter Size,Batche Size,Prediction Window\n')

#Determinação do range de hiper-parametros
back_hist = [1,2,3,4,5,10,15,20,30,40]
drop_out_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
number_neurons = [20,30,40,50,60,70,80,90,100,200,300,400]
num_epochs = [10,20,30,50,70,85,100,120,150,170,200,300,500,700,1000]
num_batches = [10,20,30,40,50,60,70,80,90,100]
filter_num = [5,6,7,8,9,10,15,20,25,30,35,40,50,60,70,80,90,100]

#Determinar o número de iterações para determinar os hiper-parâmetros
n_iterações = 20

#Determinar o número de validações cruzados a serem realizas
n_cv = 5
dfs = []
#Declaração do arquivo a ser usado para treinar e testar o modelo
file = 'BTCBRL_final.csv'


for i in range (0,n_iterações):
    
    back_hist_i = random.choice (back_hist)
    drop_out_rate_i = random.choice (drop_out_rate)
    number_neurons_i = random.choice (number_neurons)
    num_epochs_i = random.choice (num_epochs)
    num_batches_i = random.choice (num_batches)
    filter_num_i = random.choice (filter_num)
    
    
    
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
        
        #Para previsão de +5   
        n_atraso_i = 5
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = CNN_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i,filter_num = filter_num_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,filter_num_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 5 dia")
        
              
        #Para previsão de +10
        n_atraso_i = 10
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = CNN_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i,filter_num = filter_num_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,filter_num_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 10 dia")
        
                 
        #Para previsão de +30
        n_atraso_i = 30
        dfs = divide_base(n_cv,file,janela_passado = back_hist_i,time_size = 600 + n_atraso_i)
        a,b,c,d,e,f,g,h,i_1,j_1 = CNN_Model(back_hist = back_hist_i, drop_out_rate = drop_out_rate_i, number_neurons= number_neurons_i,num_epochs = num_epochs_i,filter_num = filter_num_i, num_batches = num_batches_i, n_atraso = n_atraso_i,data = dfs[j])
        arquivo.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,a, b, c, d, e,f,g,h,i_1,j_1,back_hist_i,drop_out_rate_i,number_neurons_i,num_epochs_i,filter_num_i,num_batches_i,n_atraso_i))
        print(time.time() - start)
        print("Iteração: %.0f" %i)
        print("Cross-validation: %.0f" %j)
        print("Previsão para 30 dia")
          
        
      
#Fecha o arquivo com todas as interações de hiperparâmetros feitas    
arquivo.close()



#=============================================================

## função que tira a média dos valores no arquivo e salva em novo arquivo3
#def gera_medias(nu_cv,file_2,n_iterações):
#    import os
#    import pandas as pd
#    directory = os.getcwd()
#    os.chdir(directory) 
#    data = pd.read_csv(file_2,sep = ';')
#    dataset_total = pd.DataFrame(data)
#    mean_dataset = []
#    mean_dataset = pd.DataFrame(mean_dataset)
#    file = open('resutlados_LSTM_CV.csv', 'a')
#    file.write('Hyper_id,MAE_test,MSE_test,RMSE_test,MAPE_test,MPE_test,MAE_train,MSE_train,RMSE_train,MAPE_train,MPE_train,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs, Batche Size, Prediction Window\n')
#    file.close()
#    for i in range(0,n_iterações):
#        dataset_temp = dataset_total.iloc[(nu_cv*4*i):(nu_cv)*4*(i+1),:]
#        mean_dataset = mean_dataset.append(pd.DataFrame(dataset_temp.groupby(dataset_temp.index %4).mean()))
#    mean_dataset.to_csv('resutlados_LSTM_CV.csv',sep=',')    
#
#file_2 = 'resutlados_LSTM.csv'
#gera_medias(n_cv,file_2,n_iterações)   

#=============================================================


#nu_cv = 2
#i = 1

#
#def gera_medias(nu_cv,file):
#    import os
#    import pandas as pd
#    directory = os.getcwd()
#    os.chdir(directory) 
#    data = pd.read_csv(file,sep = ';')
#    dataset_total = pd.DataFrame(data)
#    mean_dataset = []
#    mean_dataset = pd.DataFrame(mean_dataset)
#    for i in range(0,len(dataset_total),nu_cv):
#        mean_dataset = mean_dataset.append(pd.DataFrame(dataset_total.iloc[i:i+nu_cv,:].mean(axis=0)).transpose())
#    file = open('resutlados_LSTM_CV.csv', 'a')
#    file.write('Hyper_id,MAE,MSE,RMSE,MAPE,MPE,Window,Dropout Rate,Number of Neurons (each layer),Number of Epochs, Batche Size, Prediction Window\n')
#    file.close()
#    mean_dataset.to_csv('resutlados_LSTM_CV.csv',sep=',')
#**************************************
#Lixo
#**************************************
#Função para dividir a base 
#def divide_base(nu_cv,file):
#    for factor in range (nu_cv):
#        import os
#        import pandas as pd
#        directory = os.getcwd()
#        os.chdir(directory)
#        data = pd.read_csv(file,sep = ';')
#        dataset_total = pd.DataFrame(data)
#        dataset_total = dataset_total[int(round(len(dataset_total)/nu_cv))*factor:int(round(len(dataset_total)/nu_cv))*(factor+1)]
#        dfs.extend([dataset_total])
#    return dfs


#factor = 0
#dfs = []
#n_cv= 5
#dfs = divide_base(n_cv,'BTCBRL_final (1).csv',janela_passado = 40)