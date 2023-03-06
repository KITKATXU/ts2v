import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio


move_dataFile='.\moves_13s.mat'
load_dataFile='.\loads_13s.mat'
move_data=scio.loadmat(move_dataFile)
load_data=scio.loadmat(load_dataFile)

moves_train_dict = {}
moves_test_dict = {}
loads_train_dict = {}
loads_test_dict = {}
for i in range(1,14):
    moves_train_dict['moves_'+str(i)] = move_data['moves_'+str(i)][:int(len(move_data['moves_'+str(i)])/10*7)]
    moves_test_dict['moves_' + str(i)] = move_data['moves_' + str(i)][:int(len(move_data['moves_' + str(i)]) / 10 * 3)]
    loads_train_dict['loads_' + str(i)] = load_data['loads_' + str(i)][:int(len(load_data['loads_' + str(i)]) / 10 * 7)]
    loads_test_dict['loads_' + str(i)] = load_data['loads_' + str(i)][:int(len(load_data['loads_' + str(i)]) / 10 * 3)]

save_path_moves_train = '.\moves_train.mat'
save_path_moves_test = '.\moves_test.mat'
save_path_loads_train = '.\loads_train.mat'
save_path_loads_test = '.\loads_test.mat'
scio.savemat(save_path_moves_train,moves_train_dict)
scio.savemat(save_path_moves_test,moves_test_dict)
scio.savemat(save_path_loads_train,loads_train_dict)
scio.savemat(save_path_loads_test,loads_test_dict)










# scio.savemat(save_path_train,{'moves_1':data['moves_1'][:int(len(data['moves_1'])/10*7)],\
#                         'moves_2':data['moves_2'][:int(len(data['moves_2'])/10*7)],\
#                         'moves_3':data['moves_3'][:int(len(data['moves_3'])/10*7)],\
#                         'moves_4':data['moves_4'][:int(len(data['moves_4'])/10*7)],\
#                         'moves_5':data['moves_5'][:int(len(data['moves_5'])/10*7)],\
#                         'moves_6':data['moves_6'][:int(len(data['moves_6'])/10*7)],\
#                         'moves_7':data['moves_7'][:int(len(data['moves_7'])/10*7)],\
#                         'moves_8':data['moves_8'][:int(len(data['moves_8'])/10*7)],\
#                         'moves_9': data['moves_9'][:int(len(data['moves_9'])/10*7)],\
#                         'moves_10': data['moves_10'][:int(len(data['moves_10'])/10*7)],\
#                         'moves_11': data['moves_11'][:int(len(data['moves_11'])/10*7)],\
#                         'moves_12': data['moves_12'][:int(len(data['moves_12'])/10*7)],\
#                         'moves_13': data['moves_13'][:int(len(data['moves_13'])/10*7)],\
#                                                 })
