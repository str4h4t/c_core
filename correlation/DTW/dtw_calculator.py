import pickle
import pandas as pd
import numpy as np
from dtw_calc import *
from numba import njit
import cudf
import timeit
from datetime import datetime



def calc_dtw(t_1,t_2, w_size, w_type):
    #cud = cudf.DataFrame.from_pandas(t_1)
    if w_type == "itakura":
        try:
            distance = dtw(np.asarray(t_1), np.asarray(t_2), keep_internals=True, window_type=w_type).normalizedDistance
        except:
            distance = 9999
    else:
        try:
            distance = dtw(np.asarray(t_1), np.asarray(t_2), keep_internals=True, window_type=w_type,
                           window_args={'window_size': w_size},
                           step_pattern=rabinerJuangStepPattern(6, "c")).normalizedDistance
        except:
            distance = 9999
    return distance

def calc(w_type, w_size):

    with open('/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_data_oct30_filtered_interpolated.pkl', 'rb') as f:
        data_set = pickle.load(f)
    result = pd.DataFrame(columns=['series_1', 'series_2', 'path_s_1', 'path_s_2', 'distance'])
    ctr = 0

    for i in range(0,data_set['z-score'].shape[0]):
        print(i)
        if (i != (data_set['z-score'].shape[0]-1)):
            t_1 = pd.Series(data_set['z-score'][i])
            print(datetime.now())
            for j in range(i+1,data_set['z-score'].shape[0]):
                ctr += 1
                #print(ctr)
                #print("i = "+str(i)+" j = "+str(j))
                t_2 = pd.Series(data_set['z-score'][j])
                # if w_type == "itakura":
                #     try:
                #         distance = dtw(np.asarray(t_1), np.asarray(t_2), keep_internals=True, window_type = w_type).normalizedDistance
                #         print(datetime.now())
                #     except:
                #         distance = 9999
                # else:
                #     try:
                #         distance = dtw(np.asarray(t_1), np.asarray(t_2), keep_internals=True, window_type = w_type,
                #                    window_args = {'window_size':w_size}, step_pattern=rabinerJuangStepPattern(6, "c")).normalizedDistance
                #         print(datetime.now())
                #     except:
                #         distance = 9999
                distance = calc_dtw(t_1,t_2,w_size,w_type)
                result = result.append({'series_1': data_set['node'][i] + "_" + str(data_set['port'][i]),
                                        'series_2': data_set['node'][j] + "_" + str(data_set['port'][j]),
                                        'path_s_1': data_set['path'][i],
                                        'path_s_2': data_set['path'][j],
                                        'distance': str(distance)}, ignore_index = True)
            print(datetime.now())
            print("-----------------------")
    cudata = cudf.DataFrame.from_pandas(result)
    result['distance'].replace('9999', int(result['distance'].astype(float).sort_values().unique()[-2]) + 1)
    result.to_csv('vodafone_oct_30_interpolated_filtered_z-score_result_dtw_' + w_type + '_' + str(w_size) + '_' + '.csv')
#int(result['distance'].astype(float).sort_values().unique()[-2]+1)
print("done")