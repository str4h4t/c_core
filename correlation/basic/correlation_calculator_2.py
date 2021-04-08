import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

def crosscorr(datax, datay, meth, lag=0):
    return datax.corr(datay.shift(lag), method = meth)


with open('/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_data_oct30_filtered_interpolated.pkl', 'rb') as f:
    data_set = pickle.load(f)
result = pd.DataFrame(columns=['series_1', 'series_2', 'path_s_1', 'path_s_2', 'pearson','spearman', 'kendall'])
#result = pd.DataFrame(columns=['series_1', 'series_2', 'path', 'pearson'])

#ctr = 0
pool = mp.Pool(mp.cpu_count())
#pool.map(howmany_within_range_rowonly, [row for row in data])
for i in range(0,data_set['z-score'].shape[0]):
    print(i)
    if (i != (data_set['z-score'].shape[0]-1)):
        t_1 = pd.Series(data_set['z-score'][i])
        for j in range(i+1,data_set['z-score'].shape[0]):
            #ctr += 1
            #print(ctr)
            #print("i = "+str(i)+" j = "+str(j))
            t_2 = pd.Series(data_set['z-score'][j])
            func = partial(crosscorr, t_1, t_2, 'pearson')
            rs_pearson = pool.map(func, [lag for lag in range(-4,4)])
            func = partial(crosscorr, t_1, t_2, 'spearman')
            rs_spearman = pool.map(func, [lag for lag in range(-4, 4)])
            func = partial(crosscorr, t_1, t_2, 'kendall')
            rs_kendall = pool.map(func, [lag for lag in range(-4, 4)])
            #rs_pearson = [crosscorr(t_1, t_2, 'pearson', lag) for lag in range(-5,5)]
            #rs_spearman = [crosscorr(t_1, t_2, 'spearman', lag) for lag in range(-5,5)]
            #rs_kendall = [crosscorr(t_1, t_2, 'kendall', lag) for lag in range(-5,5)]
            rs_pearson = list(np.nan_to_num(rs_pearson))
            rs_spearman = list(np.nan_to_num(rs_spearman))
            rs_kendall = list(np.nan_to_num(rs_kendall))

            result = result.append({'series_1': data_set['node'][i] + "_" + str(data_set['port'][i]),
                                    'series_2': data_set['node'][j] + "_" + str(data_set['port'][j]),
                                    'path_s_1': data_set['path'][i],
                                    'path_s_2': data_set['path'][j],
                                    'pearson': max(max(rs_pearson),abs(min(rs_pearson))),
                                    'spearman': max(max(rs_spearman),abs(min(rs_spearman))),
                                    'kendall': max(max(rs_kendall),abs(min(rs_kendall)))}, ignore_index = True)

result.to_csv('vodafone_oct_30_z_score_result_filtered_interpolated.csv')

print("done")