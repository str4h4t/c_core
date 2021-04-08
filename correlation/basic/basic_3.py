import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

def crosscorr(datax, datay, meth, lag=0):
    return datax.corr(datay.shift(lag), method = meth)

def pearson(data):
    return data

def spearman(data):
    return data

def kendall(data):
    return data

def executor(data, method, param, boost):
    if boost:
        window_size = param[0]
        correlation_matrix = np.zeros([data.shape[0],data.shape[0]])
        #np.fill_diagonal(correlation_matrix,99)
        pool = mp.Pool(mp.cpu_count())
        for i in range(0, data.shape[0]):
            #print(i)
            if (i != (data.shape[0] - 1)):
                t_1 = pd.Series(data[i])
                for j in range(i + 1, data.shape[0]):
                    t_2 = pd.Series(data[j])
                    func = partial(crosscorr, t_1, t_2, method)
                    result = pool.map(func, [lag for lag in range(-window_size, window_size)])
                    result = list(np.nan_to_num(result))
                    correlation_matrix[i,j] = max(max(result),abs(min(result)))
        correlation_matrix = correlation_matrix.transpose() + correlation_matrix
        pool.terminate()
        return correlation_matrix