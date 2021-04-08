# from dtw import *
# import cudf
from datetime import datetime
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd

class dtw_calc:
    distance_matrix = []

    def __init__(self, data, param, boost):
        self.data = data
        self.w_size = param[0]
        self.w_shape = param[1]
        self.boost = boost


    def calc_dtw(self, t_1,t_2, w_size, w_type):
        #cud = cudf.DataFrame.from_pandas(t_1)
        if w_type == "itakura":
            try:
                distance = dtw(np.asarray(t_1), np.asarray(t_2), keep_internals=True, window_type = w_type).normalizedDistance
            except:
                distance = 9999
        else:
            try:
                distance = dtw(np.asarray(t_1), np.asarray(t_2), keep_internals=True, window_type = w_type,
                           window_args = {'window_size':w_size}, step_pattern=rabinerJuangStepPattern(6, "c")).normalizedDistance
            except:
                distance = 9999
        return distance

    def loop(self, t_1, i, w_type, w_size, j):
        t_2 = pd.Series(self.data[j])
        distance = self.calc_dtw(t_1, t_2, w_size, w_type)
        self.distance_matrix[i, j] = distance
        return self.distance_matrix

    def executor(self):
        pool = mp.Pool(mp.cpu_count())
        self.distance_matrix = np.zeros([self.data.shape[0], self.data.shape[0]])
        result = np.zeros([self.data.shape[0], self.data.shape[0]])
        for i in range(0,self.data.shape[0]):
            if (i != (self.data.shape[0]-1)):
                t_1 = pd.Series(self.data[i])
                func = partial(self.loop, t_1, i, self.w_shape, self.w_size)
                d = pool.map(func, [j for j in range(i+1, self.data.shape[0])])
                result+=np.asarray(d).sum(axis=0)
                #print("hello")
        result = result.transpose() + result
        pool.terminate()
        return result