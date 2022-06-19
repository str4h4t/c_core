import pickle

import numba.cuda
import numpy as np
import pandas as pd
import numba
import cupy

import scipy.stats as st

def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)

@numba.cuda.jit()
def corr_calc(sub_data):
    correlation_pearson = sub_data.corr(method='pearson')
    correlation_spearman = sub_data.corr(method='spearman')
    correlation_kendall = sub_data.corr(method='kendall')
    return correlation_pearson, correlation_spearman, correlation_kendall

print("HELLO")
alpha = 0.80
beta = 0.80
gamma = 0.75
with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
data_set = data_set.reset_index()
data = data_set.iloc[:,4:].transpose()
N = data_set.__len__()
window_size = 3
L = data.shape[1]
l = 0
u = l+window_size
correlation_pearson = []
correlation_spearman = []
correlation_kendall = []
while u <= 10:#data.shape[0]:
    print(u)
    sub_data = data.iloc[l:u]
    c_p, c_s, c_k = corr_calc(sub_data)
    correlation_pearson.append(c_p)
    correlation_spearman.append(c_s)
    correlation_kendall.append(c_k)
    l += 3
    u += 3
print("done")