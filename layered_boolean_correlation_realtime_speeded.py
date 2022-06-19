import pickle
import numpy as np
import pandas as pd
import correlation.DCCA.dcca_calc as dc
import numba

import scipy.stats as st

def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)


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
    correlation_pearson.append(sub_data.corr(method = 'pearson'))
    correlation_spearman.append(sub_data.corr(method='spearman'))
    correlation_kendall.append(sub_data.corr(method='kendall'))
    l += 3
    u += 3


file = open("vodafone_z_norm_0.8_0.8_0.75_realtime_pearson.pkl", "wb")
pickle.dump(predicted_pairs_pearson, file)
file.close()
file = open("vodafone_z_norm_0.8_0.8_0.75_realtime_spearman.pkl", "wb")
pickle.dump(predicted_pairs_spearman, file)
file.close()
file = open("vodafone_z_norm_0.8_0.8_0.75_realtime_kendall.pkl", "wb")
pickle.dump(predicted_pairs_kendall, file)
file.close()
file = open("vodafone_z_norm_0.8_0.8_0.75_realtime_dcca.pkl", "wb")
pickle.dump(predicted_pairs_dcca, file)
file.close()

print("done")
