import numpy as np
from datetime import timezone
from tick.dataset import fetch_hawkes_bund_data
from tick.hawkes import HawkesConditionalLaw
from tick.hawkes import HawkesExpKern as hexp
from tick.hawkes import HawkesSumGaussians as hgauss
from tick.plot import plot_hawkes_kernel_norms

# timestamps_list = fetch_hawkes_bund_data()
import pickle
import numpy as np
import pandas as pd
import correlation.DCCA.dcca_calc as dc
import scipy.stats as st
from MHP import MHP
def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
# data_set = data_set.loc[data_set['pm'] == 'OPOUT-OTS']
data = np.asarray(data_set.iloc[:, 3:])
# data = z_norm(data)
# data = data[:,50:]
N = data.shape[0]
D = data.shape[1]
mean_array= data.mean(axis=1)
mean_array = np.repeat(mean_array.reshape([N,1]),D,axis=1)
macro_data = np.where(data > mean_array, 1, 0)
macro_data = np.asarray(macro_data)
t_minus_1 = np.delete(np.concatenate((np.zeros((N,1)), data), axis=1),-1,1)
micro_data = np.where(data > t_minus_1, 1, 0)
micro_data[:,0] = np.ones([N])
timestamps = data_set.columns[3:]
ts = np.asarray([t.replace(tzinfo=timezone.utc).timestamp() for t in timestamps])
timestamps_list = np.asarray(ts[np.where(micro_data[0])])
#
ts = ((ts - ts.min())/(ts - ts.min()).sum())*1000

P = MHP()
P.generate_seq(60)

m = np.array([0.2, 0.0, 0.0, 0.0])
a = np.array([[0.1, 0.0, 0.0, 0.0],
              [0.9, 0.0, 0.0, 0.0],
              [0.0, 0.9, 0.0, 0.0],
              [0.0, 0.1, 0.8, 0.0]])
w = 3.1

P = MHP(mu=m, alpha=a, omega=w)
# P.generate_seq(60)
# P.plot_events()
# mhat = np.random.uniform(0,1, size=4)
# ahat = np.random.uniform(0,1, size=(4,4))
# w = 3.

# x = P.EM(ahat, mhat, w)
new_micro_dat = []
ctr = 0
for i in micro_data:
    ctr+=1
    dat = []
    for j in i:
        dat.append(np.asarray(j))
    # dat =np.asarray(dat)
    df = np.asarray(pd.DataFrame({0: ts, 1: dat}))
    mhat = np.random.uniform(0, 1, size=2)
    ahat = np.random.uniform(0, 1, size=(2, 2))
    w = 3.
    P.data = df
    temp = P.EM(ahat, mhat, w)
    new_micro_dat.append([temp[1][0],temp[1][1]])
    print(ctr)
new_micro_dat = np.asarray(new_micro_dat)
gamma = 0.75
predicted_pairs_pearson = []
predicted_pairs_spearman = []
predicted_pairs_kendall = []
predicted_pairs_dcca = []
for i in range(0,N-1):
    print(i)
    for j in range(i+1, N):
        pair = data_set.iloc[i]['node'] + ';' + data_set.iloc[j]['node']
        df = pd.DataFrame([new_micro_dat[i], new_micro_dat[j]]).transpose()
        pearson = df.corr(method='pearson')[0][1]
        spearman = df.corr(method='spearman')[0][1]
        kendall = df.corr(method='kendall')[0][1]
        dcca = dc.executor(np.asarray(df).transpose(), 6)[0][1]
        if pearson > gamma:
            predicted_pairs_pearson.append({'pair': pair, 'correlation': pearson, 'corr_counter': 'Dummy'})
        if spearman > gamma:
            predicted_pairs_spearman.append({'pair': pair, 'correlation': spearman, 'corr_counter': 'Dummy'})
        if kendall > gamma:
            predicted_pairs_kendall.append({'pair': pair, 'correlation': kendall, 'corr_counter': 'Dummy'})
        if dcca > gamma:
            predicted_pairs_dcca.append({'pair': pair, 'correlation': dcca, 'corr_counter': 'Dummy'})
pd.DataFrame(predicted_pairs_dcca).to_csv('vodafone_raw_0.8_0.8_0.8_dcca_port_lvl_hawkes_micro_0921.csv')
pd.DataFrame(predicted_pairs_pearson).to_csv('vodafone_raw_0.8_0.8_0.8_pearson_port_lvl_hawkes_micro_0921.csv')
pd.DataFrame(predicted_pairs_spearman).to_csv('vodafone_raw_0.8_0.8_0.8_spearman_port_lvl_hawkes_micro_0921.csv')
pd.DataFrame(predicted_pairs_kendall).to_csv('vodafone_raw_0.8_0.8_0.8_kendall_port_lvl_hawkes_micro_0921.csv')

new_macro_dat = []
ctr = 0
for i in macro_data:
    ctr+=1
    dat = []
    for j in i:
        dat.append(np.asarray(j))
    # dat =np.asarray(dat)
    df = np.asarray(pd.DataFrame({0: ts, 1: dat}))
    mhat = np.random.uniform(0, 1, size=2)
    ahat = np.random.uniform(0, 1, size=(2, 2))
    w = 3.
    P.data = df
    temp = P.EM(ahat, mhat, w)
    new_macro_dat.append([temp[1][0],temp[1][1]])
    print(ctr)
new_macro_dat = np.asarray(new_macro_dat)
gamma = 0.75
predicted_pairs_pearson = []
predicted_pairs_spearman = []
predicted_pairs_kendall = []
predicted_pairs_dcca = []
for i in range(0,N-1):
    print(i)
    for j in range(i+1, N):
        pair = data_set.iloc[i]['node'] + ';' + data_set.iloc[j]['node']
        df = pd.DataFrame([new_macro_dat[i], new_macro_dat[j]]).transpose()
        pearson = df.corr(method='pearson')[0][1]
        spearman = df.corr(method='spearman')[0][1]
        kendall = df.corr(method='kendall')[0][1]
        dcca = dc.executor(np.asarray(df).transpose(), 6)[0][1]
        if pearson > gamma:
            predicted_pairs_pearson.append({'pair': pair, 'correlation': pearson, 'corr_counter': 'Dummy'})
        if spearman > gamma:
            predicted_pairs_spearman.append({'pair': pair, 'correlation': spearman, 'corr_counter': 'Dummy'})
        if kendall > gamma:
            predicted_pairs_kendall.append({'pair': pair, 'correlation': kendall, 'corr_counter': 'Dummy'})
        if dcca > gamma:
            predicted_pairs_dcca.append({'pair': pair, 'correlation': dcca, 'corr_counter': 'Dummy'})
pd.DataFrame(predicted_pairs_dcca).to_csv('vodafone_raw_0.8_0.8_0.8_dcca_port_lvl_hawkes_macro_0921.csv')
pd.DataFrame(predicted_pairs_pearson).to_csv('vodafone_raw_0.8_0.8_0.8_pearson_port_lvl_hawkes_macro_0921.csv')
pd.DataFrame(predicted_pairs_spearman).to_csv('vodafone_raw_0.8_0.8_0.8_spearman_port_lvl_hawkes_macro_0921.csv')
pd.DataFrame(predicted_pairs_kendall).to_csv('vodafone_raw_0.8_0.8_0.8_kendall_port_lvl_hawkes_macro_0921.csv')
print("hello")