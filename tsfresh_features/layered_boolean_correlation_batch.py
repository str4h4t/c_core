import pickle
import numpy as np
import pandas as pd
import correlation.DCCA.dcca_calc as dc
import scipy.stats as st

def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
data = np.asarray(data_set.iloc[:, 4:])
data = z_norm(data)
N = data_set.__len__()
macro_data = []
micro_data = []
mean_array = []
for i in range(0,N):
    m = data[i].mean()
    mean_array.append(m)
    macro_data.append(np.where(data[i] > m, 1, 0).astype('bool'))
    micro = []
    for t in range(0,data[i].__len__()):
        if t == 0:
            micro.append(1)
            continue
        if data[i][t] >= data[i][t-1]:
            micro.append(1)
        else:
            micro.append(0)
    micro = np.asarray(micro)
    micro_data.append(micro.astype('bool'))
macro_data = np.asarray(macro_data)
micro_data = np.asarray(micro_data)
correlation_complete = []
macro_pairs = []
micro_pairs = []
L = data.shape[1]
alpha = 0.80
beta = 0.80
gamma = 0.75
predicted_pairs_pearson = []
predicted_pairs_spearman = []
predicted_pairs_kendall = []
predicted_pairs_dcca = []
ctr_mi = 0
ctr_ma = 0
for i in range(0,N-1):
    print(i)
    for j in range(i+1, N):
        if data_set.iloc[i]['node'] == data_set.iloc[j]['node']:
            continue
        macro_corr = 1 - (macro_data[i] ^ macro_data[j]).sum()/L
        micro_corr = 1 - (micro_data[i] ^ micro_data[j]).sum() / L
        pair = data_set.iloc[i]['node'] + ';' + data_set.iloc[j]['node']
        correlation_complete.append({'pair': pair, 'macro_correlation': macro_corr,
                                     'micro_correlation': micro_corr, 'corr_counter': 'Dummy'})
        if macro_corr > alpha:
            ctr_ma += 1
            macro_pairs.append({'pair': pair, 'correlation': macro_corr, 'corr_counter': 'Dummy'})
            if micro_corr > beta:
                ctr_mi += 1
                micro_pairs.append({'pair': pair, 'correlation': micro_corr, 'corr_counter': 'Dummy'})
                df = pd.DataFrame([data[i],data[j]]).transpose()
                pearson = df.corr(method = 'pearson')[0][1]
                spearman = df.corr(method = 'spearman')[0][1]
                kendall = df.corr(method = 'kendall')[0][1]
                dcca = dc.executor(np.asarray(df).transpose(),6)[0][1]
                if pearson > gamma:
                    predicted_pairs_pearson.append({'pair': pair, 'correlation': pearson, 'corr_counter': 'Dummy'})
                if spearman > gamma:
                    predicted_pairs_spearman.append({'pair': pair, 'correlation': spearman, 'corr_counter': 'Dummy'})
                if kendall > gamma:
                    predicted_pairs_kendall.append({'pair': pair, 'correlation': kendall, 'corr_counter': 'Dummy'})
                if dcca > gamma:
                    predicted_pairs_dcca.append({'pair': pair, 'correlation': dcca, 'corr_counter': 'Dummy'})
print(ctr_ma)
print(ctr_mi)
pd.DataFrame(macro_pairs).to_csv('vodafone_raw_0.8_0.8_macro_data_pairs_port_lvl_11000_0921.csv')
pd.DataFrame(micro_pairs).to_csv('vodafone_raw_0.8_0.8_micro_data_pairs_port_lvl_11000_0921.csv')
pd.DataFrame(predicted_pairs_dcca).to_csv('vodafone_raw_0.8_0.8_0.8_dcca_port_lvl_11000_0921.csv')
pd.DataFrame(predicted_pairs_pearson).to_csv('vodafone_raw_0.8_0.8_0.8_pearson_port_lvl_11000_0921.csv')
pd.DataFrame(predicted_pairs_spearman).to_csv('vodafone_raw_0.8_0.8_0.8_spearman_port_lvl_11000_0921.csv')
pd.DataFrame(predicted_pairs_kendall).to_csv('vodafone_raw_0.8_0.8_0.8_kendall_port_lvl_11000_0921.csv')
print("done")
