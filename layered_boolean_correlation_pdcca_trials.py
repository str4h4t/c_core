import pickle
import numpy as np
import pandas as pd
# import correlation.DCCA.dcca_calc as dc
import correlation.DCCA.pdcca_calc as dc
import scipy.stats as st

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
macro_data = np.where(data > mean_array, True, False)
macro_data = np.asarray(macro_data)
t_minus_1 = np.delete(np.concatenate((np.zeros((N,1)), data), axis=1),-1,1)
micro_data = np.where(data > t_minus_1, True, False)
micro_data[:,0] = np.ones([N])
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
predicted_pairs_pdcca = []
predicted_pairs_pc = []
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
                dcca, pdcca, pc = dc.executor(np.asarray(df).transpose(),6)
                dcca = dcca[0][1]
                pdcca = pdcca[0][1]
                pc = pc[0][1]
                if pearson > gamma:
                    predicted_pairs_pearson.append({'pair': pair, 'correlation': pearson, 'corr_counter': 'Dummy'})
                if spearman > gamma:
                    predicted_pairs_spearman.append({'pair': pair, 'correlation': spearman, 'corr_counter': 'Dummy'})
                if kendall > gamma:
                    predicted_pairs_kendall.append({'pair': pair, 'correlation': kendall, 'corr_counter': 'Dummy'})
                if dcca > gamma:
                    predicted_pairs_dcca.append({'pair': pair, 'correlation': dcca, 'corr_counter': 'Dummy'})
                if pdcca > gamma:
                    predicted_pairs_pdcca.append({'pair': pair, 'correlation': pdcca, 'corr_counter': 'Dummy'})
                if pc > gamma:
                    predicted_pairs_pc.append({'pair': pair, 'correlation': pc, 'corr_counter': 'Dummy'})
print(ctr_ma)
print(ctr_mi)
pd.DataFrame(macro_pairs).to_csv('vodafone_raw_0.8_0.8_macro_data_pairs_port_lvl_0921.csv')
pd.DataFrame(micro_pairs).to_csv('vodafone_raw_0.8_0.8_micro_data_pairs_port_lvl_0921.csv')
pd.DataFrame(predicted_pairs_dcca).to_csv('vodafone_raw_0.8_0.8_0.8_dcca_port_lvl_0921.csv')
pd.DataFrame(predicted_pairs_pdcca).to_csv('vodafone_raw_0.8_0.8_0.8_pdcca_port_lvl_0921.csv')
pd.DataFrame(predicted_pairs_pc).to_csv('vodafone_raw_0.8_0.8_0.8_pc_port_lvl_0921.csv')
pd.DataFrame(predicted_pairs_pearson).to_csv('vodafone_raw_0.8_0.8_0.8_pearson_port_lvl_0921.csv')
pd.DataFrame(predicted_pairs_spearman).to_csv('vodafone_raw_0.8_0.8_0.8_spearman_port_lvl_0921.csv')
pd.DataFrame(predicted_pairs_kendall).to_csv('vodafone_raw_0.8_0.8_0.8_kendall_port_lvl_0921.csv')
print("done")
