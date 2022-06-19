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
data_set_opin = data_set.loc[data_set['pm'] == 'OPIN-OTS']
data_set_opout = data_set = data_set.loc[data_set['pm'] == 'OPOUT-OTS']
data_opin = np.asarray(data_set_opin.iloc[:, 4:])
data_opout = np.asarray(data_set_opout.iloc[:, 4:])
# data = z_norm(data)

N_opout = data_set_opout.__len__()
macro_data_opout = []
micro_data_opout = []
mean_array_opout = []
for i in range(0,N_opout):
    m = data_opout[i].mean()
    mean_array_opout.append(m)
    macro_data_opout.append(np.where(data_opout[i] > m, 1, 0).astype('bool'))
    micro_opout = []
    for t in range(0,data_opout[i].__len__()):
        if t == 0:
            micro_opout.append(1)
            continue
        if data_opout[i][t] >= data_opout[i][t-1]:
            micro_opout.append(1)
        else:
            micro_opout.append(0)
    micro_opout = np.asarray(micro_opout)
    micro_data_opout.append(micro_opout.astype('bool'))
macro_data_opout = np.asarray(macro_data_opout)
micro_data_opout = np.asarray(micro_data_opout)

N_opin = data_set_opin.__len__()
macro_data_opin = []
micro_data_opin = []
mean_array_opin = []
for i in range(0,N_opin):
    m = data_opin[i].mean()
    mean_array_opin.append(m)
    macro_data_opin.append(np.where(data_opin[i] > m, 1, 0).astype('bool'))
    micro_opin = []
    for t in range(0,data_opin[i].__len__()):
        if t == 0:
            micro_opin.append(1)
            continue
        if data_opin[i][t] >= data_opin[i][t-1]:
            micro_opin.append(1)
        else:
            micro_opin.append(0)
    micro_opin = np.asarray(micro_opin)
    micro_data_opin.append(micro_opin.astype('bool'))
macro_data_opin = np.asarray(macro_data_opin)
micro_data_opin = np.asarray(micro_data_opin)

#opin at first means each opin is compared with every other opout
# correlation_complete_opin = []
macro_pairs_opout = []
micro_pairs_opout = []
L = data_opout.shape[1]
alpha = 0.80
beta = 0.80
gamma = 0.75
predicted_pairs_opout_pearson = []
predicted_pairs_opout_spearman = []
predicted_pairs_opout_kendall = []
predicted_pairs_opout_dcca = []
ctr_mi = 0
ctr_ma = 0
for i in range(0,N_opout):
    print(i)
    for j in range(0, N_opin):
        if data_set_opout.iloc[i]['node'] == data_set_opin.iloc[j]['node']:
            continue
        macro_corr_opout = 1 - (macro_data_opout[i] ^ macro_data_opin[j]).sum()/L
        micro_corr_opout = 1 - (micro_data_opout[i] ^ micro_data_opin[j]).sum() / L
        pair = data_set_opout.iloc[i]['node'] + ';' + data_set_opin.iloc[j]['node']
        # correlation_complete_opin.append({'pair': pair, 'macro_correlation': macro_corr_opin,
        #                              'micro_correlation': micro_corr_opin, 'corr_counter': 'Dummy'})
        if macro_corr_opout > alpha:
            ctr_ma += 1
            macro_pairs_opout.append({'pair': pair, 'correlation': macro_corr_opout, 'corr_counter': 'Dummy'})
            if micro_corr_opout > beta:
                ctr_mi += 1
                micro_pairs_opout.append({'pair': pair, 'correlation': micro_corr_opout, 'corr_counter': 'Dummy'})
                df = pd.DataFrame([data_opout[i],data_opin[j]]).transpose()
                pearson = df.corr(method = 'pearson')[0][1]
                spearman = df.corr(method = 'spearman')[0][1]
                kendall = df.corr(method = 'kendall')[0][1]
                dcca = dc.executor(np.asarray(df).transpose(),6)[0][1]
                if pearson > gamma:
                    predicted_pairs_opout_pearson.append({'pair': pair, 'correlation': pearson, 'corr_counter': 'Dummy'})
                if spearman > gamma:
                    predicted_pairs_opout_spearman.append({'pair': pair, 'correlation': spearman, 'corr_counter': 'Dummy'})
                if kendall > gamma:
                    predicted_pairs_opout_kendall.append({'pair': pair, 'correlation': kendall, 'corr_counter': 'Dummy'})
                if dcca > gamma:
                    predicted_pairs_opout_dcca.append({'pair': pair, 'correlation': dcca, 'corr_counter': 'Dummy'})
print(ctr_ma)
print(ctr_mi)
pd.DataFrame(macro_pairs_opout).to_csv('vodafone_raw_0.8_0.8_macro_data_pairs_port_lvl_opout_0921.csv')
pd.DataFrame(micro_pairs_opout).to_csv('vodafone_raw_0.8_0.8_micro_data_pairs_port_lvl_opout_0921.csv')
pd.DataFrame(predicted_pairs_opout_dcca).to_csv('vodafone_raw_0.8_0.8_0.8_dcca_port_lvl_opout_0921.csv')
pd.DataFrame(predicted_pairs_opout_pearson).to_csv('vodafone_raw_0.8_0.8_0.8_pearson_port_lvl_opout_0921.csv')
pd.DataFrame(predicted_pairs_opout_spearman).to_csv('vodafone_raw_0.8_0.8_0.8_spearman_port_lvl_opout_0921.csv')
pd.DataFrame(predicted_pairs_opout_kendall).to_csv('vodafone_raw_0.8_0.8_0.8_kendall_port_lvl_opout_0921.csv')
print("done")

macro_pairs_opin = []
micro_pairs_opin = []
L = data_opin.shape[1]
alpha = 0.80
beta = 0.80
gamma = 0.75
predicted_pairs_opin_pearson = []
predicted_pairs_opin_spearman = []
predicted_pairs_opin_kendall = []
predicted_pairs_opin_dcca = []
ctr_mi = 0
ctr_ma = 0
for i in range(0,N_opin):
    print(i)
    for j in range(0, N_opout):
        if data_set_opin.iloc[i]['node'] == data_set_opout.iloc[j]['node']:
            continue
        macro_corr_opin = 1 - (macro_data_opin[i] ^ macro_data_opout[j]).sum()/L
        micro_corr_opin = 1 - (micro_data_opin[i] ^ micro_data_opout[j]).sum() / L
        pair = data_set_opin.iloc[i]['node'] + ';' + data_set_opout.iloc[j]['node']
        # correlation_complete_opin.append({'pair': pair, 'macro_correlation': macro_corr_opin,
        #                              'micro_correlation': micro_corr_opin, 'corr_counter': 'Dummy'})
        if macro_corr_opin > alpha:
            ctr_ma += 1
            macro_pairs_opin.append({'pair': pair, 'correlation': macro_corr_opin, 'corr_counter': 'Dummy'})
            if micro_corr_opin > beta:
                ctr_mi += 1
                micro_pairs_opin.append({'pair': pair, 'correlation': micro_corr_opin, 'corr_counter': 'Dummy'})
                df = pd.DataFrame([data_opin[i],data_opout[j]]).transpose()
                pearson = df.corr(method = 'pearson')[0][1]
                spearman = df.corr(method = 'spearman')[0][1]
                kendall = df.corr(method = 'kendall')[0][1]
                dcca = dc.executor(np.asarray(df).transpose(),6)[0][1]
                if pearson > gamma:
                    predicted_pairs_opin_pearson.append({'pair': pair, 'correlation': pearson, 'corr_counter': 'Dummy'})
                if spearman > gamma:
                    predicted_pairs_opin_spearman.append({'pair': pair, 'correlation': spearman, 'corr_counter': 'Dummy'})
                if kendall > gamma:
                    predicted_pairs_opin_kendall.append({'pair': pair, 'correlation': kendall, 'corr_counter': 'Dummy'})
                if dcca > gamma:
                    predicted_pairs_opin_dcca.append({'pair': pair, 'correlation': dcca, 'corr_counter': 'Dummy'})
print(ctr_ma)
print(ctr_mi)
pd.DataFrame(macro_pairs_opin).to_csv('vodafone_raw_0.8_0.8_macro_data_pairs_port_lvl_opin_0921.csv')
pd.DataFrame(micro_pairs_opin).to_csv('vodafone_raw_0.8_0.8_micro_data_pairs_port_lvl_opin_0921.csv')
pd.DataFrame(predicted_pairs_opin_dcca).to_csv('vodafone_raw_0.8_0.8_0.8_dcca_port_lvl_opin_0921.csv')
pd.DataFrame(predicted_pairs_opin_pearson).to_csv('vodafone_raw_0.8_0.8_0.8_pearson_port_lvl_opin_0921.csv')
pd.DataFrame(predicted_pairs_opin_spearman).to_csv('vodafone_raw_0.8_0.8_0.8_spearman_port_lvl_opin_0921.csv')
pd.DataFrame(predicted_pairs_opin_kendall).to_csv('vodafone_raw_0.8_0.8_0.8_kendall_port_lvl_opin_0921.csv')
print("done")
