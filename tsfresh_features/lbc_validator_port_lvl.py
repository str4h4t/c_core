import numpy as np
import pandas as pd

import pandas as pd
import glob
from tqdm import tqdm
import pickle

# topo_files = glob.glob('Data//topo_with_different_osids_updated//df_topo_*.csv')
#
# list_topo_files = []
# for file in tqdm(topo_files):
#     df = pd.read_csv(file, usecols=['tid_scrambled', 'lim', 'neighbor1_final', 'neighbor1-lim'])
#     lim_1 = df["lim"].str.split('-', expand=True)
#     lim_2 = df["neighbor1-lim"].str.split('-', expand=True)
#     neighbour = df.neighbor1_final.str.split('-', expand=True)
#     neighbour = neighbour[1] + '-' + neighbour[2] + '-' + neighbour[3]
#
#     col_1 = df['tid_scrambled'] + '_' + lim_1[1] + '_' + lim_1[2]
#     col_2 = neighbour + '_' + lim_2[1] + '_' + lim_2[2]
#     df = pd.DataFrame({'node_1': col_1, 'node_2': col_2})
#     df = pd.DataFrame({'node_1': df.min(axis=1), 'node_2': df.max(axis=1)}).drop_duplicates(ignore_index=True)
#
#     list_topo_files.append(df)
# df = pd.concat(list_topo_files, ignore_index=True).drop_duplicates(ignore_index=True).sort_values(by='node_1', ignore_index=True)
# # topo_nodes are all the nodes in the topo files
# # nh_nodes are the nodes from the interpolated filtered nh files
# topo_nodes = pd.concat([df.node_1, df.node_2], axis=0,ignore_index=True).drop_duplicates().to_numpy()
#nh_nodes = pd.read_pickle('pmvalues_interpolated_ahmed_mar_20.pkl').columns.to_series().str.split('_').apply(lambda x: x[0]+'_'+x[1]+'_'+x[2]).drop_duplicates().to_numpy()
with open('topo_connections_port_lvl_0921.pkl', 'rb') as f:
    df = pickle.load(f)
total_positives = 0

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    tsd = pickle.load(f)
nh_nodes = tsd['node'].unique()

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set_1 = pickle.load(f)
with open('pmvalues_interpolated_complete_vodafone_11000.pkl', 'rb') as f:
    data_set_2 = pickle.load(f)

data_set_2=data_set_2.transpose()
data_2 = np.asarray(data_set_2)
data_2 = data_2[:3027]
#data = z_norm(data)
nodes = []
for index in data_set_1.index:
    nodes.append("_".join(index.split('_')[:4]))
ctr = 0
for index in data_set_2.index:
    ctr += 1
    if ctr < 3028:
        nodes.append("_".join(index.split('_')[:4]))
nodes = pd.DataFrame(nodes, columns = {'node'})
nh_nodes = nodes['node'].unique()
df_combined = []
for row in df.iterrows():
    #if row[1]['node1'].split('_')[0] == row[1]['node2'].split('_')[0]:
        df_combined.append('_'.join([row[1]['node_1'],row[1]['node_2']]))
        if (row[1]['node_1'] in nh_nodes) and (row[1]['node_2'] in nh_nodes):
            total_positives += 1
print(total_positives)
methods = ['0.8_pearson', '0.8_spearman', '0.8_kendall', '0.8_dcca', 'macro_data_pairs', 'micro_data_pairs']
for meth in methods:
    print(meth)
    results =[]
    for threshold in np.arange(0.75, 1, 0.01):
        pairs = pd.read_csv('vodafone_raw_0.8_0.8_' + meth + '_port_lvl_5000_0921.csv')
        #print('DCCA')
        TP = 0
        predicted = []
        for row in pairs.iterrows():
            nodes = row[1][1].split(';')
            nodes.sort()
            nodes = '_'.join(nodes)
            corr = row[1]['correlation']
            if nodes not in predicted and corr > threshold:
                predicted.append(nodes)
        for p in predicted:
            if p in df_combined:
                TP += 1
        FP = predicted.__len__() - TP
        FN = total_positives - TP
        if TP + FN == 0 or TP + FP == 0 or TP == 0:
            Precision = 1
            Recall = 1
            fscore = 2 * (Precision * Recall) / (Precision + Recall)
            results.append(
                {'threshold': threshold, 'tp': TP, 'fp': FP, 'fn': FN, 'precision': Precision, 'recall': Recall,
                 'f1_score': fscore})
            continue
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        fscore = 2 * (Precision * Recall) / (Precision + Recall)
        results.append({'threshold': threshold, 'tp': TP, 'fp': FP, 'fn': FN, 'precision': Precision, 'recall': Recall,
                        'f1_score': fscore})
    pd.DataFrame(results).to_csv('vodafone_raw_lbc_batch_result_port_lvl_0921' + meth + '.csv')

