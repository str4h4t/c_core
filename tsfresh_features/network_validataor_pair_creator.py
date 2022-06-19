import numpy as np
import pandas as pd
import pickle
import os

booster = True
gt_mat = np.load('adj_gt_cross_osids_vodafone_port_level_0921.npy')
gt_mat = gt_mat.astype(int)
param = ["pearson", "spearman", "kendall", "dcca"]

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
data_set = data_set.reset_index()
# data_set['node'] = data_set['node'].str.replace('-', '_')
# with open('kendall_complete_oct_30_raw_data_corr_matrix.pkl', 'rb') as f:
#      cor_mat = pickle.load(f)
f_ctr = 0
with open('topo_connections_port_lvl_0921.pkl', 'rb') as f:
    df = pickle.load(f)
nh_nodes = data_set['node'].unique()
total_positives = 0
df_combined = []
for row in df.iterrows():
    # if row[1]['node1'].split('_')[0] == row[1]['node2'].split('_')[0]:
        df_combined.append('_'.join([row[1]['node_1'],row[1]['node_2']]))
        if (row[1]['node_1'] in nh_nodes) and (row[1]['node_2'] in nh_nodes):
            total_positives += 1
print(total_positives)
for filename in os.listdir('C:\\Users\\kamal\\PycharmProjects\\c_core\\tsfresh_features\\correlation_matrices_2'):
    if filename.endswith('.npy'):
        cor_mat = np.load(os.path.join('C:\\Users\\kamal\\PycharmProjects\\c_core\\tsfresh_features\\correlation_matrices_2', filename))
    else:
        continue
    #cor_mat = np.load('kshape_cor_matrix_cross_osids_vodafone_test_netrd_wt.npy')
    f_ctr += 1
    print("file_no: " + str(f_ctr))
    results = []
    for threshold in np.arange(0.60, 1, 0.01):
        c_mat = np.where(cor_mat > threshold, 1, 0)
        for n in data_set['node'].unique():
            indices = data_set[data_set['node'] == n].index
            if indices.__len__() > 1:
                temp = c_mat[indices].sum(axis=0).reshape([1, cor_mat.shape[0]]).repeat(indices.__len__(), axis=0)
                temp = np.where(temp > 0, 1, 0)
                c_mat[indices] = temp
        print(threshold)
        TP = 0
        predicted = []
        for i in range(0,c_mat.__len__()):
            # print(i)
            for j in range(0,c_mat.__len__()):
                if data_set.iloc[i]['node'] == data_set.iloc[j]['node']:
                    continue
                if c_mat[i][j] == 1:
                    tmp = ('_').join((data_set.iloc[i]['node'],data_set.iloc[j]['node']))
                    if tmp not in predicted:
                        predicted.append(tmp)
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
        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        fscore = 2 * (Precision * Recall) / (Precision + Recall)
        results.append(
            {'threshold': threshold, 'tp': TP, 'fp': FP, 'fn': FN, 'precision': Precision, 'recall': Recall,
             'f1_score': fscore})
    out_filename = 'results_pairs_' + filename.split('.')[0] + '.csv'
    pd.DataFrame(results).to_csv(out_filename)
print("done")
