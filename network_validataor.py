import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import pickle

booster = True
gt_mat = np.load('adj_gt_cross_osids_vodafone_port_level_0921_OPOUT_OTS.npy')
gt_mat = gt_mat.astype(int)
param = ["pearson", "spearman", "kendall", "dcca"]

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
data_set = data_set.loc[data_set['pm'] == 'OPOUT-OTS']
data_set = data_set.reset_index()
data_set['node'] = data_set['node'].str.replace('-', '_')
# with open('kendall_complete_oct_30_raw_data_corr_matrix.pkl', 'rb') as f:
#      cor_mat = pickle.load(f)
for p in param:
    #cor_mat = np.load('kshape_cor_matrix_cross_osids_vodafone_test_netrd_wt.npy')
    cor_mat = np.load('kshape_cor_matrix_cross_osids_port_level_0921_OPOUT_OTS_vodafone_'+ p +'.npy')
    results = []
    for threshold in np.arange(0.75, 1, 0.01):
        c_mat = np.where(cor_mat > threshold, 1, 0)
        for n in data_set['node'].unique():
            indices = data_set[data_set['node'] == n].index
            if indices.__len__() > 1:
                temp = c_mat[indices].sum(axis=0).reshape([1, cor_mat.shape[0]]).repeat(indices.__len__(), axis=0)
                temp = np.where(temp > 0, 1, 0)
                c_mat[indices] = temp
        print(threshold)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0,c_mat.__len__()):
            #print(i)
            for j in range(0,c_mat.__len__()):
                if gt_mat[i][j] == 0:
                    if c_mat[i][j] == 0:
                        tn+=1
                    if c_mat[i][j] == 1:
                        fp+=1
                if gt_mat[i][j] == 1:
                    if c_mat[i][j] == 0:
                        fn+=1
                    if c_mat[i][j] == 1:
                        tp+=1
        accuracy = (tp + tn)/(tp+fp+fn+tn)
        if tp+fp == 0:
            precision = 0
        else:
            precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2*(precision*recall)/(precision+recall)
        results.append({'threshold': threshold, 'tp': tp, 'fp': fp,'fn': fn,'tn': tn,'accuracy':accuracy,
                        'precision': precision, 'recall': recall, 'f1_score': f1_score})
    pd.DataFrame(results).to_csv(p + '_kshape_port_level_raw_result__OPOUT_OTS_0921.csv')
print("done")
