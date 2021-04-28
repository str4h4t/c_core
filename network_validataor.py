import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import pickle

booster = True

results = []
with open('Data//vodafone_data_oct30_cross_osid_mar_19_filtered_interpolated.pkl', 'rb') as f:
    data_set = pickle.load(f)
with open('kendall_complete_oct_30_raw_data_corr_matrix.pkl', 'rb') as f:
     cor_mat = pickle.load(f)
#cor_mat = np.load('cor_matrix_cross_osids_vodafone_z-_kendall.npy')


gt_mat = np.load('adj_gt_cross_osids_vodafone.npy')
gt_mat = gt_mat.astype(int)
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
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*(precision*recall)/(precision+recall)
    results.append({'threshold': threshold, 'tp': tp, 'fp': fp,'fn': fn,'tn': tn,'accuracy':accuracy,
                    'precision': precision, 'recall': recall, 'f1_score': f1_score})
pd.DataFrame(results).to_csv('kendall_cross_osid_mar19_filtered_raw_data_not_boosted_complete_result.csv')
print("done")
