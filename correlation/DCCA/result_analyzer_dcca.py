import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle

with open('/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_data_oct30_filtered_interpolated.pkl', 'rb') as f:
    data_set = pickle.load(f)
data = np.zeros(data_set['z-score'][0].__len__())
for d in range(0, data_set['z-score'].__len__()):
    data = np.vstack((data, data_set['z-score'][d]))

data = np.delete(data, 0, axis=0)
data_set = data_set[~np.isnan(data).any(axis=1)]
data_set.index = range(0, data_set.__len__())

result = np.genfromtxt('vodafone_complete_oct_30_dcca_normalized.csv', delimiter=',')
def result_maker(threshold, data_set = data_set, correlation = result):
    print(threshold)
    metrics = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F_score'])
    tp_ctr = 0
    
    tn_ctr = 0
    fp_ctr = 0
    fn_ctr = 0
    for i in range(0,data_set.__len__()-1):
        for j in range(i+1,data_set.__len__()):
            #print("i: " + str(i) + "  j: " + str(j))
            o1 = data_set['path'][i]
            o2 = data_set['path'][j]
            if o1 == o2:
                if abs(correlation[i][j]) >= threshold:
                    tp_ctr += 1
                else:
                    fn_ctr += 1
            else:
                if abs(correlation[i][j]) < threshold:
                    tn_ctr += 1
                else:
                    fp_ctr += 1
    precision = tp_ctr / (tp_ctr + fp_ctr)
    recall = tp_ctr / (tp_ctr + fn_ctr)
    f_score = 2 * ((precision * recall) / (precision + recall))
    metrics = metrics.append(
        {'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F_score': f_score, }, ignore_index=True)
    return metrics
if __name__ == "__main__":
    threshold = np.arange(0.75, 1, 0.01)
    pool = mp.Pool(mp.cpu_count())
    metrics = pool.map(result_maker, [i for i in threshold])
    df = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F_score'])
    for frame in metrics:
        df = df.append(frame)
    df.to_csv('vodafone_metrics_complete_oct_30_dcca_normalized.csv')