import pandas as pd
import multiprocessing as mp
import numpy as np

result = pd.read_csv("basic/vodafone_oct_30_z_score_result_filtered_interpolated.csv")

def populate_result(threshold,result = result):
    metrics = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F_score'])
    print(threshold)
    tp_ctr = 0
    tn_ctr = 0
    fp_ctr = 0
    fn_ctr = 0
    for index, row in result.iterrows():
        #print(index)
        o1 = row['path_s_1']
        o2 = row['path_s_2']
        correlation = row['kendall']
        # correlation = max([row['pearson'],row['spearman'],row['kendall']])
        if o1 == o2:
            if correlation >= threshold:
                tp_ctr += 1
            else:
                fn_ctr += 1
        else:
            if correlation < threshold:
                tn_ctr += 1
            else:
                fp_ctr += 1
    precision = tp_ctr / (tp_ctr + fp_ctr)
    recall = tp_ctr / (tp_ctr + fn_ctr)
    f_score = 2 * ((precision * recall) / (precision + recall))
    print("Threshold: " + str(threshold) + "\nPrecision: " + str(precision)
          + "\nRecall" + str(recall) + "\nF-score: " + str(f_score))
    metrics = metrics.append(
        {'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F_score': f_score, }, ignore_index=True)
    return metrics
if __name__ == "__main__":
    threshold = np.arange(0.75, 0.95, 0.01)
    pool = mp.Pool(mp.cpu_count()-1)
    #populate_result(0.75)
    metrics = pool.map(populate_result, [i for i in threshold])
    met = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F_score'])
    for m in metrics:
        met = met.append(m)
    met.to_csv('vodafone_metrics_updated_dec_30_z-score_kendall.csv')
    print("done")