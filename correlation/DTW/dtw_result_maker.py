import pandas as pd
import multiprocessing as mp
def calc_metrics(file,t,size):
    result = pd.read_csv(file)
    result['distance'] = result['distance'].replace(9999, int(result['distance'].astype(float).sort_values().unique()[-2]) + 1)
    p_threshold = 0.4
    metrics = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall','F_score'])
    while p_threshold > 0:
        tp_ctr = 0
        tn_ctr = 0
        fp_ctr = 0
        fn_ctr = 0
        threshold = max(result['distance']) * p_threshold
        for index, row in result.iterrows():
            o1 = row['path_s_1']
            o2 = row['path_s_2']
            distance = row['distance']
            if o1 == o2:
                if distance <= threshold:
                    tp_ctr += 1
                else:
                    fn_ctr += 1
            else:
                if distance > threshold:
                    tn_ctr += 1
                else:
                    fp_ctr += 1
        total = result.shape[0]
        #accuracy = ((tp_ctr+tn_ctr)/total) * 100
        precision = tp_ctr/(tp_ctr+fp_ctr)
        recall = tp_ctr/(tp_ctr+fn_ctr)
        f_score = 2 * ((precision * recall) / (precision + recall))
        print("Threshold: "+ str(threshold) + "\nPrecision: " + str(precision)
              + "\nRecall" + str(recall) + "\nF-score: " + str(f_score))
        metrics = metrics.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F_score': f_score,}, ignore_index=True)
        # print("Threshold: "+ str(threshold) + "\nTrue Positive: "
        #       + str(tp_ctr) + "\tFalse Positive: " + str(fp_ctr)
        #       + "\nFalse Negative: " + str(fn_ctr) + "\tTrue Negative: "
        #       + str(tn_ctr) + "\nAccuracy: "+str(accuracy)+"%")
        p_threshold -= 0.01
    metrics.to_csv("vodafone_oct_30_interpolated_filtered_z-score_metrics_dtw_" + t + "_" + str(size) + "_" + ".csv")
    print("done")

def it(size):
    type = ["sakoechiba", "itakura", "slantedband"]

    for t in type:
        print("Type: " + t)
        print("Window Size: " + str(size))

        file = "vodafone_oct_30_interpolated_filtered_z-score_result_dtw_" + t + "_" + str(size) + "_" + ".csv"
        calc_metrics(file,t,size)

if __name__ == "__main__":
    size = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    pool = mp.Pool(mp.cpu_count())
    pool.map(it, [i for i in size])
    #[it(i) for i in size]