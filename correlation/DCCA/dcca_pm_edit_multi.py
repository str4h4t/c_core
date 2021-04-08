import numpy as np
from numpy.matlib import repmat
import pickle
import pandas as pd
import multiprocessing as mp
#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

with open('/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_complete_data_filtered_interpolated.pkl', 'rb') as f:
    data_set = pickle.load(f)
data = np.zeros(data_set['z-score'][0].__len__())
for d in range(0, data_set['z-score'].__len__()):
    data = np.vstack((data, data_set['z-score'][d]))

data = np.delete(data, 0, axis=0)
data_set = data_set[~np.isnan(data).any(axis=1)]
data_set.index = range(0, data_set.__len__())
k = 6
pms = data_set['pm'].unique()

# Return sliding windows
def sliding_window(xx, k):
    idx = np.arange(k)[None, :] + np.arange(len(xx) - k + 1)[:, None]
    return xx[idx], idx


def compute_dpcca_others(cdata, k):
    nsamples, nvars = cdata.shape
    cdata = cdata - cdata.mean(axis=0)
    xx = np.cumsum(cdata, axis=0)

    F2_dfa_x = np.zeros(nvars)
    allxdif = []

    for ivar in range(nvars):
        #print(ivar)
        xx_swin, idx = sliding_window(xx[:, ivar], k)
        nwin = xx_swin.shape[0]
        b1, b0 = np.polyfit(np.arange(k), xx_swin.T, deg=1)


        x_hatx = repmat(b1, k, 1).T * repmat(range(k), nwin, 1) + repmat(b0, k, 1).T

        # Store differences to the linear fit
        xdif = xx_swin - x_hatx
        allxdif.append(xdif)
        # Eq.4
        F2_dfa_x[ivar] = (xdif ** 2).mean()

    # Get the DCCA matrix
    dcca = np.zeros([nvars, nvars])
    for i in range(nvars):  # do for all vars
        for j in range(nvars):  # do for all vars
            # Eq.5 and 6
            #print("i: "+str(i)+"  j: "+str(j))
            F2_dcca = (allxdif[i] * allxdif[j]).mean()
            # Eq.1: DCCA
            dcca[i, j] = F2_dcca / np.sqrt(F2_dfa_x[i] * F2_dfa_x[j])
    corr = np.corrcoef(cdata.T)
    return corr, dcca

def iterator(ii):
    pm1 = pms[ii]
    for jj in range(ii, pms.__len__()):
        pm2 = pms[jj]
        print(pm1)
        print(pm2)
        data_set_1 = data_set[data_set['pm'] == pm1]
        data_set_1.index = range(0, data_set_1.__len__())
        data_set_2 = data_set[data_set['pm'] == pm2]
        data_set_2.index = range(0, data_set_2.__len__())
        threshold = 0.75
        metrics_pearson = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F_score'])
        metrics_dcca = pd.DataFrame(columns=['Threshold', 'Precision', 'Recall', 'F_score'])
        while threshold < 0.95:
            print(threshold)
            tp_ctr_p = 0
            tn_ctr_p = 0
            fp_ctr_p = 0
            fn_ctr_p = 0
            tp_ctr_d = 0
            tn_ctr_d = 0
            fp_ctr_d = 0
            fn_ctr_d = 0
            for i in range(0, data_set_1.__len__() - 1):
                for j in range(0, data_set_2.__len__()):
                    #print("i: " + str(i) + "  j: " + str(j))
                    o1 = data_set_1['osid'][i]
                    o2 = data_set_2['osid'][j]
                    d1 = data_set_1['z-score'][i]
                    d2 = data_set_2['z-score'][j]
                    d = np.vstack((d1, d2))
                    cdata = d.T
                    pearson, dcca = compute_dpcca_others(cdata, k)
                    if o1 == o2:
                        if abs(pearson[0][1]) >= threshold:
                            tp_ctr_p += 1
                        else:
                            fn_ctr_p += 1
                        if abs(pearson[0][1]) >= threshold:
                            tp_ctr_d += 1
                        else:
                            fn_ctr_d += 1
                    else:
                        if abs(pearson[0][1]) < threshold:
                            tn_ctr_p += 1
                        else:
                            fp_ctr_p += 1
                        if abs(dcca[0][1]) < threshold:
                            tn_ctr_d += 1
                        else:
                            fp_ctr_d += 1
            precision_p = tp_ctr_p / (tp_ctr_p + fp_ctr_p)
            recall_p = tp_ctr_p / (tp_ctr_p + fn_ctr_p)
            f_score_p = 2 * ((precision_p * recall_p) / (precision_p + recall_p))
            metrics_pearson = metrics_pearson.append(
                {'Threshold': threshold, 'Precision': precision_p, 'Recall': recall_p, 'F_score': f_score_p},
                ignore_index=True)
            precision_d = tp_ctr_d / (tp_ctr_d + fp_ctr_d)
            recall_d = tp_ctr_d / (tp_ctr_d + fn_ctr_d)
            f_score_d = 2 * ((precision_d * recall_d) / (precision_d + recall_d))
            metrics_dcca = metrics_dcca.append(
                {'Threshold': threshold, 'Precision': precision_d, 'Recall': recall_d, 'F_score': f_score_d},
                ignore_index=True)
            threshold += 0.01
        np.savetxt("vodafone_metrics_" + pm1 + "_" + pm2 + "_pearson_normalized.csv", metrics_pearson, delimiter=",")
        np.savetxt("vodafone_metrics_" + pm1 + "_" + pm2 + "_dcca_normalized.csv", metrics_dcca, delimiter=",")
if __name__ == "__main__":
    r = range(0,pms.__len__())
    pool = mp.Pool(mp.cpu_count())
    pool.map(iterator, [ii for ii in r])