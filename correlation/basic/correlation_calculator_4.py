import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp

with open(
        'C://Users//k_mathin//PycharmProjects//Masters//ciena_trials//Kamal//data//vodafone_updated_data_filtered_not_extrapolated.pkl',
        'rb') as f:
    data_set = pickle.load(f)

def crosscorr(datax, datay, meth, lag=0):
    return datax.corr(datay.shift(lag), method = meth)

def calculator(i, data_set = data_set):
    result = pd.DataFrame(columns=['series_1', 'series_2', 'pearson'])
    #result = pd.DataFrame(columns=['series_1', 'series_2', 'pearson', 'spearman', 'kendall'])
    #for i in range(0,data_set['data'].shape[0]):
    print(i)
    if (i != (data_set['data'].shape[0]-1)):
        t_1 = pd.Series(data_set['data'][i][:61])
        for j in range(i+1,data_set['data'].shape[0]):
                #ctr += 1
                #print(ctr)
                #print("i = "+str(i)+" j = "+str(j))
            t_2 = pd.Series(data_set['data'][j][:61])
            rs_pearson = [crosscorr(t_1, t_2, 'pearson', lag) for lag in
                          range(-5,5)]
                # rs_spearman = [crosscorr(t_1, t_2, 'spearman', lag) for lag in
                #                range(-5,5)]
                # rs_kendall = [crosscorr(t_1, t_2, 'kendall', lag) for lag in
                #               range(-5,5)]
            rs_pearson = list(np.nan_to_num(rs_pearson))
                # rs_spearman = list(np.nan_to_num(rs_spearman))
                # rs_kendall = list(np.nan_to_num(rs_kendall))

                # result = result.append({'series_1': data_set['osid'][i] + "_" + data_set['shelf'][i] + "_" + str(i),
                #                         'series_2': data_set['osid'][j] + "_" + str(j) + data_set['shelf'][j],
                #                         'pearson': max(max(rs_pearson),abs(min(rs_pearson))),
                #                         'spearman': max(max(rs_spearman),abs(min(rs_spearman))),
                #                         'kendall': max(max(rs_kendall),abs(min(rs_kendall))),}, ignore_index = True)
            result = result.append({'series_1': data_set['osid'][i] + "_" + data_set['shelf'][i] + "_" + str(i),
                                    'series_2': data_set['osid'][j] + "_" + str(j) + data_set['shelf'][j],
                                     'pearson': max(max(rs_pearson), abs(min(rs_pearson)))}, ignore_index=True)
    #result.to_csv('vodafone_updated_result_filtered_60.csv')
    return(result)
if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(calculator, [i for i in range(0,data_set['data'].shape[0])])
    # pool.map(howmany_within_range_rowonly, [row for row in data])
    #calculator(data_set)
    pool.close()
    print("done")