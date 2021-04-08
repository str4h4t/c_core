import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp


def crosscorr(datax, datay, meth, lag=0, wrap = False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag), method = meth)


with open('C://Users//k_mathin//PycharmProjects//Masters//ciena_trials//Kamal//data//vodafone_updated_data_filtered_not_extrapolated.pkl', 'rb') as f:
    data_set = pickle.load(f)
result = pd.DataFrame(columns=['series_1', 'series_2', 'pearson'])
for i in range(0,data_set['data'].shape[0]):
    print(i)
    if (i != (data_set['data'].shape[0]-1)):
        t_1 = pd.Series(data_set['data'][i][:61])
        for j in range(i+1,data_set['data'].shape[0]):
            t_2 = pd.Series(data_set['data'][j][:61])
            rs_pearson = [crosscorr(t_1, t_2, 'pearson', lag) for lag in
                          range(-2,2)]
            rs_pearson = list(np.nan_to_num(rs_pearson))
            result = result.append({'series_1': data_set['osid'][i] + "_" + data_set['shelf'][i] + "_" + str(i),
                                    'series_2': data_set['osid'][j] + "_" + str(j) + data_set['shelf'][j],
                                    'pearson': max(max(rs_pearson), abs(min(rs_pearson)))}, ignore_index=True)
result.to_csv('vodafone_updated_result_filtered_60.csv')

print("done")