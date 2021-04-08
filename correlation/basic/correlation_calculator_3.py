import pickle
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from numba import vectorize
from math import sqrt

@vectorize(["float32(float32,float32,int32)"], target ='cuda')
def pearson(series_1, series_2, lag):
    #series_1 = [float(pair[0]) for pair in pairs]
    #series_2 = [float(pair[1]) for pair in pairs]
    sum_ctr_1 = 0
    sum_ctr_2 = 0
    sq_ctr_1 = 0
    sq_ctr_2 = 0
    # squares1 = [(sum_ctr_1 + n,s1 + (n * n)) for n in series_1]
    # squares2 = [(sum_ctr_1 + n,s2 + (n * n)) for n in series_2]
    #for n in series_1:
       # sum_ctr_1 += n

    pairs = 1
    #product_sum = np.cumsum([n * m for n, m in pairs])

    #size = len(pairs)

    #numerator = product_sum - ((sum1 * sum2) / size)
    #denominator = sqrt((squares1 - (sum1 * sum1) / size) * (squares2 - (sum2 * sum2) / size))

    # if denominator == 0:
    #     return 0
    return 1.2
    #return numerator / denominator

@vectorize(["float32(float32,float32)"], target ='cuda')
def crosscorr_pearson(datax, datay):
    #datax = pd.Series(datax)
    #datay = pd.Series(datay)
    c = [pearson(datax, datay, lag) for lag in range(-5,5)]
    return c

def crosscorr(datax, datay, meth, lag=0, wrap = False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag), method = meth)
with open('C://Users//k_mathin//PycharmProjects//Masters//ciena_trials//Kamal//data//vodafone_updated_data_filtered_not_extrapolated.pkl', 'rb') as f:
    data_set = pickle.load(f)
result = pd.DataFrame(columns=['series_1', 'series_2', 'pearson','spearman', 'kendall'])
#ctr = 0

for i in range(0,data_set['data'].shape[0]):
    print(i)
    if (i != (data_set['data'].shape[0]-1)):
        #t_1 = pd.Series(data_set['data'][i][:61])
        t_1 = data_set['data'][i][:61]
        for j in range(i+1,data_set['data'].shape[0]):
            #ctr += 1
            #print(ctr)
            #print("i = "+str(i)+" j = "+str(j))
            #t_2 = pd.Series(data_set['data'][j][:61])
            t_2 = data_set['data'][j][:61]
            start = timer()
            rs_pearson = crosscorr_pearson(t_1,t_2)
            print(timer()-start)
            start2 = timer()
            rs_spearman = [crosscorr(t_1, t_2, 'spearman', lag) for lag in
                           range(-5,5)]
            print(timer()-start2)
            rs_kendall = [crosscorr(t_1, t_2, 'kendall', lag) for lag in
                          range(-5,5)]
            rs_pearson = list(np.nan_to_num(rs_pearson))
            rs_spearman = list(np.nan_to_num(rs_spearman))
            rs_kendall = list(np.nan_to_num(rs_kendall))

            result = result.append({'series_1': data_set['osid'][i] + "_" + data_set['shelf'][i] + "_" + str(i),
                                    'series_2': data_set['osid'][j] + "_" + str(j) + data_set['shelf'][j],
                                    'pearson': max(max(rs_pearson),abs(min(rs_pearson))),
                                    'spearman': max(max(rs_spearman),abs(min(rs_spearman))),
                                    'kendall': max(max(rs_kendall),abs(min(rs_kendall))),}, ignore_index = True)
result.to_csv('vodafone_updated_result_filtered_60.csv')

print("done")