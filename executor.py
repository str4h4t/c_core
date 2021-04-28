import numpy as np
import pickle
import correlation_engine as ce
import pandas as pd

if __name__ == '__main__':
    boost = False
    f = open("config.txt","r")
    config = []
    for x in f:
        config.append(x.split('"')[1])
    ip_file = config[0]
    model = config[1]
    parameters = config[2]
    data_format = config[3]
    if config[4] == "1":
        boost = True

    with open(ip_file, 'rb') as f:
        data_set = pickle.load(f)
    data = np.zeros(data_set[data_format][0].__len__())
    for i in data_set.index:
        data = np.vstack((data, data_set[data_format][i]))
    data = np.delete(data, 0, axis=0)
    corr = ce.correlation_engine(data, "dcca", 6, boost)
    c_matrix = corr.execute()
    pd.DataFrame(c_matrix).to_pickle('dcca_complete_oct_30_z_score_corr_matrix.pkl')
    print("done")