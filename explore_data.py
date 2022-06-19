import correlation_engine as ce
import pickle
import numpy as np
import pandas as pd
import analyzer_correlation_adj as a_cor
import analyzer_distance as a_dis
import scipy.stats as st

def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)

if __name__ == "__main__":
    print("Starting Procedure...")
    #threshold = 0.95
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

    with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
        data_set = pickle.load(f)
    #data_set = data_set.transpose()
    #data = np.asarray(data_set)
    data_set = data_set.reset_index()
    data = np.asarray(data_set.iloc[:,4:])