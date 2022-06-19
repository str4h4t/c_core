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
        data_set_org = pickle.load(f)
    # with open('pmvalues_interpolated_complete_vodafone_11000.pkl', 'rb') as f:
    #     data_set = pickle.load(f)
    data_set = pd.read_csv('fft_abs_vodafone_raw_2310.csv')
    data_set = data_set.drop(columns='Unnamed: 0')
    data = np.asarray(data_set)
    data_set = data_set.reset_index()
    all_results = {'Pearson': [], 'Spearman': [], 'Kendall': [], 'DCCA': []}
    N = data_set.__len__()
    #param = ["pearson", "spearman","kendall"]
    adj_m_pred_pearson = np.zeros([data_set_org.shape[0], data_set_org.shape[0]])
    adj_m_pred_spearman = np.zeros([data_set_org.shape[0], data_set_org.shape[0]])
    adj_m_pred_kendall = np.zeros([data_set_org.shape[0], data_set_org.shape[0]])
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            if data_set_org.iloc[i]['node'] == data_set_org.iloc[j]['node']:
                continue
            df = pd.DataFrame([data[i], data[j]]).transpose()
            pearson = df.corr(method='pearson')[0][1]
            spearman = df.corr(method='spearman')[0][1]
            kendall = df.corr(method='kendall')[0][1]
            adj_m_pred_pearson[i][j] = pearson
            adj_m_pred_pearson[j][i] = pearson
            adj_m_pred_spearman[i][j] = spearman
            adj_m_pred_spearman[j][i] = spearman
            adj_m_pred_kendall[i][j] = kendall
            adj_m_pred_kendall[j][i] = kendall
    np.save('cor_matrix_no_layers_cross_osids_port_level_0921_fft_abs_features_vodafone_pearson.npy', adj_m_pred_pearson)
    # np.save('cor_matrix_no_layers_cross_osids_port_level_0921_agg_features_vodafone_spearman.npy', adj_m_pred_spearman)
    # np.save('cor_matrix_no_layers_cross_osids_port_level_0921_agg_features_vodafone_kendall.npy',  adj_m_pred_kendall)

