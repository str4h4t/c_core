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
    data_set_org = data_set_org.reset_index()
    data_org = np.asarray(data_set_org.iloc[:, 4:])
    # data_set_feat = pd.read_csv('fft_complete_vodafone_raw_2310.csv')
    # data_set_feat = data_set_feat.reset_index()
    # data_set_feat = data_set_feat.drop(columns='Unnamed: 0')
    # data_set_feat = data_set_feat.drop(columns='index')
    # data_feat = np.asarray(data_set_feat)
    data_feat = np.load('spectral_props.npy')
    corr = ce.correlation_engine(data_feat, "kshape", parameters, boost)
    # corr = ce.correlation_engine(data_org, "kshape", parameters, boost)
    assignments = corr.execute()
    adj_m_pred_pearson = np.zeros([data_set_org.shape[0], data_set_org.shape[0]])
    adj_m_pred_spearman = np.zeros([data_set_org.shape[0], data_set_org.shape[0]])
    adj_m_pred_kendall = np.zeros([data_set_org.shape[0], data_set_org.shape[0]])
    for index, row in assignments.iterrows():
        print(index)
        members  = row[1]
        member_data = data_feat[members]
        member_details = data_set_org.iloc[members]
        corr2 = ce.correlation_engine(member_data, 'all', None , boost)
        curr_adj_pearson, curr_adj_spearman, curr_adj_kendall = corr2.execute()
        #result = a_cor.executor(member_details, curr_result, adj_m_pred, members, boost)
        indices = member_details.index
        ctr = 0
        for ind in indices:
            adj_m_pred_pearson[ind,indices] = curr_adj_pearson[ctr]
            adj_m_pred_spearman[ind, indices] = curr_adj_spearman[ctr]
            adj_m_pred_kendall[ind, indices] = curr_adj_kendall[ctr]
            ctr+=1

    np.fill_diagonal(adj_m_pred_pearson, 0)
    np.fill_diagonal(adj_m_pred_spearman, 0)
    np.fill_diagonal(adj_m_pred_kendall, 0)
    np.save('spectral_props_before_kshape_cor_matrix_cross_osids_port_level_0921_vodafone_pearson.npy',adj_m_pred_pearson)
    np.save('spectral_props_before_kshape_cor_matrix_cross_osids_port_level_0921_vodafone_spearman.npy', adj_m_pred_spearman)
    np.save('spectral_props_before_kshape_cor_matrix_cross_osids_port_level_0921_vodafone_kendall.npy', adj_m_pred_kendall)

