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

    with open('riskvalues_interpolated_filtered_simpleindex.pkl', 'rb') as f:
        data_set = pickle.load(f)
    data_set = data_set.reset_index()
    data_set['node'] = data_set['node'].str.replace('-', '_')
    data = np.asarray(data_set.iloc[:,5:])
    data = z_norm(data)
    #data = np.load("/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_data_oct30_filtered_interpolated.pkl")
    # with open('100_clusters_newdata_notnormalized.pkl', 'rb') as f:
    #     assignments = pickle.load(f)
    corr = ce.correlation_engine(data, "kshape", parameters, boost)
    assignments = corr.execute()
    all_results = {'Pearson': [], 'Spearman': [], 'Kendall': [], 'DCCA': []}


    result = []

    param = [["pearson",[4]], ["spearman",[4]], ["kendall",[4]], ["dcca", 6]]
    for p in param:
        adj_m_pred = np.zeros([data_set.shape[0], data_set.shape[0]])
        for index, row in assignments.iterrows():
            print(index)
            members  = row[1]
            member_data = data[members]
            member_details = data_set.iloc[members]
            corr2 = ce.correlation_engine(member_data, p[0], p[1], boost)
            curr_result = corr2.execute()
            #result = a_cor.executor(member_details, curr_result, adj_m_pred, members, boost)
            indices = member_details.index
            ctr = 0
            for ind in indices:
                adj_m_pred[ind,indices] = curr_result[ctr]
                ctr+=1
        np.save('kshape_cor_matrix_cross_osids_vodafone_risk_a_' + p[0] + '.npy',adj_m_pred)

    # track_accuracy = result['truth'].sum()/result.__len__()
    # all_results['Pearson'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
    #                                'total_predictions': result.__len__(), 'accuracy': track_accuracy})
    #print("track Accuracy (Pearson): " + str(track_accuracy) + " for " + str(result.__len__()) + " predictions")
    # result = []
    # for index, row in assignments.iterrows():
    #     members  = row[1]
    #     member_data = data[members]
    #     member_details = data_set.iloc[members]
    #     corr2 = ce.correlation_engine(member_data, "spearman", [4], boost)
    #     curr_result = corr2.execute()
    #     result = a_cor.executor(member_details, curr_result, result, threshold, members, boost)
    # result = pd.DataFrame(result)
    # result = result.drop_duplicates(subset='pair_str', keep="first")
    # track_accuracy = result['truth'].sum() / result.__len__()
    # all_results['Spearman'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
    #                                'total_predictions': result.__len__(), 'accuracy': track_accuracy})
    # #print("track Accuracy (Spearman): " + str(track_accuracy) + " for " + str(result.__len__()) + " predictions")
    # result = []
    # for index, row in assignments.iterrows():
    #     members  = row[1]
    #     member_data = data[members]
    #     member_details = data_set.iloc[members]
    #     corr2 = ce.correlation_engine(member_data, "kendall", [4], boost)
    #     curr_result = corr2.execute()
    #     result = a_cor.executor(member_details, curr_result, result, threshold, members, boost)
    # result = pd.DataFrame(result)
    # result = result.drop_duplicates(subset='pair_str', keep="first")
    # track_accuracy = result['truth'].sum() / result.__len__()
    # all_results['Kendall'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
    #                                'total_predictions': result.__len__(), 'accuracy': track_accuracy})
    # #print("track Accuracy (Kendall): " + str(track_accuracy) + " for " + str(result.__len__()) + " predictions")
    #
    # result = []
    # for index, row in assignments.iterrows():
    #     members  = row[1]
    #     member_data = data[members]
    #     member_details = data_set.iloc[members]
    #     corr4 = ce.correlation_engine(member_data, "dcca", 6, boost)
    #     curr_result = corr4.execute()
    #     result = a_cor.executor(member_details, curr_result, result, threshold, members, boost)
    # if not result:
    #     continue
    # result = pd.DataFrame(result)
    # result = result.drop_duplicates(subset='pair_str', keep="first")
    # track_accuracy = result['truth'].sum() / result.__len__()
    # all_results['DCCA'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
    #                                'total_predictions': result.__len__(), 'accuracy': track_accuracy})