import correlation_engine as ce
import pickle
import numpy as np
import pandas as pd
import analyzer_correlation as a_cor
import analyzer_distance as a_dis

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

    with open(ip_file, 'rb') as f:
        data_set = pickle.load(f)
    data = np.zeros(data_set[data_format][0].__len__())
    for i in data_set.index:
        data = np.vstack((data, data_set[data_format][i]))
    data = np.delete(data, 0, axis=0)
    #data = np.load("/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_data_oct30_filtered_interpolated.pkl")
    corr = ce.correlation_engine(data, "kshape", parameters, boost)
    assignments = corr.execute()
    all_results = {'Pearson': [], 'Spearman': [], 'Kendall': [], 'DCCA': []}

    for threshold in np.arange(0.75, 1, 0.01):
        print(threshold)
        result = []
        for index, row in assignments.iterrows():
            members  = row[1]
            member_data = data[members]
            member_details = data_set.iloc[members]
            corr2 = ce.correlation_engine(member_data, "pearson", [4], boost)
            curr_result = corr2.execute()
            result = a_cor.executor(member_details, curr_result, result, threshold, members, boost)
            #print(index)
        result = pd.DataFrame(result)
        track_accuracy = result['truth'].sum()/result.__len__()
        all_results['Pearson'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
                                       'total_predictions': result.__len__(), 'accuracy': track_accuracy})
        #print("track Accuracy (Pearson): " + str(track_accuracy) + " for " + str(result.__len__()) + " predictions")
        result = []
        for index, row in assignments.iterrows():
            members  = row[1]
            member_data = data[members]
            member_details = data_set.iloc[members]
            corr2 = ce.correlation_engine(member_data, "spearman", [4], boost)
            curr_result = corr2.execute()
            result = a_cor.executor(member_details, curr_result, result, threshold, members, boost)
        result = pd.DataFrame(result)
        track_accuracy = result['truth'].sum() / result.__len__()
        all_results['Spearman'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
                                       'total_predictions': result.__len__(), 'accuracy': track_accuracy})
        #print("track Accuracy (Spearman): " + str(track_accuracy) + " for " + str(result.__len__()) + " predictions")
        result = []
        for index, row in assignments.iterrows():
            members  = row[1]
            member_data = data[members]
            member_details = data_set.iloc[members]
            corr2 = ce.correlation_engine(member_data, "kendall", [4], boost)
            curr_result = corr2.execute()
            result = a_cor.executor(member_details, curr_result, result, threshold, members, boost)
        result = pd.DataFrame(result)
        track_accuracy = result['truth'].sum() / result.__len__()
        all_results['Kendall'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
                                       'total_predictions': result.__len__(), 'accuracy': track_accuracy})
        #print("track Accuracy (Kendall): " + str(track_accuracy) + " for " + str(result.__len__()) + " predictions")

        result = []
        for index, row in assignments.iterrows():
            members  = row[1]
            member_data = data[members]
            member_details = data_set.iloc[members]
            corr4 = ce.correlation_engine(member_data, "dcca", 6, boost)
            curr_result = corr4.execute()
            result = a_cor.executor(member_details, curr_result, result, threshold, members, boost)
        if not result:
            continue
        result = pd.DataFrame(result)
        track_accuracy = result['truth'].sum() / result.__len__()
        all_results['DCCA'].append({'threshold': threshold, 'correct_predictions': result['truth'].sum(),
                                       'total_predictions': result.__len__(), 'accuracy': track_accuracy})
    #pd.DataFrame(all_results['DCCA']).to_csv('DCCA_20_result.csv')
        #print("track Accuracy (DCCA): " + str(track_accuracy) + " for " + str(result.__len__()) + "predictions")

    # dist_threshold = 0.4
    # dtw_window_type = ["sakoechiba", "itakura", "slantedband"]
    # dtw_window_size = 6
    # dtw_all_results = {}
    # for w in range(2, 6):
    #     for type in dtw_window_type:
    #         temp_key = "dtw" + str(w) + type
    #         dtw_all_results[temp_key] = []
    # while dist_threshold > 0:
    #     print(dist_threshold)
    #     result = {}
    #     for w in range(2, 6):
    #         for type in dtw_window_type:
    #             temp_key = "dtw" + str(w) + type
    #             result[temp_key] = []
    #     for index, row in assignments.iterrows():
    #         members = row[1]
    #         member_data = data[members]
    #         for w in range(2, 6):
    #             for type in dtw_window_type:
    #                 corr3 = ce.correlation_engine(member_data, "dtw", [w, type], boost)
    #                 curr_result = corr3.execute()
    #                 member_details = data_set.iloc[members]
    #                 temp_key = "dtw" + str(w) + type
    #                 result[temp_key] = a_dis.executor(member_details, curr_result, result[temp_key], dist_threshold, members,
    #                                                   boost)
    #     for w in range(2, 6):
    #         for type in dtw_window_type:
    #             temp_key = "dtw" + str(w) + type
    #             result[temp_key] = pd.DataFrame(result[temp_key])
    #             track_accuracy = result[temp_key]['truth'].sum() / result[temp_key].__len__()
    #             dtw_all_results[temp_key].append({'threshold': dist_threshold, 'correct_predictions': result[temp_key]['truth'].sum(),
    #                                         'total_predictions': result[temp_key].__len__(), 'accuracy': track_accuracy})
    #             #print("track Accuracy (DTW_" + temp_key + "): " + str(track_accuracy) + " for " + str(result.__len__()) + "predictions")
    #     #print("Ending Procedure...")
    #     dist_threshold -= 0.01
    for key in all_results.keys():
        pd.DataFrame(all_results[key]).to_csv(key + '_cross_osid_mar19_filtered_result.csv')
    # for key in dtw_all_results.keys():
    #     pd.DataFrame(dtw_all_results[key]).to_csv(key + 'not_pm_filtered_result.csv')