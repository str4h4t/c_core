# %%

import pandas as pd
import numpy as np
import pickle
import glob
import scipy.stats as st
from scipy.special import softmax
import datetime
from tslearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, confusion_matrix

# %%

LIM_fac = ["AMP"]
ESAM_fac = ["OPTMON", "OSC"]
OSC_fac = ["OPTMON, OSC"]
WSSOPM_fac = ["OPTMON"]
prime_pms = ["OCH-OPR", "OPR-OTS", "OPIN-OTS", "OPROSC-OTS", "OCH-OPT", "OPT-OTS", "OPOUT-OTS"]
variance_threshold = 0.05
seq_length_threshold = 60
# nh_files = glob.glob('results_anonymised_updated//df_nhresult_vodafone_01-02_2020//part*.csv')
topo_files = glob.glob('results_anonymised_Oct_30//df_topo_*.csv')
nh_parent_files = glob.glob('results_anonymised_Oct_30//df_nhresult_vodafone*')

# %%

import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# %%

# a = np.load('adj_gt_cross_osids_vodafone_ahmed.npy')


# %%

# get connection information of the network from topo data
# -----------------------------------------------------

path_dictionary = {}
path = 1

for topo_file in topo_files:
    topo_data = pd.read_csv(topo_file)
    for index, row in topo_data.iterrows():

        current_tid = row['tid_scrambled']
        _, current_shelf, current_slot = row['lim'].split('-')
        current_tid = current_tid + "_" + current_shelf + "_" + current_slot
        current_n1 = row['neighbor1_scrambled']
        _, n1_shelf, n1_slot = row['neighbor1-lim'].split('-')
        current_n1 = current_n1 + "_" + n1_shelf + "_" + n1_slot
        if path == 1:
            key_name = "path_" + str(path)
            path_dictionary[key_name] = []
            path_dictionary[key_name].append(current_tid)
            path_dictionary[key_name].append(current_n1)
            path += 1
            continue
        flag = 0
        for p in path_dictionary.keys():
            if (current_tid in path_dictionary[p]) and (current_n1 in path_dictionary[p]):
                flag = 1
                continue
            if current_tid in path_dictionary[p]:
                if current_n1 not in path_dictionary[p]:
                    path_dictionary[p].append(current_n1)
                    flag = 1
                    continue
            if current_n1 in path_dictionary[p]:
                if current_tid not in path_dictionary[p]:
                    path_dictionary[p].append(current_tid)
                    flag = 1
                    continue
        if flag == 0:
            key_name = "path_" + str(path)
            path_dictionary[key_name] = []
            path_dictionary[key_name].append(current_tid)
            path_dictionary[key_name].append(current_n1)
            path += 1

# %%

# show initial cluster information
# -----------------------------------------------------
keys = path_dictionary.keys()
values = path_dictionary.values()
columns = ['TID_1', 'TID_2', 'TID_3', 'TID_4', 'TID_5']
cluster_information = pd.DataFrame(values, keys, columns=columns)
print("Number of cluster: %d" % (cluster_information.shape[0]))
print(cluster_information)

# %%

pmvalues = load_obj('pmvalues_interpolated_filtered_simpleindex')
time_series_data = pmvalues
time_series_data

# %%

# Change the format of time_series_data by adding z-score collumn
time_series_essential = []
for i in range(len(time_series_data)):
    series = time_series_data.iloc[i][4:]
    zscore = st.zscore(np.asarray(series, dtype='float32'))
    time_series_essential.append(
        {"osid": time_series_data.iloc[i]['osid'], "node": time_series_data.iloc[i]['node'],
         "port": time_series_data.iloc[i]['port'], "fac_pm": time_series_data.iloc[i]['fac_pm'],
         "raw_data": np.asarray(series), "z-score": zscore})

# %%

pd.DataFrame(time_series_essential)

# %%

# Add path information to the data

interpolated_time_series = []
for path in path_dictionary.keys():
    for node in path_dictionary[path]:
        for i in range(len(time_series_essential)):
            if time_series_essential[i]['node'] == node:
                interpolated_time_series.append({"path": path, "osid": time_series_essential[i]['osid'],
                                                 "node": time_series_essential[i]['node'],
                                                 "port": time_series_essential[i]['port'],
                                                 "fac_pm": time_series_essential[i]['fac_pm'],
                                                 "raw_data": time_series_essential[i]['raw_data'],
                                                 "z-score": time_series_essential[i]['z-score']})

# %%

pd.DataFrame(interpolated_time_series)

# %%

# VALIDATION
# -----------------------------------------------------

# create the target matrix
# -----------------------------------------------------

key_attributes = ['path', 'node', 'z-score']
ts_with_node = pd.DataFrame(interpolated_time_series)[key_attributes]
n = len(ts_with_node)

group_number = ts_with_node['path']
target_matrix = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(n):
        if group_number[i] == group_number[j]:
            target = 1
        else:
            target = 0
        target_matrix[i, j] = target

# %%

target_matrix[:15, :15]

# %%

# create the prediction matrix
# -----------------------------------------------------

similarity_matrix = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(i, n):
        similarity = (np.power(
            metrics.dtw(ts_with_node.iloc[i]['z-score'], np.zeros(ts_with_node.iloc[i]['z-score'].size)), 2)
                      + np.power(
                    metrics.dtw(ts_with_node.iloc[j]['z-score'], np.zeros(ts_with_node.iloc[j]['z-score'].size)), 2)
                      - np.power(metrics.dtw(ts_with_node.iloc[i]['z-score'], ts_with_node.iloc[j]['z-score']), 2)) / 2
        similarity_matrix[i, j] = similarity

for i in range(1, n):
    for j in range(i):
        similarity_matrix[i, j] = similarity_matrix[j, i]

similarity_matrix_normalized = softmax(similarity_matrix, axis=0)

# %%

save_obj(similarity_matrix, 'similarity_matrix_c5_ahmed')

# %%

# similarity_matrix = load_obj('similarity_matrix_c2')
# similarity_matrix.shape

# %%

score_matrix = softmax(similarity_matrix, axis=0)

# %%

# Flatten matrix
# -----------------------------------------------------

target_flatted = target_matrix.reshape(-1, 1).astype('int32')
prediction_score_flatted = score_matrix.reshape(-1, 1)

# %%

prediction_score_flatted.shape, target_flatted.shape

# %%

precisions, recalls, thresholds = precision_recall_curve(target_flatted, prediction_score_flatted)


# %%

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)  # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)  # Not shown
    plt.grid(True)  # Not shown
    plt.axis([0, 1, 0, 1])  # Not shown


recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
threshold_100_precision = thresholds[np.argmax(precisions >= 1)]
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
plt.plot([0, threshold_90_precision], [0.9, 0.9], "r:")  # Not shown
plt.plot([0, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([threshold_90_precision], [0.9], "ro")  # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro");

# %%

threshold_90_precision

# %%

f1_scores = 2 * (precisions * recalls) / (precisions + recalls)


def plot_precision_recall_f1_score_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.plot(thresholds, f1_scores[:-1], "k:", label="f1_score", linewidth=2)
    plt.legend(loc="center right", fontsize=16)  # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)  # Not shown
    plt.grid(True)  # Not shown
    plt.axis([0, 1, 0, 1])  # Not shown


recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
plot_precision_recall_f1_score_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
plt.plot([0, threshold_90_precision], [0.9, 0.9], "r:")  # Not shown
plt.plot([0, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([threshold_90_precision], [0.9], "ro")  # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro");

# %%

threshold_max_f1_score = thresholds[np.argwhere(f1_scores == max(f1_scores))]
print("The max value of f1_score is %.2f at the threshold of %.2f" % (max(f1_scores), threshold_max_f1_score))

# %%

precisions[np.argwhere(f1_scores == max(f1_scores))]

# %%

recalls[np.argwhere(f1_scores == max(f1_scores))]

# %%

max(f1_scores)


# %%

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
# save_fig("precision_vs_recall_plot")
plt.show()

# %%

## CONSTRUCT THE TOPOLOGY OF THE NETWORK FROM PM VALUES ON PORTS OF ROUTERS

# get the threshold = 0.02 for the highest F1-score
# -----------------------------------------------------

threshold = 0.02
prediction_matrix = np.where(score_matrix > threshold, 1, 0)

constructed_connections = {'connection_0': ['test']}
c = 1
for i in range(n):
    acc_flag = 0  # accumulated value
    for connection in constructed_connections.keys():
        if ts_with_node.iloc[i]['node'] in constructed_connections[connection]:
            acc_flag = acc_flag + 1
        else:
            continue
    if acc_flag == 0:
        key_name = "connection_" + str(c)
        constructed_connections[key_name] = []
        constructed_connections[key_name].append(ts_with_node.iloc[i]['node'])
        for j in range(n):
            if prediction_matrix[i, j] == 1 and ts_with_node.iloc[j]['node'] not in constructed_connections[key_name]:
                constructed_connections[key_name].append(ts_with_node.iloc[j]['node'])
        c = c + 1
    else:
        continue

del constructed_connections['connection_0']
constructed_connections

# %%

## TRUE TOPOLOGY OF NETWORK PROVIDED IN ADVANCE


network_connections = {'connection_0': ['test']}
c = 1

for i in range(n):
    acc_flag = 0  # accumulated value
    for connection in network_connections.keys():
        if ts_with_node.iloc[i]['node'] in network_connections[connection]:
            acc_flag = acc_flag + 1
        else:
            continue
    if acc_flag == 0:
        key_name = "connection_" + str(c)
        network_connections[key_name] = []
        network_connections[key_name].append(ts_with_node.iloc[i]['node'])
        for j in range(i + 1, n):
            if (ts_with_node.iloc[j]['path'] == ts_with_node.iloc[i]['path']
                    and ts_with_node.iloc[j]['node'] not in network_connections[key_name]):
                network_connections[key_name].append(ts_with_node.iloc[j]['node'])
        c = c + 1
    else:
        continue

del network_connections['connection_0']
network_connections

# %%

n = 0
for connection in network_connections.keys():
    if len(network_connections[connection]) > 1:
        n = n + 1
n

# %%

## CHECK WHICH CONNECTIONS ARE CONSTRUCTED CORRECTLY

correct_connection = 0
i = 0
for key_true, nodes_true in network_connections.items():
    for key_pred, nodes_pred in constructed_connections.items():
        if nodes_true == nodes_pred and len(nodes_true) > 1:
            correct_connection = correct_connection + 1
            print("Number %d" % i)
            print(key_true, nodes_true)
            print(key_pred, nodes_pred)
            i = i + 1

# %%

print("Number of connections constructed correctly", correct_connection)
print("Constructed rate: %f%%" % (100 * correct_connection / len(network_connections)))

# %%

## CALCULATE THE PRECISION, RECALL
TP = 0
FP = 0
FN = 0
i = 1
nodes_true = [node_true for node_true in network_connections.values()]
nodes_pred = [node_pred for node_pred in constructed_connections.values()]
for key_true, node_true in network_connections.items():
    for key_pred, node_pred in constructed_connections.items():
        if node_true == node_pred:
            TP = TP + 1

FP = len(nodes_pred) - TP
FN = len(nodes_true) - TP
print("TP: %d, FP: %d, FN: %d" % (TP, FP, FN))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print("Precision: %.3f%%" % (100 * precision))
print("Recall: %.3f%%" % (100 * recall))
print("F1_score: %.3f%%" % (100 * f1_score))
