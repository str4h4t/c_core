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


with open("/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_data_oct30_filtered_interpolated.pkl", 'rb') as f:
    interpolated_time_series = pickle.load(f)

time_series_df = pd.DataFrame(interpolated_time_series)['z-score']
time_series_list = time_series_df.to_list()
time_series = []
for row in time_series_list:
    time_series.append(row)

time_series_array = np.array(time_series)

attributes = ['path', 'node', 'port', 'pm']
index_df = pd.DataFrame(interpolated_time_series)[attributes]
# print(index)
ts_df = pd.DataFrame(time_series_array)
frames = [index_df, ts_df]
time_series_clustered = pd.concat(frames, axis=1)
# print(time_series_clustered)
time_series_clustered.to_csv('time_series_clustered.csv')

n = time_series_array.shape[0]
# n = 100

# create the target matrix
# -----------------------------------------------------

group_number = pd.DataFrame(interpolated_time_series)['path']
target_matrix = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(n):
        if group_number[i] == group_number[j]:
            target = 1
        else:
            target = 0
        target_matrix[i, j] = target

# print(target_matrix)

# create the prediction matrix
# -----------------------------------------------------

similarity_matrix = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(n):
        similarity = (np.power(metrics.dtw(time_series_array[i], np.zeros(time_series_array[i].size)), 2) +
                   np.power(metrics.dtw(time_series_array[j], np.zeros(time_series_array[j].size)), 2) -
                   np.power(metrics.dtw(time_series_array[i], time_series_array[j]), 2))/2
#         similarity = 1/(1 + metrics.dtw(time_series_array[i], time_series_array[j]))
        similarity_matrix[i, j] = similarity

similarity_matrix_normalized = softmax(similarity_matrix, axis=0)

# target_flatted = target_matrix.reshape(-1, 1).astype('int32')
# prediction_flatted = prediction_matrix.reshape(-1, 1).astype('int32')
# prediction_score_flatted = similarity_matrix_normalized.reshape(-1, 1)
#
# precision = precision_score(target_flatted, prediction_flatted)
# recall = recall_score(target_flatted, prediction_flatted)
# print("precision_score: %.3f" %precision)
# print("recall_score: %.3f" %recall)

# precisions, recalls, thresholds = precision_recall_curve(target_flatted, prediction_score_flatted)

threshold = 0.07
prediction_matrix = np.where(similarity_matrix_normalized > threshold, 1, 0)
target_flatted = target_matrix.reshape(-1, 1).astype('int32')
prediction_flatted = prediction_matrix.reshape(-1, 1).astype('int32')
a = target_matrix.reshape(-1, 1).astype('int')
b = prediction_matrix.reshape(-1, 1).astype('int')
TP = 0
FP = 0
FN = 0
for i in range(a.shape[0]):
    if (b[i] == 1) and (a[i] == 1):
        TP = TP + 1
    elif (b[i] == 1) and (a[i] == 0):
        FP = FP + 1
    elif (b[i] == 0) and (a[i] == 1):
        FN = FN + 1

print("TP: %d FP: %d FN: %d " % (TP, FP, FN))

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print("Precision: %.3f" %precision)
print("Recall: %.3f" %recall)
confusion_matrix(target_flatted, prediction_flatted)
#
# n = 20
#
# group_number = pd.DataFrame(interpolated_time_series)['path']
# target_matrix = np.zeros(shape=(n, n))
# for i in range(n):
#     for j in range(n):
#         if group_number[i] == group_number[j]:
#             target = 1
#         else:
#             target = 0
#         target_matrix[i, j] = target
#
# similarity_matrix = np.zeros(shape=(n, n))
# for i in range(n):
#     for j in range(n):
#         #         similarity = (np.power(metrics.dtw(time_series_array[i], np.zeros(time_series_array[i].size)), 2) +
#         #                    np.power(metrics.dtw(time_series_array[j], np.zeros(time_series_array[j].size)), 2) -
#         #                    np.power(metrics.dtw(time_series_array[i], time_series_array[j]), 2))/2
#         similarity = 1 / (1 + metrics.dtw(time_series_array[i], time_series_array[j]))
#         similarity_matrix[i, j] = similarity
#
# similarity_matrix_normalized = softmax(similarity_matrix, axis=0)
