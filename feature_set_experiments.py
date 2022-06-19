import pandas as pd
import numpy as np
import pickle

f_data = pd.read_csv('vodafone_raw_port_lvl_0921_extracted_features.csv')
f_data = f_data.dropna(axis='columns')
f_data = f_data.drop(columns = 'Unnamed: 0')
for col in f_data.columns:
    count = (f_data[col] == 1).sum() + (f_data[col] == 0).sum()
    if count == f_data.shape[0]:
        f_data = f_data.drop(columns =col)

with open('topo_connections_port_lvl_0921.pkl', 'rb') as f:
    connections = pickle.load(f)
total_positives = 0

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
data_set = data_set.reset_index()
nh_nodes = data_set['node'].unique()
print("done")

for c in range(0,connections.shape[0]):
    n1 = connections.iloc[c]['node_1']
    n2 = connections.iloc[c]['node_2']
    curr_feature_set = []
    if n1 in nh_nodes and n2 in nh_nodes:
        indices = data_set.index[data_set['node'] == n1]
        indices = indices.append(data_set.index[data_set['node'] == n2])
        curr_feature_set = f_data.iloc[indices]
        print('Done')