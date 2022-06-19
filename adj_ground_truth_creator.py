import correlation_engine as ce
import pickle
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import pickle


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
with open('topo_connections_port_lvl_0921.pkl', 'rb') as f:
    topo_pairs = pickle.load(f)
data_set = data_set.loc[data_set['pm'] == 'OPOUT-OTS']
data_set = data_set.reset_index()
# data_set['node'] = data_set['node'].str.replace('-','_')
#with open('spearman_complete_oct_30_z_score_corr_matrix.pkl', 'rb') as f:
    #c_mat = pickle.load(f)

# topo_files = glob.glob('Data//sx//results_anonymised//df_topo_*.csv')
#
# list_topo_files = []
# for file in tqdm(topo_files):
#     df = pd.read_csv(file, usecols=['tid_scrambled', 'lim', 'neighbor1_final', 'neighbor1-lim'])
#     lim_1 = df["lim"].str.split('-', expand=True)
#     lim_2 = df["neighbor1-lim"].str.split('-', expand=True)
#     neighbour = df.neighbor1_final.str.split('-', expand=True)
#     neighbour = neighbour[1] + '-' + neighbour[2] + '-' + neighbour[3]
#
#     col_1 = df['tid_scrambled'] + '_' + lim_1[1] + '_' + lim_1[2]
#     col_2 = neighbour + '_' + lim_2[1] + '_' + lim_2[2]
#     df = pd.DataFrame({'node_1': col_1, 'node_2': col_2})
#     df = pd.DataFrame({'node_1': df.min(axis=1), 'node_2': df.max(axis=1)}).drop_duplicates(ignore_index=True)
#
#     list_topo_files.append(df)
# df = pd.concat(list_topo_files, ignore_index=True).drop_duplicates(ignore_index=True).sort_values(by='node_1', ignore_index=True)
# topo_pairs = pd.concat([df['node_1'].str.replace('-','_'),df['node_2'].str.replace('-','_')],axis =1)
adj_gt = np.zeros([data_set.__len__(),data_set.__len__()])
for i in data_set.index:
    neighbors = topo_pairs[topo_pairs['node_1'].str.contains(data_set['node'][i]) | topo_pairs['node_2'].str.contains(data_set['node'][i])]
    neighbors = pd.unique(neighbors.values.ravel())
    neighbors = np.delete(neighbors, np.argwhere(neighbors == data_set['node'][i]))
    for n in neighbors:
        ind = data_set[data_set['node'] == n].index
        adj_gt[i, ind] = 1
np.save('adj_gt_cross_osids_vodafone_port_level_0921_OPOUT_OTS.npy', adj_gt)
print("hello")
