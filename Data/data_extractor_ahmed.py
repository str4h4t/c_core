import pandas as pd
import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt
import pickle
import glob
from tqdm import tqdm
import sys

topo_files = glob.glob('topo_with_different_osids_updated//df_topo_*.csv')

list_topo_files = []
for file in tqdm(topo_files):
    df = pd.read_csv(file, usecols=['tid_scrambled', 'lim', 'neighbor1_final', 'neighbor1-lim'])
    lim_1 = df["lim"].str.split('-', expand=True)
    lim_2 = df["neighbor1-lim"].str.split('-', expand=True)
    neighbour = df.neighbor1_final.str.split('-', expand=True)
    neighbour = neighbour[1] + '-' + neighbour[2] + '-' + neighbour[3]

    col_1 = df['tid_scrambled'] + '_' + lim_1[1] + '_' + lim_1[2]
    col_2 = neighbour + '_' + lim_2[1] + '_' + lim_2[2]
    df = pd.DataFrame({'node_1': col_1, 'node_2': col_2})
    df = pd.DataFrame({'node_1': df.min(axis=1), 'node_2': df.max(axis=1)}).drop_duplicates(ignore_index=True)

    list_topo_files.append(df)
df = pd.concat(list_topo_files, ignore_index=True).drop_duplicates(ignore_index=True).sort_values(by='node_1',
                                                                                                  ignore_index=True)
# topo_nodes are all the nodes in the topo files
# nh_nodes are the nodes from the interpolated filtered nh files
topo_nodes = pd.concat([df.node_1, df.node_2], axis=0,ignore_index=True).drop_duplicates().to_numpy()
nh_nodes = pd.read_pickle('pmvalues_interpolated_ahmed_mar_20.pkl').columns.to_series().str.split('_').apply(lambda x: x[0]+'_'+x[1]+'_'+x[2]).drop_duplicates().to_numpy()
# INTERSECTION sorted numpy array of the both series of nodes above
intersection_nodes = np.intersect1d(topo_nodes, nh_nodes)
print("Number of nodes in:\n{:20}{}\n{:20}{}\n{:20}{}".format('topo', topo_nodes.shape[0], 'nh_nodes', nh_nodes.shape[0], 'intersection_nodes', intersection_nodes.shape[0]))

df = df.where(df.isin(intersection_nodes)).dropna().reset_index(drop=True)
topo_nodes = pd.concat([df.node_1, df.node_2], axis=0,ignore_index=True).drop_duplicates().to_numpy()
print("There are only {} remaining connection and {} different nodes from the topo files".format(df.shape[0], topo_nodes.shape[0]))

path_dictionary = {}
path = 0


def get_path(dictionary, node):
    for key, values in dictionary.items():
        if node in values:
            return key
    return 'Nan'


def merge_paths(dictionary, path_1, path_2):
    dictionary[path_1].extend(dictionary[path_2])
    del dictionary[path_2]


for i, row in df.iterrows():
    node_1 = row[0]
    node_2 = row[1]

    key_1 = get_path(path_dictionary, node_1)
    key_2 = get_path(path_dictionary, node_2)

    if key_1 == 'Nan' and key_2 == 'Nan':
        path_dictionary['group_' + str(path)] = [node_1, node_2]
        path += 1
    elif key_1 == 'Nan':
        path_dictionary[key_2].append(node_1)
    elif key_2 == 'Nan':
        path_dictionary[key_1].append(node_2)
    elif key_1 != key_2:
        merge_paths(path_dictionary, key_1, key_2)
print('There are', len(path_dictionary.keys()), 'paths')
pd.Series([len(v) for _, v in path_dictionary.items()]).value_counts()
assign_path_dict = {}
s = 0
for key, values in path_dictionary.items():
    for e in values:
        assign_path_dict[e] = key

pmvalues = pd.read_pickle('interpolated_data/pmvalues_interpolated.pkl').transpose()
info = pmvalues.index.to_series().str.split('_', expand=True)
df = pd.concat([pd.DataFrame({'node': info[0]+'_'+info[1]+'_'+info[2], 'port': info[3], 'fac_pm': info[4]+'_'+info[5]}), pmvalues], axis=1)
df = df[df.node.isin(list(topo_nodes))]
df.insert(0, 'group', df.node.apply(lambda x: assign_path_dict[x]))
df.to_pickle('./interpolated_filtered_data/pmvalues_interpolated_filtered_simpleindex.pkl')


pmvalues = pd.read_pickle('interpolated_data/pmvalues_interpolated.pkl').transpose()
info = pmvalues.index.to_series().str.split('_', expand=True)
df = pd.concat([pd.DataFrame({'node': info[0]+'_'+info[1]+'_'+info[2], 'port': info[3], 'fac_pm': info[4]+'_'+info[5]}), pmvalues], axis=1)
df = df[df.node.isin(topo_nodes)]
df.insert(0, 'group', df.node.apply(lambda x: assign_path_dict[x]))
df = df.groupby(['group', 'node', 'port', 'fac_pm']).apply(lambda x: np.array(x)[0][4:])
df.to_pickle('./interpolated_filtered_data/pmvalues_interpolated_filtered_multiindex.pkl')


# riskvalues = pd.read_pickle('interpolated_data/riskvalues_interpolated.pkl').transpose()
# info = riskvalues.index.to_series().str.split('_', expand=True)
# df = pd.concat([pd.DataFrame({'node': info[0]+'_'+info[1]+'_'+info[2], 'port': info[3], 'fac_pm': info[4]+'_'+info[5]}), riskvalues], axis=1)
# df = df[df.node.isin(topo_nodes)]
# df.insert(0, 'group', df.node.apply(lambda x: assign_path_dict[x]))
# df = df.groupby(['group', 'node', 'port', 'fac_pm']).apply(lambda x: np.array(x)[0][4:])
# df


# df.to_pickle('./interpolated_filtered_data/riskvalues_interpolated_filtered.pkl')