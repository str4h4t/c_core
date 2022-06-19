import pandas as pd
import numpy as np
import pickle

topo = pd.read_csv('Data/df_topo_with_ots_02-12-2020.csv', usecols=['tid_scrambled', 'lim', 'neighbor1_final', 'neighbor1_lim']).dropna().reset_index(drop=True)
lim_1 = topo["lim"].str.split('-', expand=True)
lim_2 = topo["neighbor1_lim"].str.split('-', expand=True)
col_1 = topo['tid_scrambled']+'_'+lim_1[1]+'_'+lim_1[2]
col_2 = topo['neighbor1_final'].str.split('-', expand=True)
col_2 = col_2[1]+'-'+col_2[2]+'-'+col_2[3]+'_'+lim_2[1]+'_'+lim_2[2]

connections = pd.concat((col_1, col_2), axis=1, keys=['node_1', 'node_2'])

g = lambda x: x[0]
tid_connections = pd.concat((g(connections.node_1.str.split('_', expand=True)), g(connections.node_2.str.split('_', expand=True))), axis=1, keys=['node_1', 'node_2'])

connections = connections[tid_connections.node_1 != tid_connections.node_2].sort_values(by='node_1').reset_index(drop=True)
connections = pd.concat((connections.min(axis=1), connections.max(axis=1)), axis=1, keys=['node_1', 'node_2']).drop_duplicates().reset_index(drop=True)

port_lvl = pd.concat((pd.concat((connections.node_1+'_6', connections.node_2+'_8'), axis=1, keys=['node_1', 'node_2']), pd.concat((connections.node_1+'_8', connections.node_2+'_6'), axis=1, keys=['node_1', 'node_2'])), axis=0)




