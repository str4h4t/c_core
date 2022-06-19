import pandas as pd
import numpy as np


df = pd.read_csv("./sx//results_anonymised//df_topo_sorted_Sep-2019_new.csv", usecols=['tid_scrambled', 'lim', 'neighbor1_final', 'neighbor1-lim']).dropna().reset_index(drop=True)
lim_1 = df["lim"].str.split('-', expand=True)
lim_2 = df["neighbor1-lim"].str.split('-', expand=True)
col_1 = df['tid_scrambled']+'_'+lim_1[1]+'_'+lim_1[2]
col_2 = df['neighbor1_final'].str.split('-', expand=True)
df = pd.DataFrame({'node_1': col_1, 'node_2': col_2[1]+'-'+col_2[2]+'-'+col_2[3]+'_'+lim_2[1]+'_'+lim_2[2]})
df = pd.DataFrame({'node_1':df.min(axis=1), 'node_2':df.max(axis=1)}).drop_duplicates(ignore_index=True)

topo_nodes = pd.concat([df.node_1, df.node_2], axis=0,ignore_index=True).drop_duplicates()
nh_nodes = pd.read_pickle('pmvalues.pkl').node.drop_duplicates().reset_index(drop=True)
intersection_nodes = np.intersect1d(topo_nodes.to_numpy(), nh_nodes.to_numpy(), assume_unique=True)

pmvalues = pd.read_pickle('pmvalues.pkl')
#info = pmvalues.index.to_series().str.split('_', expand=True)
#df = pd.concat([pd.DataFrame({'node1': info[0]+'_'+info[1]+'_'+info[2], 'port': info[3], 'fac_pm': info[4]+'_'+info[5]}), pmvalues], axis=1)
nh_filtered = pmvalues[pmvalues.node.isin(list(topo_nodes))]
nh_nodes = nh_filtered['node'].unique()
ctr = 0
ctr2 = 0
for row in df.iterrows():
    if (row[1]['node_1'] in nh_nodes) and (row[1]['node_2'] in nh_nodes):
        ctr += 1
    else:
        ctr2 += 1
print(ctr)
print(ctr2)
nh_filtered = nh_filtered.reset_index()
adj_gt = np.zeros([nh_filtered.__len__(),nh_filtered.__len__()])
for i in nh_filtered.index:
    neighbors = df[df['node_1'].str.contains(nh_filtered['node'][i]) | df['node_2'].str.contains(nh_filtered['node'][i])]
    neighbors = pd.unique(neighbors.values.ravel())
    neighbors = np.delete(neighbors, np.argwhere(neighbors == nh_filtered['node'][i]))
    for n in neighbors:
        ind = nh_filtered[nh_filtered['node'] == n].index
        if ind.size == 0:
            continue
        adj_gt[i, ind] = 1
np.save('../adj_gt_cross_osids_sx.npy', adj_gt)
nh_filtered.to_pickle('pmvalues_sx_interpolated_filtered.pkl')