import pandas as pd
import glob
from tqdm import tqdm
import pickle

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
df = pd.concat(list_topo_files, ignore_index=True).drop_duplicates(ignore_index=True).sort_values(by='node_1', ignore_index=True)
# topo_nodes are all the nodes in the topo files
# nh_nodes are the nodes from the interpolated filtered nh files
topo_nodes = pd.concat([df.node_1, df.node_2], axis=0,ignore_index=True).drop_duplicates().to_numpy()
#nh_nodes = pd.read_pickle('pmvalues_interpolated_ahmed_mar_20.pkl').columns.to_series().str.split('_').apply(lambda x: x[0]+'_'+x[1]+'_'+x[2]).drop_duplicates().to_numpy()
ctr = 0

with open('vodafone_data_oct30_cross_osid_mar_19_filtered_interpolated.pkl', 'rb') as f:
    tsd = pickle.load(f)

nh_nodes = tsd['node'].unique()
df = pd.concat([df['node_1'].str.replace('-','_'),df['node_2'].str.replace('-','_')],axis =1)
for row in df.iterrows():
    if (row[1]['node_1'] in nh_nodes) and (row[1]['node_2'] in nh_nodes):
        ctr += 1

print(ctr)