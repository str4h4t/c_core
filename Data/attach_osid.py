import pickle
import scipy.stats as st
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

with open('pmvalues_sx_interpolated_filtered.pkl', 'rb') as f:
    time_series_data = pickle.load(f)

df = pd.read_csv("./sx//results_anonymised//df_topo_sorted_Sep-2019_new.csv", usecols=['osid_scrambled','tid_scrambled', 'lim', 'neighbor1_final', 'neighbor1-lim']).dropna().reset_index(drop=True)
lim_1 = df["lim"].str.split('-', expand=True)
lim_2 = df["neighbor1-lim"].str.split('-', expand=True)
col_1 = df['tid_scrambled']+'_'+lim_1[1]+'_'+lim_1[2]
col_2 = df['neighbor1_final'].str.split('-', expand=True)
pairs = pd.DataFrame({'node_1': col_1, 'node_2': col_2[1]+'-'+col_2[2]+'-'+col_2[3]+'_'+lim_2[1]+'_'+lim_2[2]})
pairs = pd.DataFrame({'node_1':pairs.min(axis=1), 'node_2':pairs.max(axis=1)}).drop_duplicates(ignore_index=True)
time_series_data['group'] = ""
ctr = 0
for index,rows in pairs.iterrows():
    ctr += 1
    found = time_series_data.loc[time_series_data['node'] == rows['node_1']]
    found = found.append(time_series_data.loc[time_series_data['node'] == rows['node_2']])
    for i in found.index:
        time_series_data.at[i,'group'] = "group_"+str(ctr)

for group in time_series_data['group'].unique():
    print(group)
    current_portkeys = time_series_data[time_series_data['group'] == group]
    f, ax = plt.subplots(figsize=(14, 5))
    for i in current_portkeys.index:
        data = st.zscore(time_series_data.iloc[i, 5:-1])
        ax.plot(data,label = current_portkeys['node'][i] + "_" + current_portkeys['fac_pm'][i] + "_port_" + str(current_portkeys['port'][i]))
    ax.legend(loc='upper right', shadow=False, fontsize='small')
    ax.set(xlabel = 'date', ylabel= 'raw_data')
    fig_name = group +"_raw_plot.png"
    fig_name = fig_name.replace(":", "-")
    plt.savefig(fig_name)
    plt.clf()
print("done")
