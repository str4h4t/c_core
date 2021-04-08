from kshape.core import kshape, zscore
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("C://Users//k_mathin//PycharmProjects//phd_year_1//ciena//data//part-00000-aep_selected.csv")
time_series_data = {}
time_series_dat = {}
for index, row in data.iterrows():
    current_group = row['pec']+"_"+row['port_key_anonymised']+"_"+row['pm']
    if current_group not in time_series_data.keys():
        time_series_data[current_group] = []
    time_series_data[current_group].append({"ts": row['pmtime'], "pmvalue": row['pmvalue']})
    #time_series_dat[current_group].append(row['pmvalue'])
time_series = {}
for key in time_series_data.keys():
    x = pd.DataFrame(time_series_data[key])
    x = x.sort_values('ts')
    x = np.asarray(x['pmvalue'])
    time_series[key] = x.tolist()

f = open("time_series_data.pkl","wb")
pickle.dump(time_series,f)
f.close()
print("done")