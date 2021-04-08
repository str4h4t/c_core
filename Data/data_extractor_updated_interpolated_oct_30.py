import pandas as pd
import numpy as np
import pickle
import glob
import scipy.stats as st
import datetime

LIM_fac = ["AMP"]
ESAM_fac = ["OPTMON", "OSC"]
OSC_fac = ["OPTMON, OSC"]
WSSOPM_fac = ["OPTMON"]
prime_pms = ["OCH-OPR","OPR-OTS","OPIN-OTS","OPROSC-OTS","OCH-OPT","OPT-OTS","OPOUT-OTS"]
variance_threshold = 0.05
seq_length_threshold = 60
# nh_files = glob.glob('results_anonymised_updated//df_nhresult_vodafone_01-02_2020//part*.csv')
topo_files = glob.glob('results_anonymised_Oct_30//df_topo_*.csv')
nh_parent_files = glob.glob('results_anonymised_Oct_30//df_nhresult_vodafone*')


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
            path+=1
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



time_series_data = {}
for nh_parent_file in nh_parent_files:
    print(nh_parent_file)
    nh_files = glob.glob(nh_parent_file+'//part*.csv')
    for nh_file in nh_files:
        data = pd.read_csv(nh_file)

        for index, row in data.iterrows():
            facility = row['port_key_anonymised'].split("::")[1].split("-")[0]
            # if facility != "AMP":
            #     continue
            # if row['pm'] not in prime_pms:
            #      continue
            current_group = row['pec']+"_"+row['port_key_anonymised']+"_"+row['pm']
            if current_group not in time_series_data.keys():
                time_series_data[current_group] = []
            compare_string = row['mename_anonymised'].replace('-','_')+"_"+str(row['shelf'])+"_"+str(row['slot'])
            time_series_data[current_group].append({"ts": row['pmtime'], "pmvalue": row['pmvalue'], "pm": row['pm'],
                                                    "slot": row['slot'], "port": row['port'], "comp": compare_string})

max_len = max([pd.DataFrame(time_series_data[key])['pmvalue'].count() for key in time_series_data.keys()])

time_series_essential = []
min_length = []
for path in path_dictionary.keys():
    for node in path_dictionary[path]:
        for key in time_series_data.keys():
            if time_series_data[key][0]['comp'] == node.replace("-","_"):
                x = pd.DataFrame(time_series_data[key])
                if (np.var(x['pmvalue']) > variance_threshold) & (x['pmvalue'].__len__() > seq_length_threshold):
                    min_length.append(x['ts'].count())
                    x = x.sort_values('ts')
                    time_series_essential.append({"data": np.asarray(x['pmvalue']), "path": path, "node": node,
                                                  "slot": x['slot'][0], "port": x['port'][0], "pm": x['pm'][0],
                                                  "timestamp": pd.to_datetime(x['ts'], unit='s').dt.date})

tsd = pd.DataFrame(time_series_essential)

date_bucket = []
for i in range(0,tsd.__len__()):
    print(i)
    #pd.to_datetime(tsd['timestamp'][i], unit='s').dt.date
    date_bucket = np.hstack((date_bucket, np.asarray(tsd['timestamp'][i])))
date_bucket = np.unique(date_bucket)
date_bucket = date_bucket[date_bucket > datetime.date(2020, 1, 1)]
interpolated_time_series = []
for i in range(0,tsd.__len__()):
    series = pd.DataFrame(tsd['data'][i], tsd['timestamp'][i])
    series = series[~series.index.duplicated(keep='first')]
    series = series[series.index > datetime.date(2020, 1, 1)]
    series = series.reindex(date_bucket, fill_value=0).sort_index().mask(series == 0).interpolate()
    if series.isna().sum()[0] > series.__len__() / 2 or series.var()[0] < variance_threshold:
        continue
    elif series.isna().sum()[0] > 0:
        series = series.fillna(0)
    series['zscore'] = st.zscore(series)
    interpolated_time_series.append(
        {"path": tsd['path'][i], "node": tsd['node'][i], "slot": tsd['slot'][i],
         "port": tsd['port'][i], "pm": tsd['pm'][i], "raw_data": np.asarray(series[0]),
         "z-score": np.asarray(series['zscore']), "timestamp": np.asarray(series.index)})
    print(i)
f = open("vodafone_data_oct30_not_pm_filtered_interpolated.pkl","wb")
pickle.dump(pd.DataFrame(interpolated_time_series),f)
f.close()