import pandas as pd
import numpy as np
import pickle
import glob

nh_files = glob.glob('*-aep.csv')
topo_file = "df_topo_sorted_aep.csv"
time_series_data = {}
osid_dictionary = {}
topo_data = pd.read_csv(topo_file)

for index, row in topo_data.iterrows():
    current_osid = row['osid_scrambled']
    if current_osid not in osid_dictionary.keys():
        osid_dictionary[current_osid] = []
    if row['neighbor1'] != "-":
        port_name = "_".join(row['neighbor1'].split("-")[1:-1])
        if port_name not in osid_dictionary[current_osid]:
            osid_dictionary[current_osid].append(port_name)
    if row['neighbor2'] != "-":
        port_name = "_".join(row['neighbor2'].split("-")[1:-1])
        if port_name not in osid_dictionary[current_osid]:
            osid_dictionary[current_osid].append(port_name)

prime_pms = ["OCH-OPR","OPR-OTS","OPIN-OTS","OPROSC-OTS","OCH-OPT","OPT-OTS","OPOUT-OTS"]
for nh_file in nh_files:
    f_name = "C://Users//k_mathin//PycharmProjects//Masters//ciena_trials//Kamal//data//"+nh_file
    data = pd.read_csv(f_name)

    for index, row in data.iterrows():
        if row['pm'] not in prime_pms:
            continue
        current_group = row['pec']+"_"+row['port_key_anonymised']+"_"+row['pm']
        if current_group not in time_series_data.keys():
            time_series_data[current_group] = []
        compare_string = row['mename_anonymised'].replace('-','_')+"_"+str(row['shelf'])
        time_series_data[current_group].append({"ts": row['pmtime'], "pmvalue": row['pmvalue'], "pm": row['pm'], "comp": compare_string})

max_len = max([pd.DataFrame(time_series_data[key])['pmvalue'].count() for key in time_series_data.keys()])

time_series = []
min_length = []
for osid in osid_dictionary:
    for shelf in osid_dictionary[osid]:
        for key in time_series_data.keys():
            if time_series_data[key][1]['comp'] == shelf:
                x = pd.DataFrame(time_series_data[key])
                if np.var(x['pmvalue']) > 0.05:
                    x = x.sort_values('ts')
                    # x_val = np.arange(x['ts'].count())
                    # y_val = np.asarray(x['pmvalue'])
                    #min_length.append(x['ts'].count())
                    # x_new = np.arange(x['ts'].count(), max_len, 1)
                    # z = np.polyfit(x_val, y_val, 3)
                    # f = np.poly1d(z)
                    # y_new = f(x_new)
                    # y = np.append(y_val, y_new)
                    # time_series.append({"data": y, "osid": osid, "shelf": shelf, "pm": x['pm'][0]})
                    time_series.append({"data": np.asarray(x['pmvalue']), "osid": osid, "shelf": shelf, "pm": x['pm'][0]})
#print(min(min_length))
f = open("aep_data_clusters_filtered_not_extrapolated.pkl","wb")
pickle.dump(pd.DataFrame(time_series),f)
f.close()
print("done")





























#                 current_group = row['pec']+"_"+row['port_key_anonymised']+"_"+row['pm']
#                 if current_group not in time_series_data.keys():
#                     time_series_data[current_group] = []
#                 time_series_data[current_group].append({"ts": row['pmtime'], "pmvalue": row['pmvalue']})
#             time_series = {}
#             for key in time_series_data.keys():
#                 x = pd.DataFrame(time_series_data[key])
#                 x = x.sort_values('ts')
#                 x = np.asarray(x['pmvalue'])
#                 time_series[key] = x.tolist()
#
#         f = open("time_series_data.pkl","wb")
#         pickle.dump(time_series,f)
#         f.close()
# print("done")