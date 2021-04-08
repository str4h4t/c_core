import pandas as pd
import numpy as np
import pickle
import glob


osid_dictionary = {}
# nh_files = glob.glob('results_anonymised_updated//df_nhresult_vodafone_01-02_2020//part*.csv')
topo_files = glob.glob('vodafone_complete_data_october//df_topo_sorted_vodafone*.csv')
for topo_file in topo_files:
    topo_data = pd.read_csv(topo_file)
    for index, row in topo_data.iterrows():
        current_osid = row['osid_scrambled']
        if current_osid not in osid_dictionary.keys():
            osid_dictionary[current_osid] = []
        tid = "_".join(row['tid_scrambled'].split("-"))+"_"+str(row['shelf'])
        if tid not in osid_dictionary[current_osid]:
            osid_dictionary[current_osid].append(tid)
        if row['neighbor1_final'] != "-":
            port_name = "_".join(row['neighbor1_final'].split("-")[1:-1])
            if port_name not in osid_dictionary[current_osid]:
                osid_dictionary[current_osid].append(port_name)
        if row['neighbor2_final'] != "-":
            port_name = "_".join(row['neighbor2_final'].split("-")[1:-1])
            if port_name not in osid_dictionary[current_osid]:
                osid_dictionary[current_osid].append(port_name)

prime_pms = ["OCH-OPR","OPR-OTS","OPIN-OTS","OPROSC-OTS","OCH-OPT","OPT-OTS","OPOUT-OTS"]
nh_parent_files = glob.glob('vodafone_complete_data_october//df_nhresult_vodafone*')
time_series_data = {}
for nh_parent_file in nh_parent_files:
    print(nh_parent_file)
    nh_files = glob.glob(nh_parent_file+'//part*.csv')
    for nh_file in nh_files:
        #f_name = "/home/k_mathin/PycharmProjects/Ciena/Data/vodafone_complete_data_october"+nh_file
        data = pd.read_csv(nh_file)

        for index, row in data.iterrows():
            if row['pm'] not in prime_pms:
                continue
            current_group = row['pec']+"_"+row['port_key_anonymised']+"_"+row['pm']
            if current_group not in time_series_data.keys():
                time_series_data[current_group] = []
            compare_string = row['mename_anonymised'].replace('-','_')+"_"+str(row['shelf'])
            time_series_data[current_group].append({"ts": row['pmtime'], "pmvalue": row['pmvalue'], "pm": row['pm'], "slot": row['slot'], "port": row['port'], "comp": compare_string})

max_len = max([pd.DataFrame(time_series_data[key])['pmvalue'].count() for key in time_series_data.keys()])

time_series_essential = []
min_length = []
for osid in osid_dictionary:
    print(osid)
    for shelf in osid_dictionary[osid]:
        for key in time_series_data.keys():
            if time_series_data[key][0]['comp'] == shelf:
                x = pd.DataFrame(time_series_data[key])
                if (np.var(x['pmvalue']) > 0.05) & (x['pmvalue'].__len__()>60):
                    min_length.append(x['ts'].count())
                    x = x.sort_values('ts')
                    time_series_essential.append({"data": np.asarray(x['pmvalue']), "osid": osid, "shelf": shelf,"slot": x['slot'][0], "port": x['port'][0], "pm": x['pm'][0], "timestamp": x['ts']})
print(min(min_length))
f = open("vodafone_complete_data_filtered_not_extrapolated.pkl","wb")
pickle.dump(pd.DataFrame(time_series_essential),f)
f.close()
print("done")
#print(min(min_length))
