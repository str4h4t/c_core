import pickle
import pandas as pd
import numpy as np
import tsfresh

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
data = np.asarray(data_set.iloc[:, 3:])

df = pd.DataFrame()
for i, ts in enumerate(data):
    data = [[x, i] for x in ts]
    df = df.append(data, ignore_index=True)
df.columns = ['value', 'id']
tf=tsfresh.extract_features(df, column_id='id')
tf.to_csv('vodafone_raw_port_lvl_0921_extracted_features.csv')

