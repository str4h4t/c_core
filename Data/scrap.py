# plots all the time series within a osid


import pickle
import scipy.stats as st
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
variance_threshold = 0.05

with open('SX_data_cross_osid_mar_19_tsd.pkl', 'rb') as f:
    tsd = pickle.load(f)

with open('SX_data_cross_osid_mar_19_filtered_interpolated.pkl', 'rb') as f:
    data = pickle.load(f)
#tsd.t
date_bucket = []
for i in range(0,tsd.__len__()):
    print(i)
    #pd.to_datetime(tsd['timestamp'][i], unit='s').dt.date
    date_bucket = np.hstack((date_bucket, np.asarray(tsd['timestamp'][i])))
date_bucket = np.unique(date_bucket)
#date_bucket = date_bucket[date_bucket > datetime.date(2020, 1, 1)]
interpolated_time_series = []
for i in range(0,tsd.__len__()):
    series = pd.DataFrame(tsd['data'][i], tsd['timestamp'][i])
    series = series[~series.index.duplicated(keep='first')]
    #series = series[series.index > datetime.date(2020, 1, 1)]
    series = series.reindex(date_bucket, fill_value=0).sort_index().mask(series == 0).interpolate()
    if series.isna().sum()[0] > series.__len__() / 2 or series.var()[0] < variance_threshold:
        continue
    elif series.isna().sum()[0] > 0:
        series = series.fillna(0)
    series['zscore'] = st.zscore(series)
    interpolated_time_series.append(
        {"track": tsd['track'][i], "node": tsd['node'][i], "slot": tsd['slot'][i],
         "port": tsd['port'][i], "pm": tsd['pm'][i], "raw_data": np.asarray(series[0]),
         "z-score": np.asarray(series['zscore']), "timestamp": np.asarray(series.index)})
    print(i)
f = open("SX_data_cross_osid_mar_19_filtered_interpolated.pkl","wb")
pickle.dump(pd.DataFrame(interpolated_time_series),f)
f.close()
print("hello")
