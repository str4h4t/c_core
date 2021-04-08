# plots all the time series within a osid


import pickle
import scipy.stats as st
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

with open('vodafone_complete_data_filtered_not_extrapolated.pkl', 'rb') as f:
    time_series_data = pickle.load(f)


for osid in time_series_data['osid'].unique():
    all_portkeys = time_series_data[time_series_data['osid'] == osid]
    for shelf in all_portkeys['shelf'].unique():
        shelf_portkeys = all_portkeys[all_portkeys['shelf'] == shelf]
        for pm in shelf_portkeys['pm'].unique():
            current_portkeys = shelf_portkeys[shelf_portkeys['pm'] == pm]
            date_bucket = []
            for i in current_portkeys.index:
                pd.to_datetime(current_portkeys['timestamp'][i], unit='s').dt.date
                date_bucket = np.hstack((date_bucket, np.asarray(pd.to_datetime(current_portkeys['timestamp'][i], unit='s').dt.date)))
            date_bucket = np.unique(date_bucket)
            f, ax = plt.subplots(figsize=(14, 3))
            for i in current_portkeys.index:
                series = pd.DataFrame(current_portkeys['data'][i], pd.to_datetime(current_portkeys['timestamp'][i], unit='s').dt.date)
                series = series[~series.index.duplicated(keep='first')]
                series = series.reindex(date_bucket, fill_value=0).sort_index().mask(series == 0).interpolate()
                series = series[series.index > datetime.date(2020, 1, 1)]
                series['zscore'] = st.zscore(series)
                ax.plot(series['zscore'])
            fig_name = osid + "_" + shelf + "_" + pm + "_plot.png"
            fig_name = fig_name.replace(":", "-")
            plt.savefig(fig_name)
            plt.clf()
    print(osid)