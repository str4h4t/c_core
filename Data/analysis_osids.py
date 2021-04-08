# plots all the time series within a osid


import pickle
import scipy.stats as st
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

with open('vodafone_data_oct30_filtered_interpolated.pkl', 'rb') as f:
    time_series_data = pickle.load(f)


for path in time_series_data['path'].unique():
    print(path)
    current_portkeys = time_series_data[time_series_data['path'] == path]
    date_bucket = []
    for i in current_portkeys.index:
        #pd.to_datetime(current_portkeys['timestamp'][i], unit='s').dt.date
        date_bucket = np.hstack((date_bucket, np.asarray(current_portkeys['timestamp'][i])))
    date_bucket = np.unique(date_bucket)
    f, ax = plt.subplots(figsize=(14, 3))
    for i in current_portkeys.index:
        series = pd.DataFrame(current_portkeys['raw_data'][i], current_portkeys['timestamp'][i])
        series = series[~series.index.duplicated(keep='first')]
        series = series.reindex(date_bucket, fill_value=0).sort_index().mask(series == 0).interpolate()
        series = series[series.index > datetime.date(2020, 1, 1)]
        series['zscore'] = st.zscore(series)
        ax.plot(series['zscore'],label = current_portkeys['node'][i] + "_" + current_portkeys['pm'][i] + "_port_" + str(current_portkeys['port'][i]))
    ax.legend(loc='upper right', shadow=False, fontsize='small')
    ax.set(xlabel = 'date', ylabel= 'z-score')
    fig_name = path +"_plot.png"
    fig_name = fig_name.replace(":", "-")
    plt.savefig(fig_name)
    plt.clf()
    print(path)