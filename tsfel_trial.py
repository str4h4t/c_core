# import tsfel
import pickle
import numpy as np
import pandas as pd
with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
# data_set = data_set.loc[data_set['pm'] == 'OPOUT-OTS']
data = np.asarray(data_set.iloc[:, 3:])

import warnings
warnings.filterwarnings('ignore')

import tsfel

cfg_file = tsfel.get_features_by_domain()#tsfel.extract_sheet(googleSheet_name)
f = tsfel.time_series_features_extractor(cfg_file, data, fs=1)
statistical_props = f[['0_Area under the curve', '0_Autocorrelation', '0_Centroid', '0_ECDF Percentile_0', '0_ECDF Percentile_1',
                       '0_Entropy', '0_Mean', '0_Mean absolute deviation', '0_Mean absolute diff', '0_Mean diff', '0_Median', '0_Total energy']]
spectral_props = f[['0_Power bandwidth', '0_Root mean square', '0_Signal distance', '0_Skewness', '0_Slope', '0_Spectral decrease', '0_Spectral variation']]
wavelet_props = f[['0_Wavelet absolute mean_0', '0_Wavelet absolute mean_1', '0_Wavelet absolute mean_2', '0_Wavelet absolute mean_3',
                   '0_Wavelet absolute mean_4', '0_Wavelet absolute mean_5', '0_Wavelet absolute mean_6', '0_Wavelet absolute mean_7',
                   '0_Wavelet absolute mean_8', '0_Wavelet energy_0', '0_Wavelet energy_1', '0_Wavelet energy_2', '0_Wavelet energy_3',
                   '0_Wavelet energy_4', '0_Wavelet energy_5', '0_Wavelet energy_6', '0_Wavelet energy_7', '0_Wavelet energy_8',
                   '0_Wavelet standard deviation_0', '0_Wavelet standard deviation_1', '0_Wavelet standard deviation_2',
                   '0_Wavelet standard deviation_3', '0_Wavelet standard deviation_4', '0_Wavelet standard deviation_5',
                   '0_Wavelet standard deviation_6', '0_Wavelet standard deviation_7', '0_Wavelet standard deviation_8',
                   '0_Wavelet variance_0', '0_Wavelet variance_1', '0_Wavelet variance_2', '0_Wavelet variance_3', '0_Wavelet variance_4',
                   '0_Wavelet variance_5', '0_Wavelet variance_6', '0_Wavelet variance_7', '0_Wavelet variance_8'
                   ]]
np.save('tsfresh_features/spectral_props.npy', np.asarray(spectral_props))
np.save('tsfresh_features/wavelet_props.npy', np.asarray(wavelet_props))
np.save('tsfresh_features/statistical_props.npy', np.asarray(statistical_props))
print("all")