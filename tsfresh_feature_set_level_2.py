import pandas as pd
import numpy as np

f_data = pd.read_csv('vodafone_raw_port_lvl_0921_extracted_features.csv')
f_data = f_data.dropna(axis='columns')
f_data = f_data.drop(columns = 'Unnamed: 0')
agg_features = ['value__variance', 'value__standard_deviation', 'value__cid_ce__normalize_False', 'value__spkt_welch_density__coeff_2', 'value__cid_ce__normalize_False',
                'value__autocorrelation__lag_2', 'value__approximate_entropy__m_2__r_0.7', 'value__approximate_entropy__m_2__r_0.9', 'value__linear_trend__attr_"intercept"',
                'value__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"', 'value__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
                'value__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"', 'value__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
                'value__permutation_entropy__dimension_3__tau_1', 'value__permutation_entropy__dimension_4__tau_1', 'value__permutation_entropy__dimension_5__tau_1',
                'value__permutation_entropy__dimension_6__tau_1', 'value__permutation_entropy__dimension_7__tau_1']
agg_data = f_data[agg_features]
agg_data.to_csv('agg_features_vodafone_raw_2310.csv')
for col in f_data.columns:
    count = (f_data[col] == 1).sum() + (f_data[col] == 0).sum()
    if count == f_data.shape[0]:
        f_data = f_data.drop(columns =col)

fft_cols = [col for col in f_data.columns if 'fft' in col]
fft_data = f_data[fft_cols]
for i in range(4):
    del fft_data[fft_data.columns.values[-1]]
fft_data.to_csv('fft_complete_vodafone_raw_2310.csv')
angle_cols = [col for col in fft_data.columns if 'angle' in col]
abs_cols = [col for col in fft_data.columns if 'abs' in col]
fft_data_angle = fft_data[angle_cols]
fft_data_abs = fft_data[abs_cols]
fft_data_angle.to_csv('fft_angle_vodafone_raw_2310.csv')
fft_data_abs.to_csv('fft_abs_vodafone_raw_2310.csv')
print("done")