#%%

import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

with open('pmvalues_interpolated_filtered_simpleindex.pkl', 'rb') as f:
    time_series_data = pickle.load(f)

#%%

df = time_series_data.iloc[:,4:].T

#%%


maxlag=12

test = 'ssr-chi2test'
data=np.array([ts1,ts2]).reshape(74,2)
def grangers_causality_matrix(df, variables, test = 'ssr_chi2test', verbose=False):

    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(data[[r,c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value

    dataset.columns = [var + '_x' for var in variables]

    dataset.index = [var + '_y' for var in variables]


grangers_causality_matrix(dataset, variables = dataset.columns)

#%%

grangers_causality_matrix(df, variables = df.columns)
