import numpy as np
import pandas as pd
import pickle
#from pandas.compat import StringIO

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set_1 = pickle.load(f)
with open('pmvalues_interpolated_complete_vodafone_11000.pkl', 'rb') as f:
    data_set_2 = pickle.load(f)

print("done")