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

with open('vodafone_data_oct30_cross_osid_mar_19_filtered_interpolated.pkl', 'rb') as f:
    tsd = pickle.load(f)

tsd.t
print("hello")
