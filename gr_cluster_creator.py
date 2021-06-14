import pandas as pd
import numpy as np


causality_matrix = pd.read_csv('grangers_causality_matrix.csv')
c_mat = np.where(causality_matrix < 0.05, 1, 0)