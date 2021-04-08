import pickle
from kshapemaster.kshape.core import kshape, zscore
import pandas as pd
import numpy as np


def clusterer(data, k, Boost = True):
    if Boost:
        clusters = kshape(zscore(data, axis=1), k)
        assignments = pd.DataFrame(clusters)
        return assignments

    print("done")