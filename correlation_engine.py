import pandas as pd
import numpy as np
import clustering.kshape_cluster as ks
import pickle
import correlation.basic.basic_3 as b3
#from correlation.DTW import dtw_calc
from correlation.DCCA import dcca_calc

class correlation_engine:
    result = 0
    def __init__(self, data, model_name, param = None, boost = False):
        self.data = data
        self.model_name = model_name
        self.param = param
        self.boost = boost

    def kshape(self):
        k = int(self.data.__len__() / 20)
        assignments = ks.clusterer(self.data, k, self.boost)
        return assignments

    def basic_cor(self):
        result = b3.executor( self.data, self.model_name, self.param, self.boost)
        return result

    def dcca(self):
        return dcca_calc.executor(self.data, self.param)

    def pdcca(self):
        return self.result

    # def dtw(self):
    #     dtw_instance = dtw_calc.dtw_calc(self.data, self.param, self.boost)
    #     result = dtw_instance.executor()
    #     return result

    def execute(self):
        if self.model_name == "kshape":
            return self.kshape()
        if self.model_name == "pearson" or self.model_name == "spearman" or self.model_name == "kendall" or self.model_name == "all":
            return self.basic_cor()
        if self.model_name == "dtw":
            return self.dtw()
        if self.model_name == "dcca":
            return self.dcca()

        print("hello")

    

