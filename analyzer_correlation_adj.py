import pandas as pd
import numpy as np
import pickle

node_count = 4

def comparator(arr1, arr2):
    arr1.sort()
    arr2.sort()
    for i in [0,1]:
        if (arr1[i] != arr2[i]):
            return False
    return True

def executor(details, result, adj, members = None, boost = False):
    if boost:
        members = np.asarray(members)
        # indices = member_details.index
        # ctr = 0
        # for ind in indices:
        #     adj_m_pred[ind,indices] = curr_result[ctr]
        #     ctr+=1
        return adj
        print("done")