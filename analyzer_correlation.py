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

def executor(details, result, out, threshold, members = None, boost = False):
    if boost:
        thres = threshold
        members = np.asarray(members)
        for i in range(0,result.__len__()):
            order = result[i].argsort()[:][::-1]
            desc = np.sort(result[i])[::-1]
            max_ind = order[:node_count]
            max_nodes = members[max_ind]
            root = details.loc[members[i]]
            neighbors_pred = details.loc[max_nodes]
            ctr = 0
            for n in neighbors_pred.index:
                ctr +=1
                found = 0
                truth = 0
                if neighbors_pred['node'][n] != root['node']:
                    if out.__len__() == 0:
                        pair = [root['node'], neighbors_pred['node'][n]]
                    else:
                        pair = [root['node'], neighbors_pred['node'][n]]
                        for past in range(0,out.__len__()):
                            if comparator(pair, out[past]['pair']):
                                found = 1
                                break
                    if found == 1:
                        continue
                    elif found == 0:
                        if desc[ctr-1] > thres:
                            if root['track'] == neighbors_pred['track'][n]:
                                truth = 1
                            pair.sort()
                            out.append( {'ind': n, 'root': root['node'], 'neighbor': neighbors_pred['node'][n], 'pair' : pair, 'pair_str': pair[0] + pair[1],
                                         'p_root': root['track'], 'p_neighbor': neighbors_pred['track'][n], 'pm_root': root['pm'],
                                         'pm_neighbor': neighbors_pred['pm'][n],'truth': truth})
                        break
        return out
        print("done")