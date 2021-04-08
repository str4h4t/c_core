import pickle
from kshape.core import kshape, zscore
import pandas as pd
import numpy as np


def conf_mat(y_act,y_pred):
    # classes = np.concatenate((np.zeros([1, 50]), np.ones([1, 50]), np.ones([1, 50]) * 2),axis=1)
    # ind = np.unravel_index(np.argmax(r, axis=1), r.shape)
    # x = np.asarray(ind[1]).reshape(1, N)
    y_actu = pd.Series(np.asarray(y_act.tolist()).flatten(), name='Actual')
    y_pred = pd.Series(np.asarray(y_pred).flatten(), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    return df_confusion

#def data_plotter(data):


if __name__ == '__main__':
    with open('C://Users//k_mathin//PycharmProjects//Masters//ciena_trials//Kamal//data//vodafone_data_clusters_filtered.pkl', 'rb') as f:
        data_set = pickle.load(f)
    data = []
    for d in data_set['data']:
        data.append(d)
    data = np.asarray(data)
    #data = data[:,:15]
    print(data.shape[0])
    label_data = np.asarray(data_set['osid'])
    labels, levels = pd.factorize(label_data)
    shelves = np.asarray(data_set['shelf'])
    cluster_num = levels.shape[0]
    print(cluster_num)
    clusters = kshape(zscore(data, axis=1), cluster_num)
    #clusters = kshape(data,cluster_num)
    y_pred = []
    for i in range(0,data.shape[0]):
        for j in range(0,cluster_num):
            if i in clusters[j][1]:
                y_pred.append(j)
                continue
    conf = conf_mat(labels,y_pred)

    print(conf_mat(labels,y_pred))
    print("done")