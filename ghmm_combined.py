# %%

import numpy as np
import pickle as pkl
import hmmlearn as hmm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# %%

file = open('C://Users//kamal//PycharmProjects//c_core//Data//vodafone_data_oct30_cross_osid_mar_19_filtered_interpolated.pkl', 'rb')
data = pkl.load(file)
data.head()

# %%

exp = data.iloc[0, 6]
exp

# %%

from hmmlearn.hmm import GaussianHMM

'''
def fitHMM(Q, nSamples):
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q,[len(Q),1]))

    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(Q,[len(Q),1]))

    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # find log-likelihood of Gaussian HMM
    logProb = model.score(np.reshape(Q,[len(Q),1]))

    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)

    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states

    return hidden_states, mus, sigmas, P, logProb, samples

'''

# %%


Q = data.iloc[10, 6]

# hidden_states, mus, sigmas, P, logProb, samples = fitHMM(Q, 100)
model = GaussianHMM(n_components=4, n_iter=500).fit(np.reshape(Q, [len(Q), 1]))
hidden_states = model.predict(np.reshape(Q, [len(Q), 1]))
# find parameters of Gaussian HMM
mus1 = np.array(model.means_)
sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1]),
                                    np.diag(model.covars_[2]), np.diag(model.covars_[3])])))
P = np.array(model.transmat_)

# %%

print(model.covars_)
print(sigmas)
model.startprob_


# %%

def map_hmms(data):
    '''
    map the time series to hmms
    '''
    n = 3  # number of hidden states
    # print ("------Doing Baum-welch-------")
    df = pd.DataFrame(columns=['Node', 'StartProb', 'Transition', 'Means', 'Covars'], index=range(data.shape[0]))
    for i in range(data.shape[0]):
        current = data.iloc[i, 6]
        # gmmhmm = GMHMM(n, m, d, a, means, covars, w, pi, init_type='user', verbose=True)
        # gmmhmm.train(current, 1000)
        Q = data.iloc[i, 6]
        model = GaussianHMM(n_components=3, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))
        hidden_states = model.predict(np.reshape(Q, [len(Q), 1]))
        # find parameters of Gaussian HMM
        mus = np.array(model.means_)
        sigmas = np.array(
            np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1]), np.diag(model.covars_[2])])))
        trans = np.array(model.transmat_)
        pi = model.startprob_
        node = data.iloc[i, 1]
        df.loc[i] = pd.Series(
            {'Node': node, 'StartProb': pi, 'Transition': trans, 'Means': mus, 'Covars': model.covars_})
    return (df)


def calc_prob_mat(data, df_res):
    '''calculate the probability/likelihood matrix'''
    n = data.shape[0]
    res = np.empty((n, n))
    for j in range(n):
        Q = data.iloc[j, 6]
        crt_hmm = GaussianHMM(n_components=3, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))
        # find parameters of Gaussian HMM
        mus = np.array(crt_hmm.means_)
        sigmas = np.array(
            np.sqrt(np.array([np.diag(crt_hmm.covars_[0]), np.diag(crt_hmm.covars_[1]), np.diag(crt_hmm.covars_[2])])))
        trans = np.array(crt_hmm.transmat_)
        pi = crt_hmm.startprob_
        for i in range(n):
            crt_obs = data.iloc[i, 6]
            logProb = crt_hmm.score(np.reshape(crt_obs, [len(crt_obs), 1]))
            # logProb = crt_hmm.score(crt_obs)
            res[i, j] = logProb
    return res


# %%

hmm_df = map_hmms(data)
hmm_df.head()

# %%

l_prob_mat = calc_prob_mat(data, hmm_df)

# %%

import math

nb = 0
n = l_prob_mat.shape[0]
for i in range(n):
    for j in range(n):
        if np.isinf(l_prob_mat[i, j]):
            nb += 1
print(nb)
l_prob_mat.mean(axis=0).mean()

# %%

prob_mat = np.exp(l_prob_mat)
prob_mat_norm = prob_mat / prob_mat.sum(axis=0)

# %%

# get rid of zeros in prb_mat_norm
nb = 0
for i in range(n):
    for j in range(n):
        # if nn[i,j] == True:
        #    nb+=1
        if prob_mat_norm[i, j] == 0:
            prob_mat_norm[i, j] = prob_mat_norm[:, j].mean()
            nb += 1

# %%

dist_mat = np.empty_like(prob_mat)
n = prob_mat.shape[0]
for i in range(n):
    for j in range(n):
        t_i = l_prob_mat[:, i]
        t_i_n = prob_mat_norm[:, i]
        t_j = l_prob_mat[:, j]
        t_j_n = prob_mat_norm[:, j]
        dist_mat[i, j] = (t_i_n * (t_i - t_j) + t_j_n * (t_j - t_i)).sum() / 2

# %%

nb = 0
for i in range(n):
    for j in range(n):
        # if nn[i,j] == True:
        #    nb+=1
        if dist_mat[i, j] == 0:
            # dist_mat[i,j] = prob_mat_norm[:,j].mean()
            # print(i,j)
            nb += 1
print(nb)

# %%

import random
import matplotlib.pyplot as plt


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape
    # print (m,n)

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))

    # print ('haha',valid_medoid_inds)

    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)
    # print (rs,cs)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    # print (rs,cs)
    for r, c in zip(rs, cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return (M, C)


# %%

M50, C50 = kMedoids(dist_mat, 50, tmax=100)

# %%

# M5, c5 = kMedoids(dist_mat, 5, tmax=100)
#
# # %%
#
# M10, C10 = kMedoids(dist_mat, 100, tmax=100)

# %%

labels_50 = np.zeros(shape=(1957), dtype=int) - 1
for i in range(50):
    l = list(C50[i])
    for j in l:
        labels_50[j] = int(i)
nb = 0
for i in range(n):
    if labels_50[i] == -1:
        nb += 1
print(nb)

# %%

data['label_50'] = labels_50
data.head()

# %%

d = data.iloc[0, 6].shape[0]
data_ = np.empty([n, d])
for i in range(n):
    data_[i] = data.iloc[i, 6]
print(data.iloc[0, 6])
print(data_[0])

# %%

data_df = pd.DataFrame(data_)
data_df.head()
# cluster_found = labels
# f = plt.figure(figsize = (10,5))
# f = plot_tsne(cons, cluster_found, f)
# plt.show()


# %%

data.head()

# %%

#fig = plt.figure(figsize=(10, 5))
cluster_found_sr = pd.Series(labels_50, name='label')
data = data_df.set_index(cluster_found_sr, append=True)
res = TSNE().fit_transform(np.asarray(data_df))
cluster_values = sorted(data.index.get_level_values('label').unique())
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)
#axarr = fig.add_subplot(1, 1, 1)
# plt.scatter(res[:, 0], res[:, 1], c=data.index.get_level_values('label'), alpha=0.6, )
#
# # plt.show()
# plt.title('TSNE: 5 clusters and 3 hidden states')
# plt.show()
# plt.savefig('figs/tsne_10clusters_3hs.png')

# %%

file = open('clean_extracted_data.pkl', 'rb')
# original = pkl.load(file)
# # original['label_3'] = labels_3
# # original['label_5'] = labels_5
# # original['label_10'] = labels_10
# original['label_50'] = labels_50
# original.head()
#
# # %%
#
# tp_3 = 0
# tp_5 = 0
# tp_10 = 0
# tp_20 = 0
# n = original.shape[0]
# for i in range(n):
#     for j in range(i + 1, n):
#         if (original.iloc[i, 8] == original.iloc[j, 8]) and (original.iloc[i, 0] == original.iloc[j, 0]):
#             tp_3 += 1
#         if (original.iloc[i, 9] == original.iloc[j, 9]) and (original.iloc[i, 0] == original.iloc[j, 0]):
#             tp_5 += 1
#         if (original.iloc[i, 10] == original.iloc[j, 10]) and (original.iloc[i, 0] == original.iloc[j, 0]):
#             tp_10 += 1
#         if (original.iloc[i, 11] == original.iloc[j, 11]) and (original.iloc[i, 0] == original.iloc[j, 0]):
#             tp_20 += 1
#
# # %%
#
# tp_3 = 0
# fn_3 = 0
# tp_5 = 0
# fn_5 = 0
# tp_10 = 0
# fn_10 = 0
# tp_20 = 0
# fn_20 = 0
# n = original.shape[0]
# for i in range(n):
#     for j in range(i + 1, n):
#         if (original.iloc[i, 0] == original.iloc[j, 0]):
#             if (original.iloc[i, 8] == original.iloc[j, 8]):
#                 tp_3 += 1
#             else:
#                 fn_3 += 1
#             if (original.iloc[i, 9] == original.iloc[j, 9]):
#                 tp_5 += 1
#             else:
#                 fn_5 += 1
#             if (original.iloc[i, 10] == original.iloc[j, 10]):
#                 tp_10 += 1
#             else:
#                 fn_10 += 1
#             if (original.iloc[i, 11] == original.iloc[j, 11]):
#                 tp_20 += 1
#             else:
#                 fn_20 += 1
#
# # %%
#
# tn_3 = 0
# fp_3 = 0
# tn_5 = 0
# fp_5 = 0
# tn_10 = 0
# fp_10 = 0
# tn_20 = 0
# fp_20 = 0
# n = original.shape[0]
# for i in range(n):
#     for j in range(i + 1, n):
#         if (original.iloc[i, 0] != original.iloc[j, 0]):
#             if (original.iloc[i, 8] == original.iloc[j, 8]):
#                 fp_3 += 1
#             else:
#                 tn_3 += 1
#             if (original.iloc[i, 9] == original.iloc[j, 9]):
#                 fp_5 += 1
#             else:
#                 tn_5 += 1
#             if (original.iloc[i, 10] == original.iloc[j, 10]):
#                 fp_10 += 1
#             else:
#                 tn_10 += 1
#             if (original.iloc[i, 11] == original.iloc[j, 11]):
#                 fp_20 += 1
#             else:
#                 tn_20 += 1
#
# # %%
#
# acc_3 = (tp_3 + tn_3) / (tp_3 + tn_3 + fp_3 + fn_3)
# prec_3 = tp_3 / (tp_3 + fp_3)
# rec_3 = tp_3 / (tp_3 + fn_3)
# f1_3 = 2 * prec_3 * rec_3 / (prec_3 + rec_3)
#
# # %%
#
# acc_5 = (tp_5 + tn_5) / (tp_5 + tn_5 + fp_5 + fn_5)
# prec_5 = tp_5 / (tp_5 + fp_5)
# rec_5 = tp_3 / (tp_5 + fn_5)
# f1_5 = 2 * prec_5 * rec_5 / (prec_5 + rec_5)
#
# # %%
#
# acc_10 = (tp_10 + tn_10) / (tp_10 + tn_10 + fp_10 + fn_10)
# prec_10 = tp_10 / (tp_10 + fp_10)
# rec_10 = tp_10 / (tp_10 + fn_10)
# f1_10 = 2 * prec_10 * rec_10 / (prec_10 + rec_10)
#
# # %%
#
# acc_20 = (tp_20 + tn_20) / (tp_20 + tn_20 + fp_20 + fn_20)
# prec_20 = tp_20 / (tp_20 + fp_20)
# rec_20 = tp_20 / (tp_20 + fn_20)
# f1_20 = 2 * prec_20 * rec_20 / (prec_20 + rec_20)
#
# # %%
#
# print(f1_3)
# print(f1_5)
# print(f1_10)
# print(f1_20)
#
# # %%
#
# original['label_50'] = labels_50
# original.head()
#
# # %%
#
# tp_100 = 0
# fn_100 = 0
# n = original.shape[0]
# for i in range(n):
#     for j in range(i + 1, n):
#         if (original.iloc[i, 0] == original.iloc[j, 0]):
#             if (original.iloc[i, 12] == original.iloc[j, 12]):
#                 tp_100 += 1
#             else:
#                 fn_100 += 1
#
# fp_100 = 0
# tn_100 = 0
# for i in range(n):
#     for j in range(i + 1, n):
#         if (original.iloc[i, 0] != original.iloc[j, 0]):
#             if (original.iloc[i, 12] == original.iloc[j, 12]):
#                 fp_100 += 1
#             else:
#                 tn_100 += 1
#
# # %%
#
# acc_100 = (tp_100 + tn_100) / (tp_100 + tn_100 + fp_100 + fn_100)
# prec_100 = tp_100 / (tp_100 + fp_100)
# rec_100 = tp_100 / (tp_100 + fn_100)
# f1_100 = 2 * prec_100 * rec_100 / (prec_100 + rec_100)
#
# # %%
#
# print(f1_100)
#
# # %%
#
# tp_300 = 0
# fn_300 = 0
# for i in range(n):
#     for j in range(i + 1, n):
#         if (original.iloc[i, 0] == original.iloc[j, 0]):
#             if (original.iloc[i, 13] == original.iloc[j, 13]):
#                 tp_300 += 1
#             else:
#                 fn_300 += 1

# fp_300 = 0
# tn_300 = 0
# for i in range(n):
#     for j in range(i + 1, n):
#         if (original.iloc[i, 0] != original.iloc[j, 0]):
#             if (original.iloc[i, 13] == original.iloc[j, 13]):
#                 fp_300 += 1
#             else:
#                 tn_300 += 1

# %%

# acc_300 = (tp_300 + tn_300) / (tp_300 + tn_300 + fp_300 + fn_300)
# prec_300 = tp_300 / (tp_300 + fp_300)
# rec_300 = tp_300 / (tp_300 + fn_300)
# f1_300 = 2 * prec_300 * rec_300 / (prec_300 + rec_300)
#
# # %%
#
# print(f1_300)
#
# # %%
#
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# def calc_prob_mat_2(data):
#     n = data.shape[0]
#     res = np.empty((n, n))
#     for j in range(n):
#         Q = data.iloc[j, 6]
#         crt_hmm = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))
#         # find parameters of Gaussian HMM
#         mus = np.array(crt_hmm.means_)
#         sigmas = np.array(np.sqrt(np.array([np.diag(crt_hmm.covars_[0]), np.diag(crt_hmm.covars_[1])])))
#         trans = np.array(crt_hmm.transmat_)
#         pi = crt_hmm.startprob_
#         for i in range(n):
#             crt_obs = data.iloc[i, 6]
#             logProb = crt_hmm.score(np.reshape(crt_obs, [len(crt_obs), 1]))
#             # logProb = crt_hmm.score(crt_obs)
#             res[i, j] = logProb
#     return res
#
#
# # %%
#
# l_prob_mat_2 = calc_prob_mat_2(original)
#
# # %%
#
# import math
#
# nb = 0
# n = l_prob_mat_2.shape[0]
# for i in range(n):
#     for j in range(n):
#         if np.isnan(l_prob_mat_2[i, j]):
#             nb += 1
# print(nb)
# l_prob_mat_2.mean(axis=0).mean()
#
# # %%
#
# prob_mat2 = np.exp(l_prob_mat_2)
# prob_mat_norm2 = prob_mat2 / prob_mat2.sum(axis=0)
# nb = 0
# for i in range(n):
#     for j in range(n):
#         # if nn[i,j] == True:
#         #    nb+=1
#         if prob_mat_norm2[i, j] == 0:
#             # prob_mat_norm2[i,j] = prob_mat_norm[:,j].mean()
#             nb += 1
# nb / (n * n)
#
# # %%
#
# dist_mat2 = np.empty_like(prob_mat2)
# n = prob_mat2.shape[0]
# for i in range(n):
#     for j in range(n):
#         t_i = l_prob_mat_2[:, i]
#         t_i_n = prob_mat_norm2[:, i]
#         t_j = l_prob_mat_2[:, j]
#         t_j_n = prob_mat_norm2[:, j]
#         dist_mat2[i, j] = (t_i_n * (t_i - t_j) + t_j_n * (t_j - t_i)).sum() / 2
#
# # %%
#
# nb = 0
# for i in range(n):
#     for j in range(n):
#         # if nn[i,j] == True:
#         #    nb+=1
#         if dist_mat2[i, j] == 0:
#             # dist_mat[i,j] = prob_mat_norm[:,j].mean()
#             # print(i,j)
#             nb += 1
# print(nb)
#
# # %%
#
# # M300_2, C300_2 = kMedoids(dist_mat2, 300, tmax=100)
# #
# # # %%
# #
# # cluster1_100 = np.empty(shape=(len(C50[0]), 55))
# # j = 0
# # for i in C50[0]:
# #     cluster1_100[j, :] = original.iloc[i, 6]
# #     j += 1
# #
# # # %%
# #
# # x = np.array(range(55))
# for i in range(len(C50[0])):
#     plt.plot(x, cluster1_100[i, :])
# plt.title('cluster 1 of 100')
# plt.show()
#
# # %%
#
# cluster1_10 = np.empty(shape=(len(C[0]), 55))
# j = 0
# for i in C[0]:
#     cluster1_10[j, :] = original.iloc[i, 6]
#     j += 1
# for i in range(len(C[0])):
#     plt.plot(x, cluster1_10[i, :])
# plt.title('cluster 1 of 10')
# plt.show()
#
# # %%
#
# cluster1_300 = np.empty(shape=(len([0]), 55))
# j = 0
# for i in C[0]:
#     cluster1_300[j, :] = original.iloc[i, 6]
#     j += 1
# for i in range(len(C[0])):
#     plt.plot(x, cluster1_300[i, :])
# plt.title('cluster 1 of 300')
# plt.show()
#
# # %%
#
# cluster50_300 = np.empty(shape=(len(C300[49]), 55))
# j = 0
# for i in C300[49]:
#     cluster50_300[j, :] = original.iloc[i, 6]
#     j += 1
# for i in range(len(C300[49])):
#     plt.plot(x, cluster50_300[i, :])
# plt.title('cluster 50 of 300')
# plt.show()
#
# # %%
#
# cluster100_300 = np.empty(shape=(len(C300[99]), 55))
# j = 0
# for i in C300[99]:
#     cluster50_300[j, :] = original.iloc[i, 6]
#     j += 1
# for i in range(len(C300[99])):
#     plt.plot(x, cluster100_300[i, :])
# plt.title('cluster 100 of 300')
# plt.show()

# %%


