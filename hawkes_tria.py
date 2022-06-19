import numpy as np
from datetime import timezone
from tick.dataset import fetch_hawkes_bund_data
from tick.hawkes import HawkesConditionalLaw
from tick.hawkes import HawkesExpKern as hexp
from tick.hawkes import HawkesSumGaussians as hgauss
from tick.plot import plot_hawkes_kernel_norms

# timestamps_list = fetch_hawkes_bund_data()
import pickle
import numpy as np
import pandas as pd
import correlation.DCCA.dcca_calc as dc
import scipy.stats as st

def z_norm(data):
    z_data = []
    for d in range(0,data.__len__()):
        z_data.append(st.zscore(data[d]))
    return np.asarray(z_data)

with open('pmvalues_interpolated_filtered_port_lvl_0921.pkl', 'rb') as f:
    data_set = pickle.load(f)
# data_set = data_set.loc[data_set['pm'] == 'OPOUT-OTS']
data = np.asarray(data_set.iloc[:, 3:])
# data = z_norm(data)
# data = data[:,50:]
N = data.shape[0]
D = data.shape[1]
mean_array= data.mean(axis=1)
mean_array = np.repeat(mean_array.reshape([N,1]),D,axis=1)
macro_data = np.where(data > mean_array, True, False)
macro_data = np.asarray(macro_data)
t_minus_1 = np.delete(np.concatenate((np.zeros((N,1)), data), axis=1),-1,1)
micro_data = np.where(data > t_minus_1, True, False)
micro_data[:,0] = np.ones([N])
timestamps = data_set.columns[3:]
ts = np.asarray([t.replace(tzinfo=timezone.utc).timestamp() for t in timestamps])
timestamps_list = np.asarray(ts[np.where(micro_data[0])])
# kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))
# hawkes_learner = HawkesConditionalLaw(
#     claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=1,
#     quad_method="log", n_quad=1, min_support=1e-4, max_support=1, n_threads=1)
#
# hawkes_learner.fit(timestamps_list)
#
# plot_hawkes_kernel_norms(hawkes_learner,
#                          node_names=["P_u", "P_d", "T_a", "T_b"])
# import csv
# import datetime
# from tick.hawkes import HawkesEM
# from tick.plot import plot_hawkes_kernels
# data = []
# dates = []
# with open('energydata_complete.csv') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for ind, row in enumerate(spamreader):
#         if ind == 0:
#             var_names = row
#         else:
#             dates.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
#             #line = []
#             #line.extend(np.array([np.float(x) for x in row[1:]]))
#             data.append([np.float(x) for x in row[1:]])
# X_train = [np.array([np.float(x[i]) for i,_ in enumerate(x)]) for x in np.array(data)[:250,1:6]]
# nb_dimensions = len(X_train[1])
# # We use HawkesEM, an estimation of multi-dimensional Hawkes processes
# # based on expectation maximization algorithm
# learner = HawkesEM(nb_dimensions, kernel_size=nb_dimensions**2,
#                     n_threads=1, verbose=True, tol=1e-3)
# # Data has to be a list of numpy arrays containing np.float64 variables
# learner.fit(timestamps_list)
#
# plot_hawkes_kernels(learner)
# timestamps_list = fetch_hawkes_bund_data()
# d_new = []
# for i in range(0,data.__len__()):
#     d_new.append(np.ascontiguousarray(macro_data[i]))
# kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))
# new_feat_1 = []
# new_feat_2 = []
# intensity = []
# for i in range(0,data.__len__()):
#     print(i)
#     # print(data[i].var())
#     # hawkes_learner = HawkesConditionalLaw(
#     #     claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=1,
#     #     quad_method="log", n_quad=10, min_support=1e-4, max_support=1, n_threads=9)
#     hawkes_learner = hgauss(max_mean_gaussian=100)
#     hawkes_learner.fit([np.ascontiguousarray(data[i])])
#     new_feat_1.append(hawkes_learner.kernels[0][0][0])
#     new_feat_1.append(hawkes_learner.kernels[0][0][1])
#     intensity.append(hawkes_learner.mean_intensity[0])
# np.save('feat_1.npy',np.asarray(new_feat_1))
# np.save('feat_2.npy',np.asarray(new_feat_2))
# np.save('intensity.npy',np.asarray(intensity))
# plot_hawkes_kernel_norms(hawkes_learner,
#                          node_names=["P_u", "P_d", "T_a", "T_b"])
# from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
# from pyhawkes.models import DiscreteTimeStandardHawkesModel
# # from pyhawkes
# K = 3
# p = 0.25
# dt_max = 20
# network_hypers = {"p": p, "allow_self_connections": False}
# true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
#     K=K, dt_max=dt_max, network_hypers=network_hypers)
# # true_model = DiscreteTimeStandardHawkesModel(K=K, dt_max=dt_max)
#
# # Generate T time bins of events from the the model
# # S is the TxK event count matrix, R is the TxK rate matrix
# S,R = true_model.generate(T=100)
# # true_model.plot()
#
# # Create the test model, add the event count data, and plot
# # test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
# #     K=K, dt_max=dt_max, network_hypers=network_hypers)
# test_model = DiscreteTimeStandardHawkesModel(K=K, dt_max=dt_max)
# test_model.add_data(S)
# test_model.fit_with_bfgs()
# # fig, handles = test_model.plot(color="#e41a1c")
#
# # Run a Gibbs sampler
# # N_samples = 100
# # lps = []
# # for itr in range(N_samples):
# #     test_model.resample_model()
# #     lps.append(test_model.log_probability())
#
#     # Update plots
#     # test_model.plot(handles=test_handles)
# print("done")

from MHP import MHP
P = MHP()
P.generate_seq(60)

m = np.array([0.2, 0.0, 0.0, 0.0])
a = np.array([[0.1, 0.0, 0.0, 0.0],
              [0.9, 0.0, 0.0, 0.0],
              [0.0, 0.9, 0.0, 0.0],
              [0.0, 0.1, 0.8, 0.0]])
w = 3.1

P = MHP(mu=m, alpha=a, omega=w)
P.generate_seq(60)
P.plot_events()
mhat = np.random.uniform(0,1, size=4)
ahat = np.random.uniform(0,1, size=(4,4))
w = 3.

x = P.EM(ahat, mhat, w)
print("hello")