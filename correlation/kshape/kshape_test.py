from kshape.core import kshape, zscore
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# time_series = [[1,2,3,4], [0,1,2,3], [0,1,2,3], [1,2,2,3], [2,3,1,1]]
# cluster_num = 2
# clusters = kshape(zscore(time_series, axis=1), cluster_num)
# print(clusters)


with open('time_series_data.pkl', 'rb') as f:
    data = pickle.load(f)
keys = list(data.keys())
ts_1 = data["NTK552FA_d3f-749c3-fe::AMP-1-13-8_OPIN-OTS"]
f, ax = plt.subplots(figsize=(14, 3))
ax.plot(ts_1, color='g')
ax.set_xticklabels([int(item - 10) for item in ax.get_xticks()])
#plt.show()
ts_1.append(-12)
ts_2 = data["NTK552FA_544-6c8fc-15::AMP-1-2-6_ORL-OTS"]
#ts_2.pop()
f, bx = plt.subplots(figsize=(14, 3))
bx.plot(ts_2, color='g')
bx.set_xticklabels([int(item - 10) for item in bx.get_xticks()])
ts_3 = data["NTK552FA_fa7-fb0e8-a6::AMP-1-2-8_OPIN-OTS"]
ts_3.append(-10)
f, cx = plt.subplots(figsize=(14, 3))
cx.plot(ts_3, color='g')
cx.set_xticklabels([int(item - 10) for item in cx.get_xticks()])
ts_4 = data["NTK552FA_5d1-94c4f-b6::AMP-1-13-6_OPIN-OTS"]
#ts_4.pop()
f, dx = plt.subplots(figsize=(14, 3))
dx.plot(ts_4, color='g')
dx.set_xticklabels([int(item - 10) for item in dx.get_xticks()])
plt.show()
time_series_mat = [ts_1,ts_2,ts_3,ts_4]
cluster_num = 2
clusters = kshape(zscore(time_series_mat, axis=1), cluster_num)
print(clusters)

