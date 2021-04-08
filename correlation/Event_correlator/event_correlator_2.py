import pandas as pd
import numpy as np
import pickle
import datetime
import random
from sklearn.preprocessing import MinMaxScaler

def crosscorr(datax, datay, meth, lag=0, wrap = False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag), method = meth)


def correlator(seq_1, seq_2):
    #comp = pd.concat([seq_2, pd.Series(seq_1, index=seq_2.index)], axis=1)
    score = 0
    seq_1 = pd.Series(seq_1, index=seq_2.index)
    spikes = seq_1.index[seq_1 == True].tolist()
    for spike in spikes:
        front_seq = seq_2[seq_2.index < spike]
        rear_seq = seq_2[seq_2.index > spike]
        fs_len = front_seq.__len__()
        rs_len = rear_seq.__len__()
        K = range(4,10)
        corr = []
        for k in K:
            theta = []
            if (fs_len >= (2*k)) & (rs_len >= (2 * k)):
                front_k = front_seq[-k:]
                r_loc_front = random.sample(range(0, fs_len - k), ((fs_len - k) / 2).__round__())
                [theta.append(front_seq[i:i + k]) for i in r_loc_front]
                rear_k = rear_seq[:k]
                r_loc_rear = random.sample(range(0, rs_len - k), ((rs_len - k) / 2).__round__())
                [theta.append(rear_seq[i:i + k]) for i in r_loc_rear]
                for t in theta:
                    if ~(np.asarray(t) == np.asarray(front_k)).all():
                        corr.append(np.nanmax([crosscorr(front_k, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
                    if ~(np.asarray(t) == np.asarray(rear_k)).all():
                        corr.append(np.nanmax([crosscorr(rear_k, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
            elif fs_len >= (2*k):
                front_k = front_seq[-k:]
                r_loc_front = random.sample(range(0, fs_len - k), ((fs_len - k) / 2).__round__())
                [theta.append(front_seq[i:i + k]) for i in r_loc_front]
                if rs_len >= k:
                    rear_k = rear_seq[:k]
                    for t in theta:
                        if ~(np.asarray(t) == np.asarray(front_k)).all() or t.std():
                            corr.append(np.nanmax([crosscorr(front_k, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
                        if ~(np.asarray(t) == np.asarray(rear_k)).all():
                            corr.append(np.nanmax([crosscorr(rear_k, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
                else:
                    for t in theta:
                        if ~(np.asarray(t) == np.asarray(front_k)).all():
                            corr.append(np.nanmax([crosscorr(front_seq, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
            elif rs_len >= (2 * k):
                rear_k = rear_seq[:k]
                r_loc_rear = random.sample(range(0, rs_len - k), ((rs_len - k) / 2).__round__())
                [theta.append(rear_seq[i:i + k]) for i in r_loc_rear]
                if fs_len >= k:
                    front_k = front_seq[-k:]
                    for t in theta:
                        if ~(np.asarray(t) == np.asarray(front_k)).all():
                            corr.append(np.nanmax([crosscorr(front_k, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
                        if ~(np.asarray(t) == np.asarray(rear_k)).all():
                            corr.append(np.nanmax([crosscorr(rear_k, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
                else:
                    for t in theta:
                        if ~(np.asarray(t) == np.asarray(rear_k)).all():
                            corr.append(np.nanmax([crosscorr(rear_k, t, 'pearson', lag) for lag in range(-(k/2).__round__(), (k/2).__round__())]))
            if ((abs(np.asarray(corr))>0.5).sum()<(corr.__len__()/2)) and (np.isnan(corr).sum()<(corr.__len__()/4)):
                score += 1
    if score > 0:
        return True
    else:
        return False


with open('C:\\Users\\k_mathin\\PycharmProjects\\Masters\\ciena_trials\\Kamal\\data\\vodafone_updated_data_filtered_no_extrapolated.pkl', 'rb') as f:
    time_series_data = pickle.load(f)

keys = time_series_data
ctr = 0
tp = 0
fp = 0
tn = 0
fn = 0
for index_1 in range((time_series_data.__len__()/4).__round__(),(time_series_data.__len__()/2).__round__()):
    series_1 = pd.DataFrame(time_series_data['data'][index_1], time_series_data['timestamp'][index_1])
    series_1 = series_1[~series_1.index.duplicated(keep='first')]
    series_1 = series_1[series_1.index > datetime.date(2020,1,1)]
    for index_2 in range(0, time_series_data.__len__()):
        ctr += 1
        #print(ctr)
        if index_1 != index_2:
            series_2 = pd.DataFrame(time_series_data['data'][index_2], time_series_data['timestamp'][index_2])
            series_2 = series_2[~series_2.index.duplicated(keep='first')]
            series_2 = series_2[series_2.index > datetime.date(2020,1,1)]
            unique_indices = np.unique(np.hstack((series_1.index,series_2.index)))
            #print(unique_indices.__len__())
            series_1 = series_1.reindex(unique_indices, fill_value=0).sort_index().mask(series_1==0).interpolate()
            series_2 = series_2.reindex(unique_indices, fill_value=0).sort_index().mask(series_1==0).interpolate()
            l_limit = series_1[0].mean() - 3 * series_1[0].std()
            u_limit = series_1[0].mean() + 3 * series_1[0].std()
            event_series = [sig <= l_limit or sig >= u_limit for sig in series_1[0]]
            if np.asarray(event_series).mean() == 0:
                continue
            correlation = correlator(event_series,series_2[0])
            if time_series_data['osid'][index_1] == time_series_data['osid'][index_2]:
                if correlation:
                    tp += 1
                else:
                    fn += 1
            else:
                if correlation:
                    fp += 1
                else:
                    tn += 1
            print("Counter: ",ctr)
            print("True Positive: ", tp)
            print("False Positive: ", fp)
            print("True Negative: ", tn)
            print("False Negative: ", fn)
print("True Positive: ", tp)
print("False Positive: ", fp)
print("True Negative: ", fn)
print("False Negative: ", fn)