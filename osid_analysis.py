import pickle

with open('Data//pmvalues_interpolated_filtered_simpleindex.pkl', 'rb') as f:
    data_set = pickle.load(f)

for osid in data_set['osid'].unique():
    print(data_set.loc[data_set['osid'] == osid]['node'].unique().__len__())
print("hello")