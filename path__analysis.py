import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_pickle('pmvalues_interpolated_complete_vodafone_11000.pkl')
topo = pd.read_pickle('topo_connections_port_lvl_0921.pkl')
for row in topo.iterrows():
    string_1 = row[1][0].split('_')[0]
    string_2 = row[1][1].split('_')[0]
    set_1 = data.filter(regex = string_1)
    set_2 = data.filter(regex = string_2)
    frame = pd.concat([set_1, set_2], axis=1)
    correlation_kendall = frame.corr(method='kendall')
    sns.heatmap(correlation_kendall, annot=True)
    plt.savefig('temp.png')
    print("done")
