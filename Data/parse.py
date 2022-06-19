import pandas as pd
import pickle
import glob
from tqdm import tqdm


## read the nh_files
print('(step 1/7) reading files')

nh_files = glob.glob('sx/results_anonymised/df_nhresult*/*.csv')
list_nh_files = []
for file in tqdm(nh_files):
    df = pd.read_csv(file)
    list_nh_files.append(df)
df = pd.concat(list_nh_files, ignore_index=True)

## filter by power measures
print('(step 2/7) filter power measures')

power_pm = ['OCH-OPR', 'OPR-OTS', 'OPIN-OTS',  'OCH-OPT', 'OPOUT-OTS', 'OPT-OTS']
df = df.loc[df.pm.isin(power_pm)].reset_index()

## construct the basic dataframe
print('(step 3/7) getting the info')
df['fac'] = df.port_key_anonymised.str.split('::', expand=True)[1].str.split('-', expand=True)[0]
df['pm'] = df['pm'].str.replace('_','-')
df['tid_shelf_slot_port_fac_pm'] = df['mename_anonymised']+'_'+df['shelf'].astype(str)+'_'+df['slot'].astype(str)+'_'+df['port'].astype(str)+'_'+df['fac']+'_'+df['pm']
df = df[['tid_shelf_slot_port_fac_pm', 'pmtime', 'pmvalue', 'riskvalue']].groupby(['tid_shelf_slot_port_fac_pm']).agg(lambda x: list(x))

## 
print('(step 4/7) grouping the info')
list_pmvalues = []
list_riskvalues = []

for ts in tqdm(df.iterrows(), total=df.shape[0]):
    list_pmvalues.append(pd.DataFrame({'time':pd.to_datetime(ts[1].pmtime, unit='s').normalize(), ts[0]:ts[1].pmvalue}).groupby('time').mean())
    list_riskvalues.append(pd.DataFrame({'time':pd.to_datetime(ts[1].pmtime, unit='s').normalize(), ts[0]:ts[1].riskvalue}).groupby('time').mean())

print('(step 5/7) filtering by constraints')    
pmvalues = pd.concat(list_pmvalues, axis=1)
riskvalues = pd.concat(list_riskvalues, axis=1)

riskvalues = riskvalues.where(pmvalues > -100)
pmvalues = pmvalues.where(pmvalues > -100)

riskvalues = riskvalues.loc[:, (pmvalues.isna().sum(axis=0) < 10) & (pmvalues.std(0) > 0.05)]
pmvalues = pmvalues.loc[:, (pmvalues.isna().sum(axis=0) < 10) & (pmvalues.std(0) > 0.05)]

print('(step 6/7) interpolating')
#pd.date_range('2019-08-20', '2019-10-14').difference(pmvalues.index)
pmvalues = pmvalues.reindex(pd.date_range('2019-08-20', '2019-10-14'))
riskvalues = riskvalues.reindex(pd.date_range('2019-08-20', '2019-10-14'))
pmvalues.interpolate(inplace=True)
riskvalues.interpolate(inplace=True)

riskvalues = riskvalues.loc[:, pmvalues.isna().sum() == 0]
pmvalues = pmvalues.loc[:, pmvalues.isna().sum() == 0]

pmvalues = pmvalues.transpose()
riskvalues = riskvalues.transpose()

print('(step 7/7) exporting')
info = pmvalues.index.to_series().str.split('_', expand=True)
pd.concat([pd.DataFrame({'node': info[0]+'_'+info[1]+'_'+info[2], 'port': info[3], 'fac_pm': info[4]+'_'+info[5]}), pmvalues], axis=1).to_pickle('pmvalues.pkl')

info = riskvalues.index.to_series().str.split('_', expand=True)
pd.concat([pd.DataFrame({'node': info[0]+'_'+info[1]+'_'+info[2], 'port': info[3], 'fac_pm': info[4]+'_'+info[5]}), riskvalues], axis=1).to_pickle('riskvalues.pkl')

print('file exported succefully')
print('the preprocessed data contains {} timeseries with {} timestamps'.format(pmvalues.shape[0], pmvalues.shape[1]))
