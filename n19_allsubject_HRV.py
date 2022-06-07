


import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

from n0_config_params import *





########################
######## PARMS ########
########################

metrics = 'homemade'
#metrics = 'neurokit'

save_table = True
savefig = True
pvalue = 0.05

# patients = ['TREt','CHEe','GOBc','MUGa','MAZm']
patients = ['TREt','GOBc','MAZm']








################
#### STATS ####
################


### Make DataFrame
df = pd.DataFrame()
for patient in patients:
    if metrics == 'homemade':
        file = os.path.join(path_results, patient, 'HRV', patient + '_HRV_Metrics_homemade_mean_cond.xlsx')
    elif metrics == 'neurokit':
        file = os.path.join(path_results, patient, 'HRV', patient + '_HRV_Metrics_Short_mean_cond.xlsx')
    data = pd.read_excel(file).drop(columns = 'Unnamed: 0')
    df = df.append(data)

### Defines a list of dependent variables on which test rm_anova

list_dv = list(data.columns.values)[4::]

### Loop rm_anova to assess effect of Resp Condition on dependent variables

for i , dv in enumerate(list_dv):
    stats_rm = pg.rm_anova(data=df, dv=dv, within='Condition', subject='Subject', correction='auto', detailed=False, effsize='np2')
    stats_rm.insert(0,'DepVar', dv)
    if i == 0:
        anov_rm = stats_rm
    else:
        anov_rm = anov_rm.append(stats_rm, ignore_index = True)

if save_table:
    with pd.ExcelWriter(os.path.join(path_results, 'allplot', 'HRV', 'anov_rm_short_unfiltered.xlsx')) as writer:
        anov_rm.to_excel(writer)

### Set results filtered with pvalue < 0.05

cond = anov_rm['p-unc'] <= pvalue
anov_rm_filtered = anov_rm[cond]

if save_table:
    with pd.ExcelWriter(os.path.join(path_results, 'allplot', 'HRV', 'anov_rm_short_filtered.xlsx')) as writer:
        anov_rm_filtered.to_excel(writer)

### Post-hoc ttest pairwise : set unfiltered and filtered arrays based on pvalue < 0.05

if anov_rm.shape[0] == 0:
    print('Rien de significatif !')
else:
    list_DV_signi = anov_rm_filtered['DepVar'].values

    for i, dv_signi in enumerate(list_DV_signi):
        ttest = pg.pairwise_ttests(data=df, dv=dv_signi, within='Condition', subject='Subject', effsize = 'cohen', return_desc = True, correction = True)
        ttest.insert(0,'Metric',dv_signi)
        ttest_filtered = ttest[ttest['p-unc'] <= pvalue]
        
        if i == 0:
            ttest_pairwise = ttest
            ttest_pairwise_filtered = ttest_filtered
        else:
            ttest_pairwise = ttest_pairwise.append(ttest)
            ttest_pairwise_filtered = ttest_pairwise_filtered.append(ttest_filtered)
            
list_dv_signi_ttest = list(ttest_pairwise_filtered.groupby('Metric').mean().index)


if save_table:
    with pd.ExcelWriter(os.path.join(path_results, 'allplot', 'HRV', 'ttest_pairwise_short_unfiltered.xlsx')) as writer:
        ttest_pairwise.to_excel(writer)
    with pd.ExcelWriter(os.path.join(path_results, 'allplot', 'HRV', 'ttest_pairwise_short_filtered.xlsx')) as writer:
        ttest_pairwise_filtered.to_excel(writer)

### Scatterplots on all dependent variables

for dv in list_dv:
    if dv in list_dv_signi_ttest:
        title = str(ttest_pairwise_filtered.set_index('Metric').loc[dv,['A','B']].values)
    else:
        title = 'No significant result in ttest_pairwise'
    fig, ax = plt.subplots(figsize = [25,10])
    sns.scatterplot(data=df, x='Condition', y=dv, hue = 'Condition', ax = ax).set_title(title)
    if savefig:
        plt.savefig(os.path.join(path_results, 'allplot', 'HRV', f'scatter_{dv}.png'))
    #plt.show()

