

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import mne

from n0_config import *

debug = False



############################
######## LOAD ECG ########
############################


os.chdir(os.path.join(path_prep, sujet, 'sections'))
dirlist_subject = os.listdir()

cond_keep = []
for cond in conditions_allsubjects:

    for file in dirlist_subject:

        if file.find(cond) != -1 : 
            cond_keep.append(cond)
            break

conditions = cond_keep

#### load ecg
band_prep = 'lf'
ecg_allcond = {}
ecg_stim_allcond = {}
for cond in conditions:

    load_i = []
    for session_i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
            load_i.append(session_i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]

    data_ecg = []
    data_ecg_stim = []
    for load_name in load_list:
        raw = mne.io.read_raw_fif(load_name, preload=True)
        srate = int(raw.info['sfreq'])
        chan_list = raw.info['ch_names']
        ecg_i = chan_list.index('ECG')
        stim_i = chan_list.index('ECG_cR')
        ecg = raw.get_data()[ecg_i,:]
        ecg_stim = raw.get_data()[stim_i,:]

        if sujet_ecg_adjust.get(sujet) == 'inverse':
            ecg = ecg*-1
        
        data_ecg.append(ecg)
        data_ecg_stim.append(ecg_stim)

    ecg_allcond[cond] = data_ecg
    ecg_stim_allcond[cond] = data_ecg_stim



################################
######## ECG ANALYSIS ########
################################

def ecg_analysis(cond, session_i):

    #### load cR
    times = np.arange(len(ecg_allcond[cond][session_i]))
    ecg_stim = ecg_stim_allcond[cond][session_i]
    ecg_stim = np.where(ecg_stim == 10, 1, ecg_stim) 
    peaks_dict = {'ECG_R_Peaks' : ecg_stim.astype(int)}
    ecg_peaks = pd.DataFrame(peaks_dict)

    #### verif cR
    ecg = ecg_allcond[cond][session_i]
    ecg_cR = np.where(ecg_stim == 1)[0]
    fig_verif = plt.figure(figsize=(60,40))
    plt.plot(ecg)
    plt.vlines(ecg_cR, ymax=np.max(ecg), ymin=np.min(ecg), colors='r')
    #fig.show()

    #### compute metrics
    hrv_metrics = nk.hrv(ecg_peaks, sampling_rate=srate, show=False)

    total_hrv = np.array([hrv_metrics.iloc[0]['HRV_HF'] + hrv_metrics.iloc[0]['HRV_LF']])
    
    #### export 
    total_hrv = pd.DataFrame(data=total_hrv, index=None, columns=['HRV_Total'])

    hrv_metrics = pd.concat([hrv_metrics,total_hrv],axis=1)
    hrv_metrics.insert(0,'Subject',[sujet_list.index(sujet)])
    hrv_metrics.insert(1,'Name',[sujet])
    hrv_metrics.insert(2,'RDorFR',[cond[:2]])
    hrv_metrics.insert(3,'Condition',[cond])

    col_to_drop = []
    col_hrv = list(hrv_metrics.columns.values) 
    for metric_name in col_hrv :
        if metric_name == 'Subject' or metric_name == 'Name' or metric_name == 'RDorFR' or metric_name == 'Condition':
            continue
        elif (metric_name in hrv_metrics_short_name) == False :
            col_to_drop.append(metric_name)

    hrv_metrics_short = hrv_metrics.copy()
    hrv_metrics_short = hrv_metrics_short.drop(col_to_drop, axis=1)

    return hrv_metrics, hrv_metrics_short, fig_verif



if __name__ == '__main__':

    #### compute hrv
    hrv_metrics_allcond = pd.DataFrame()
    hrv_metrics_allcond_short = pd.DataFrame()
    fig_verif_list= []
    cond_name = []
    for cond in conditions:
        if len(ecg_allcond.get(cond)) == 1:
            hrv_metrics, hrv_metrics_short, fig_verif = ecg_analysis(cond, 0)
            hrv_metrics_allcond = pd.concat([hrv_metrics_allcond, hrv_metrics])
            hrv_metrics_allcond_short = pd.concat([hrv_metrics_allcond_short, hrv_metrics_short])
            fig_verif_list.append(fig_verif)
            cond_name.append(cond + '_1')

        else:
            for session_i in range(len(ecg_allcond.get(cond))):
                hrv_metrics, hrv_metrics_short, fig_verif = ecg_analysis(cond, session_i)
                hrv_metrics_allcond = pd.concat([hrv_metrics_allcond, hrv_metrics])
                hrv_metrics_allcond_short = pd.concat([hrv_metrics_allcond_short, hrv_metrics_short])
                fig_verif_list.append(fig_verif)
                cond_name.append(cond + '_' + str(session_i+1))


    #### save fig
    os.chdir(os.path.join(path_results, sujet, 'HRV', 'fig_verif'))
    for fig_i, fig in enumerate(fig_verif_list):
        fig.savefig(sujet + '_' + cond_name[fig_i] + '.jpeg')

    #### save hrv metrics
    os.chdir(os.path.join(path_results, sujet, 'HRV', 'allcond'))
    hrv_metrics_allcond.to_excel(sujet + '_HRV_Metrics.xlsx')
    hrv_metrics_allcond_short.to_excel(sujet + '_HRV_Metrics_Short.xlsx')

