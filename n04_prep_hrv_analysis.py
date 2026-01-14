

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import mne
import scipy.signal
from bycycle.cyclepoints import find_extrema
import respirationtools

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False



############################
######## LOAD ECG ########
############################

def get_ecg_data(sujet):

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    #### load ecg
    band_prep = 'lf'
    ecg_allcond = {}
    ecg_stim_allcond = {}

    for cond in cond_HRV:

        load_i = []
        for session_i, session_name in enumerate(os.listdir()):
            if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
                load_i.append(session_i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        data_ecg = []
        data_ecg_stim = []

        for load_name_i, load_name in enumerate(load_list):

            raw = mne.io.read_raw_fif(load_name, preload=True)
            srate = int(raw.info['sfreq'])
            chan_list = raw.info['ch_names']
            ecg_i = chan_list.index('ECG')
            stim_i = chan_list.index('ECG_cR')
            ecg = raw.get_data()[ecg_i,:]
            ecg_stim = raw.get_data()[stim_i,:]

            if sujet_ecg_adjust.get(sujet) == 'inverse':
                ecg = ecg*-1

            ecg_allcond[f'{cond}_{load_name_i+1}'] = ecg
            ecg_stim_allcond[f'{cond}_{load_name_i+1}'] = ecg_stim


    return ecg_allcond, ecg_stim_allcond





################################
######## COMPILATION ########
################################

def compute_HRV_fig_df_for_sujet(sujet):

    #### get data
    ecg_allcond, ecg_stim_allcond = get_ecg_data(sujet)
    prms = get_params(sujet, monopol)

    #### compute hrv NK
    df_hrv = pd.DataFrame()
    fig_verif_list= []

    #cond = cond_HRV[0]
    for cond in ecg_allcond:

        for compute_type in ['homemade', 'nk']:

            if compute_type == 'homemade':
                hrv_metrics, fig_list = ecg_analysis_homemade(ecg_allcond[cond], prms['srate'], srate_resample_hrv, fig_token=True)
                fig_verif_list.append(fig_list)
            else:
                hrv_metrics = nk_analysis(ecg_allcond[cond], prms['srate'])

            header_dict = {'sujet' : [sujet], 'cond' : [cond[:-2]], 'session' : [cond[-1:]], 'RDorFR' : [cond[:2]], 'compute_type' : [compute_type]}
            header = pd.DataFrame(header_dict)
            df_hrv_i = pd.concat([header, hrv_metrics], axis=1)
            df_hrv = pd.concat([df_hrv, df_hrv_i], axis=0)


    #### save fig
    os.chdir(os.path.join(path_results, sujet, 'HRV'))

    for cond_i, cond in enumerate(ecg_allcond):

        for fig_type_i, fig_type in enumerate(['RRI', 'PSD', 'poincarre', 'verif', 'dHR']):

            fig_verif_list[cond_i][fig_type_i].savefig(f'{sujet}_{cond}_{fig_type}.jpeg')


    #### save hrv metrics
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_hrv.to_excel(f'{sujet}_df_HRV.xlsx')







################################
######## EXECUTE ######## 
################################



if __name__ == '__main__':

    for sujet in sujet_list_FR_CV:

        print(sujet)

        compute_HRV_fig_df_for_sujet(sujet)



    