

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd

import pickle

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







################################
######## LOAD DATA ########
################################




def get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
        
    with open(f'allcond_{sujet}_Pxx.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_cyclefreq.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_MVL.pkl', 'rb') as f:
        MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond




def get_tf_itpc_stretch_allcond(sujet, tf_mode):

    source_path = os.getcwd()

    if tf_mode == 'TF':

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        with open(f'allcond_{sujet}_tf_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    elif tf_mode == 'ITPC':
        
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        with open(f'allcond_{sujet}_itpc_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond



def load_surrogates(sujet, respfeatures_allcond, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {'Cxy' : {}, 'cyclefreq_lf' : {}, 'cyclefreq_hf' : {}, 'MVL' : {}}

    for cond in prms['conditions']:

        if len(respfeatures_allcond[cond]) == 1:

            surrogates_allcond['Cxy'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_Coh.npy')]
            surrogates_allcond['cyclefreq_lf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_lf.npy')]
            surrogates_allcond['cyclefreq_hf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_hf.npy')]
            surrogates_allcond['MVL'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_MVL_lf.npy')]

        elif len(respfeatures_allcond[cond]) > 1:

            data_load = {'Cxy' : [], 'cyclefreq_lf' : [], 'cyclefreq_hf' : [], 'MVL' : []}

            for session_i in range(len(respfeatures_allcond[cond])):

                data_load['Cxy'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy'))
                data_load['cyclefreq_lf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_lf.npy'))
                data_load['cyclefreq_hf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_hf.npy'))
                data_load['MVL'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_lf.npy'))
            
            surrogates_allcond['Cxy'][cond] = data_load['Cxy']
            surrogates_allcond['cyclefreq_lf'][cond] = data_load['cyclefreq_lf']
            surrogates_allcond['cyclefreq_hf'][cond] = data_load['cyclefreq_hf']
            surrogates_allcond['MVL'][cond] = data_load['MVL']


    return surrogates_allcond












########################################
######## EXPORT DATA IN DF ########
########################################



def export_Cxy_MVL_in_df(sujet, respfeatures_allcond, surrogates_allcond, prms):

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet)
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet)
    respi_i = prms['chan_list'].index('nasal')

    #### data count
    data_count = {}

    for cond in prms['conditions']:
        data_count[cond] = len(respfeatures_allcond[cond])

    #### identify chan params
    if sujet[:3] != 'pat':
        chan_list, chan_list_keep = modify_name(prms['chan_list_ieeg'])
    else:
        chan_list = prms['chan_list_ieeg']

    #### prepare df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'Cxy', 'Cxy_surr', 'MVL', 'MVL_surr'])
    
    #### fill df
    for chan_i, chan_name in enumerate(chan_list):

        print_advancement(chan_i, len(chan_list), steps=[25, 50, 75])

        ROI_i = df_loca['ROI'][df_loca['name'] == chan_name].values[0]
        Lobe_i = df_loca['lobes'][df_loca['name'] == chan_name].values[0]
        if chan_name.find('p') or chan_name.find("'"):
            side_i = 'l'
        else:
            side_i = 'r' 

        for cond in prms['conditions']:

            Cxy_i = 0
            Cxy_surr_i = 0

            MVL_i = 0
            MVL_surr_i = 0

            for trial_i in range(data_count[cond]):

                respi_tmp = load_data_sujet(sujet, 'lf', cond, trial_i)[respi_i,:]
                hzPxx, Pxx = scipy.signal.welch(respi_tmp, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])
                Cxy_i += Cxy_allcond[cond][chan_i,np.argmax(Pxx)]
                Cxy_surr_i += surrogates_allcond['Cxy'][cond][chan_i,np.argmax(Pxx)]
                MVL_i += MVL_allcond[cond][chan_i]
                MVL_surr_i += np.mean(surrogates_allcond['MVL'][cond][chan_i,:])

            Cxy_i /= data_count[cond]
            Cxy_surr_i /= data_count[cond]

            MVL_i /= data_count[cond]
            MVL_surr_i /= data_count[cond]

            data_export_i =   {'sujet' : [sujet], 'cond' : [cond], 'chan' : [chan_name], 'ROI' : [ROI_i], 'Lobe' : [Lobe_i], 'side' : [side_i], 
                            'Cxy' : [Cxy_i], 'Cxy_surr' : [Cxy_surr_i], 'MVL' : [MVL_i], 'MVL_surr' : [MVL_surr_i]}
            df_export_i = pd.DataFrame.from_dict(data_export_i)
            
            df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_export.to_excel(f'{sujet}_df_Cxy_MVL.xlsx')







def export_TF_in_df(sujet, respfeatures_allcond, prms):

    #### load prms
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet)
    
    #### data count
    data_count = {}

    for cond in prms['conditions']:
        data_count[cond] = len(respfeatures_allcond[cond])

    #### identify chan params
    if sujet[:3] != 'pat':
        chan_list, chan_list_keep = modify_name(prms['chan_list_ieeg'])
    else:
        chan_list = prms['chan_list_ieeg']

    #### prepare df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'band', 'phase', 'Pxx'])

    #### fill df
    #chan_i, chan_name = 0, chan_list[0]
    for chan_i, chan_name in enumerate(chan_list):

        print_advancement(chan_i, len(chan_list), steps=[25, 50, 75])

        ROI_i = df_loca['ROI'][df_loca['name'] == chan_name].values[0]
        Lobe_i = df_loca['lobes'][df_loca['name'] == chan_name].values[0]

        if chan_name.find('p') or chan_name.find("'"):
            side_i = 'l'
        else:
            side_i = 'r' 

        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #cond = 'FR_CV'
            for cond in prms['conditions']:
                #band, freq = 'theta', [2, 10]
                for band, freq in freq_band_dict[band_prep].items():

                    data = get_tf_itpc_stretch_allcond(sujet, 'TF')[band_prep][cond][band][chan_i, :, :]
                    Pxx = np.mean(data, axis=0)
                    Pxx_inspi = np.mean(Pxx[0:int(stretch_point_TF*ratio_stretch_TF)])
                    Pxx_expi = np.mean(Pxx[int(stretch_point_TF*ratio_stretch_TF):])

                    data_export_i =   {'sujet' : [sujet]*2, 'cond' : [cond]*2, 'chan' : [chan_name]*2, 'ROI' : [ROI_i]*2, 'Lobe' : [Lobe_i]*2, 'side' : [side_i]*2, 
                                    'band' : [band]*2, 'phase' : ['inspi', 'expi'], 'Pxx' : [Pxx_inspi, Pxx_expi]}
                    df_export_i = pd.DataFrame.from_dict(data_export_i)
                    
                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_export.to_excel(f'{sujet}_df_TF_IE.xlsx')











########################################
######## COMPILATION FUNCTION ########
########################################




def compilation_export_df_allplot(sujet_list):


    return





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

        
    #### export df
    sujet_list = sujet_list_FR_CV
    compilation_export_df_allplot(sujet_list)
    
    
