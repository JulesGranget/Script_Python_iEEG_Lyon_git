

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import networkx as nx
import xarray as xr

import pickle
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







################################
######## LOAD DATA ########
################################




def get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet, monopol):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    if monopol:

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

    else:

        with open(f'allcond_{sujet}_Pxx_bi.pkl', 'rb') as f:
            Pxx_allcond = pickle.load(f)

        with open(f'allcond_{sujet}_Cxy_bi.pkl', 'rb') as f:
            Cxy_allcond = pickle.load(f)

        with open(f'allcond_{sujet}_surrogates_bi.pkl', 'rb') as f:
            surrogates_allcond = pickle.load(f)

        with open(f'allcond_{sujet}_cyclefreq_bi.pkl', 'rb') as f:
            cyclefreq_allcond = pickle.load(f)

        with open(f'allcond_{sujet}_MVL_bi.pkl', 'rb') as f:
            MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond




def get_tf_itpc_stretch_allcond(sujet, tf_mode, monopol):

    source_path = os.getcwd()

    if monopol:

        if tf_mode == 'TF':

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            with open(f'allcond_{sujet}_tf_stretch.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)

        elif tf_mode == 'ITPC':
            
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            with open(f'allcond_{sujet}_itpc_stretch.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)

    else:

        if tf_mode == 'TF':

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            with open(f'allcond_{sujet}_tf_stretch_bi.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)


        elif tf_mode == 'ITPC':
            
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            with open(f'allcond_{sujet}_itpc_stretch_bi.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond



def load_surrogates(sujet, respfeatures_allcond, prms, monopol):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {'Cxy' : {}, 'cyclefreq_lf' : {}, 'cyclefreq_hf' : {}, 'MVL' : {}}

    for cond in prms['conditions']:

        if len(respfeatures_allcond[cond]) == 1:

            if monopol:

                surrogates_allcond['Cxy'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_Coh.npy')]
                surrogates_allcond['cyclefreq_lf'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_cyclefreq_lf.npy')]
                surrogates_allcond['cyclefreq_hf'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_cyclefreq_hf.npy')]
                surrogates_allcond['MVL'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_MVL_lf.npy')]

            else:

                surrogates_allcond['Cxy'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_Coh_bi.npy')]
                surrogates_allcond['cyclefreq_lf'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_cyclefreq_lf_bi.npy')]
                surrogates_allcond['cyclefreq_hf'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_cyclefreq_hf_bi.npy')]
                surrogates_allcond['MVL'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_MVL_lf_bi.npy')]


        elif len(respfeatures_allcond[cond]) > 1:

            data_load = {'Cxy' : [], 'cyclefreq_lf' : [], 'cyclefreq_hf' : [], 'MVL' : []}

            for session_i in range(len(respfeatures_allcond[cond])):

                if monopol:

                    data_load['Cxy'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_Coh.npy'))
                    data_load['cyclefreq_lf'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_lf.npy'))
                    data_load['cyclefreq_hf'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_hf.npy'))
                    data_load['MVL'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_lf.npy'))

                else:

                    data_load['Cxy'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_Coh_bi.npy'))
                    data_load['cyclefreq_lf'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_lf_bi.npy'))
                    data_load['cyclefreq_hf'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_hf_bi.npy'))
                    data_load['MVL'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_lf_bi.npy'))

            surrogates_allcond['Cxy'][cond] = data_load['Cxy']
            surrogates_allcond['cyclefreq_lf'][cond] = data_load['cyclefreq_lf']
            surrogates_allcond['cyclefreq_hf'][cond] = data_load['cyclefreq_hf']
            surrogates_allcond['MVL'][cond] = data_load['MVL']

    return surrogates_allcond












################################
######## Cxy & MVL ########
################################



def export_Cxy_MVL_in_df(sujet, respfeatures_allcond, surrogates_allcond, prms, monopol):

    #### verif computation
    if monopol:

        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_Cxy_MVL.xlsx')):
            print('Cxy MVL : ALREADY COMPUTED')
            return

    else:

        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_Cxy_MVL_bi.xlsx')):
            print('Cxy MVL : ALREADY COMPUTED')
            return

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet, monopol)
    prms = get_params(sujet, monopol)
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet, monopol)
    respi_i = prms['chan_list'].index('nasal')

    #### data count
    data_count = {}

    for cond in prms['conditions']:
        data_count[cond] = len(respfeatures_allcond[cond])

    #### identify chan params
    if sujet[:3] != 'pat':
        if monopol:
            chan_list, chan_list_keep = modify_name(prms['chan_list_ieeg'])
        else:
            chan_list = prms['chan_list_ieeg']
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

                respi_tmp = load_data_sujet(sujet, 'lf', cond, trial_i, monopol)[respi_i,:]
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

    if monopol:
        df_export.to_excel(f'{sujet}_df_Cxy_MVL.xlsx')
    else:
        df_export.to_excel(f'{sujet}_df_Cxy_MVL_bi.xlsx')
        













################################
######## TF & ITPC ########
################################




def export_TF_in_df(sujet, respfeatures_allcond, prms, monopol):

    #### verif computation
    if monopol:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_TF.xlsx')):
            print('TF : ALREADY COMPUTED')
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_TF_bi.xlsx')):
            print('TF : ALREADY COMPUTED')
            return

    #### load prms
    prms = get_params(sujet, monopol)
    df_loca = get_loca_df(sujet, monopol)
    
    #### data count
    data_count = {}

    for cond in prms['conditions']:
        data_count[cond] = len(respfeatures_allcond[cond])

    #### identify chan params
    if sujet[:3] != 'pat':
        if monopol:
            chan_list, chan_list_keep = modify_name(prms['chan_list_ieeg'])
        else:
            chan_list = prms['chan_list_ieeg']    
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

                    data = get_tf_itpc_stretch_allcond(sujet, 'TF', monopol)[band_prep][cond][band][chan_i, :, :]
                    Pxx = np.mean(data, axis=0)
                    Pxx_inspi = np.trapz(Pxx[stretch_point_I[0]:stretch_point_I[1]])
                    Pxx_expi = np.trapz(Pxx[stretch_point_E[0]:stretch_point_E[1]])
                    Pxx_IE = np.trapz(Pxx[stretch_point_IE[0]:stretch_point_IE[1]])
                    Pxx_EI = np.trapz(Pxx[stretch_point_EI[0]:]) + np.trapz(Pxx[:stretch_point_EI[1]])

                    data_export_i =   {'sujet' : [sujet]*4, 'cond' : [cond]*4, 'chan' : [chan_name]*4, 'ROI' : [ROI_i]*4, 'Lobe' : [Lobe_i]*4, 'side' : [side_i]*4, 
                                    'band' : [band]*4, 'phase' : ['inspi', 'expi', 'IE', 'EI'], 'Pxx' : [Pxx_inspi, Pxx_expi, Pxx_IE, Pxx_EI]}
                    df_export_i = pd.DataFrame.from_dict(data_export_i)
                    
                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if monopol:
        df_export.to_excel(f'{sujet}_df_TF.xlsx')
    else:
        df_export.to_excel(f'{sujet}_df_TF_bi.xlsx')









def export_ITPC_in_df(sujet, respfeatures_allcond, prms, monopol):

    #### verif computation
    if monopol:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_ITPC.xlsx')):
            print('ITPC : ALREADY COMPUTED')
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_ITPC_bi.xlsx')):
            print('ITPC : ALREADY COMPUTED')
            return

    #### load prms
    prms = get_params(sujet, monopol)
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet, monopol)
    
    #### data count
    data_count = {}

    for cond in prms['conditions']:
        data_count[cond] = len(respfeatures_allcond[cond])

    #### identify chan params
    if sujet[:3] != 'pat':
        if monopol:
            chan_list, chan_list_keep = modify_name(prms['chan_list_ieeg'])
        else:
            chan_list = prms['chan_list_ieeg']
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

                    data = get_tf_itpc_stretch_allcond(sujet, 'ITPC', monopol)[band_prep][cond][band][chan_i, :, :]
                    Pxx = np.mean(data, axis=0)
                    Pxx_inspi = np.trapz(Pxx[stretch_point_I[0]:stretch_point_I[1]])
                    Pxx_expi = np.trapz(Pxx[stretch_point_E[0]:stretch_point_E[1]])
                    Pxx_IE = np.trapz(Pxx[stretch_point_IE[0]:stretch_point_IE[1]])
                    Pxx_EI = np.trapz(Pxx[stretch_point_EI[0]:]) + np.trapz(Pxx[:stretch_point_EI[1]])

                    data_export_i =   {'sujet' : [sujet]*4, 'cond' : [cond]*4, 'chan' : [chan_name]*4, 'ROI' : [ROI_i]*4, 'Lobe' : [Lobe_i]*4, 'side' : [side_i]*4, 
                                    'band' : [band]*4, 'phase' : ['inspi', 'expi', 'IE', 'EI'], 'Pxx' : [Pxx_inspi, Pxx_expi, Pxx_IE, Pxx_EI]}
                    df_export_i = pd.DataFrame.from_dict(data_export_i)
                    
                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if monopol:
        df_export.to_excel(f'{sujet}_df_ITPC_IE.xlsx')
    else:
        df_export.to_excel(f'{sujet}_df_ITPC_IE_bi.xlsx')



########################################
######## COMPUTE GRAPH METRICS ########
########################################



def get_pair_unique_and_roi_unique(pairs):

    #### pairs unique
    pair_unique = []

    for pair_i in np.unique(pairs):
        if pair_i.split('-')[0] == pair_i.split('-')[-1]:
            continue
        if f"{pair_i.split('-')[-1]}-{pair_i.split('-')[0]}" in pair_unique:
            continue
        if pair_i not in pair_unique:
            pair_unique.append(pair_i)

    pair_unique = np.array(pair_unique)

    #### get roi in data
    roi_in_data = []

    for pair_i in np.unique(pairs):
        if pair_i.split('-')[0] not in roi_in_data:
            roi_in_data.append(pair_i.split('-')[0])

        if pair_i.split('-')[-1] not in roi_in_data:
            roi_in_data.append(pair_i.split('-')[-1])

    roi_in_data = np.array(roi_in_data)

    return pair_unique, roi_in_data





#dfc_data, pairs = xr_graph.loc[cf_metric, :, 'whole', :].data, pairs
def fc_pairs_to_mat(dfc_data, pairs, compute_mode='mean', rscore_computation=False):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)
    
    #### mean across pairs
    dfc_mean_pair = np.zeros(( pair_unique.shape[0], dfc_data.shape[-1] ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_data[pairs == pair_to_find, :]
            x_rev = dfc_data[pairs == pair_to_find_rev, :]

            x_mean = np.vstack([x, x_rev]).mean(axis=0)

            #### identify pair name mean
            try:
                pair_position = np.where(pair_unique == pair_to_find)[0][0]
            except:
                pair_position = np.where(pair_unique == pair_to_find_rev)[0][0]

            dfc_mean_pair[pair_position, :] = x_mean

    #### mean pairs to mat
    #### fill mat
    mat_dfc = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue
            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_mean_pair[pair_unique == pair_to_find, :]
            x_rev = dfc_mean_pair[pair_unique == pair_to_find_rev, :]

            x_mean = np.vstack([x, x_rev]).mean(axis=0)

            if rscore_computation:
                x_mean_rscore = rscore_mat(x_mean)
            else:
                x_mean_rscore = x_mean

            if compute_mode == 'mean':
                val_to_place = x_mean_rscore.mean(0)
            if compute_mode == 'trapz':
                val_to_place = np.trapz(x_mean_rscore, axis=1)

            mat_dfc[x_i, y_i] = val_to_place

    return mat_dfc











def compute_graph_metric_fc(sujet, prms, monopol):

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    #### verif computation
    if monopol:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_graph_DFC.xlsx')):
            print('DFC : ALREADY COMPUTED')
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_graph_DFC_bi.xlsx')):
            print('DFC : ALREADY COMPUTED')
            return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'CPL', 'GE', 'SWN'])

    #### compute
    #cond = 'FR_CV'
    for cond in prms['conditions']:
        #band_prep = 'hf'
        for band_prep in band_prep_list:
            #band, freq = 'l_gamma', [50,80]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:
                    #cf_metric_i, cf_metric = 0, 'ispc'
                    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                        if monopol:
                            xr_graph = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs.nc')
                        else:
                            xr_graph = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')
                        
                        pairs = xr_graph['pairs'].data
                        pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)
                        
                        #### transform dfc into connectivity matrices
                        mat_cf =    {'whole' : fc_pairs_to_mat(xr_graph.loc[cf_metric, :, 'whole', :].data, pairs, compute_mode='mean', rscore_computation=False),
                                    'inspi' : fc_pairs_to_mat(xr_graph.loc[cf_metric, :, 'inspi', :].data, pairs, compute_mode='mean', rscore_computation=False), 
                                    'expi' : fc_pairs_to_mat(xr_graph.loc[cf_metric, :, 'expi', :].data, pairs, compute_mode='mean', rscore_computation=False)}

                        if debug:
                            plt.plot(xr_graph[cf_metric_i, 0, :, :].data.reshape(-1))
                            plt.show()

                            plt.matshow(mat_cf['inspi'])
                            plt.matshow(mat_cf['expi'])
                            plt.show()

                        #respi_phase = 'expi'
                        for respi_phase in ['whole', 'inspi', 'expi']:

                            mat = mat_cf[respi_phase]
                            mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                            
                            if debug:
                                np.sum(mat_values > np.percentile(mat_values, 90))

                                count, bin, fig = plt.hist(mat_values)
                                plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                                plt.show()

                            #### apply thresh
                            for chan_i in range(mat.shape[0]):
                                mat[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, 50))[0]] = 0

                            #### verify that the graph is fully connected
                            chan_i_to_remove = []
                            for chan_i in range(mat.shape[0]):
                                if np.sum(mat[chan_i,:]) == 0:
                                    chan_i_to_remove.append(chan_i)

                            mat_i_mask = [i for i in range(mat.shape[0]) if i not in chan_i_to_remove]

                            if len(chan_i_to_remove) != 0:
                                for row in range(2):
                                    if row == 0:
                                        mat = mat[mat_i_mask,:]
                                    elif row == 1:
                                        mat = mat[:,mat_i_mask]

                            if debug:
                                plt.matshow(mat)
                                plt.show()

                            #### generate graph
                            G = nx.from_numpy_array(mat)
                            if debug:
                                list(G.nodes)
                                list(G.edges)
                            
                            nodes_names = {}
                            for node_i, roi_in_data_i in enumerate(mat_i_mask):
                                nodes_names[node_i] = roi_in_data[roi_in_data_i]
                        
                            nx.relabel_nodes(G, nodes_names, copy=False)
                            
                            if debug:
                                G.nodes.data()
                                nx.draw(G, with_labels=True)
                                plt.show()

                                pos = nx.circular_layout(G)
                                nx.draw(G, pos=pos, with_labels=True)
                                plt.show()

                            node_degree = {}
                            for node_i, node_name in zip(mat_i_mask, roi_in_data[mat_i_mask]):
                                node_degree[node_name] = G.degree[roi_in_data[node_i]]

                            CPL = nx.average_shortest_path_length(G)
                            GE = nx.global_efficiency(G)
                            SWN = nx.omega(G, niter=5, nrand=10, seed=None)

                            data_export_i =    {'sujet' : [sujet], 'cond' : [cond], 'band' : [band], 'metric' : [cf_metric], 'phase' : [respi_phase], 
                                            'CPL' : [CPL], 'GE' : [GE], 'SWN' : [SWN]}
                            df_export_i = pd.DataFrame.from_dict(data_export_i)

                            df_export = pd.concat([df_export, df_export_i])


    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if monopol:
        df_export.to_excel(f'{sujet}_df_graph_DFC.xlsx')
    else:
        df_export.to_excel(f'{sujet}_df_graph_DFC_bi.xlsx')






########################################
######## EXTRACT DFC VALUES ########
########################################


def generate_ROI_pairs():

    pairs_of_interest = np.array([])

    for A in ROI_for_DFC_df:

        for B in ROI_for_DFC_df:

            if A == B:
                continue

            pair_i = f'{A}-{B}'

            pairs_of_interest = np.append(pairs_of_interest, pair_i)

    return pairs_of_interest




#dfc_data, pairs = data_chunk.loc[cf_metric,:,:,:].data, data['pairs'].data
def dfc_pairs_mean(dfc_data, pairs):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)
    
    #### mean across pairs
    dfc_mean_pair = np.zeros(( pair_unique.shape[0], dfc_data.shape[1], dfc_data.shape[-1] ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_data[pairs == pair_to_find, :, :]
            x_rev = dfc_data[pairs == pair_to_find_rev, :, :]

            x_mean = np.vstack([x, x_rev]).mean(axis=0)

            #### identify pair name mean
            try:
                pair_position = np.where(pair_unique == pair_to_find)[0][0]
            except:
                pair_position = np.where(pair_unique == pair_to_find_rev)[0][0]

            dfc_mean_pair[pair_position, :, :] = x_mean

    return dfc_mean_pair, pair_unique





def compute_fc_values(sujet, prms, monopol):

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    #### verif computation
    if monopol:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_DFC.xlsx')):
            print('DFC values : ALREADY COMPUTED')
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_DFC_bi.xlsx')):
            print('DFC values : ALREADY COMPUTED')
            return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'pair', 'value'])

    #### get pairs of interest
    pairs_of_interest = generate_ROI_pairs()

    #### compute
    #cond = 'FR_CV'
    for cond in prms['conditions']:
        #band_prep = 'hf'
        for band_prep in band_prep_list:
            #band, freq = 'l_gamma', [50,80]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:
                    #cf_metric_i, cf_metric = 0, 'ispc'
                    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                        #### extract data
                        if monopol:
                            xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs.nc')
                        else:
                            xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')

                        #### mean across pairs
                        dfc_mean_pair, pairs_unique = dfc_pairs_mean(xr_dfc[cf_metric_i, :, :, :].data, xr_dfc['pairs'].data)

                        #### identify pairs of interest
                        #pair_i, pair_name = 0, pairs_of_interest[0]
                        for pair_i, pair_name in enumerate(pairs_of_interest):

                            pairs_of_interest_i_list = np.where((pairs_unique == pair_name) | (pairs_unique == f"{pair_name.split('-')[1]}-{pair_name.split('-')[0]}"))[0]
                            
                            if pairs_of_interest_i_list.shape[0] == 0:
                                continue

                            for pair_i in pairs_of_interest_i_list:

                                value_list =    [dfc_mean_pair[pair_i, 0, :].mean(),
                                                dfc_mean_pair[pair_i, 1, :].mean(),
                                                dfc_mean_pair[pair_i, 2, :].mean()
                                                ]

                                data_export_i =    {'sujet' : [sujet]*3, 'cond' : [cond]*3, 'band' : [band]*3, 'metric' : [cf_metric]*3, 
                                        'phase' : ['whole', 'inspi', 'expi'], 'pair' : [pairs_unique[pair_i]]*3, 'value' : value_list}

                                df_export_i = pd.DataFrame.from_dict(data_export_i)

                                df_export = pd.concat([df_export, df_export_i])

    #### export
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if monopol:
        df_export.to_excel(f'{sujet}_df_DFC.xlsx')
    else:
        df_export.to_excel(f'{sujet}_df_DFC_bi.xlsx')










########################################
######## COMPILATION FUNCTION ########
########################################




def compilation_export_df(sujet, monopol):

    print(sujet)

    #### load params
    prms = get_params(sujet, monopol)
    respfeatures_allcond = load_respfeatures(sujet)
    surrogates_allcond = load_surrogates(sujet, respfeatures_allcond, prms, monopol)

    # #### export
    print('COMPUTE CXY MVL')
    export_Cxy_MVL_in_df(sujet, respfeatures_allcond, surrogates_allcond, prms, monopol)
    
    print('COMPUTE TF')
    export_TF_in_df(sujet, respfeatures_allcond, prms, monopol)

    print('COMPUTE ITPC')
    export_ITPC_in_df(sujet, respfeatures_allcond, prms, monopol)

    print('COMPUTE GRAPH DFC')
    compute_graph_metric_fc(sujet, prms, monopol)

    print('COMPUTE DFC VALUES')
    compute_fc_values(sujet, prms, monopol)





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #sujet = sujet_list[-1]
    for sujet in sujet_list:

        #monopol = False
        for monopol in [True, False]:
                
            #### export df
            execute_function_in_slurm_bash('n11_res_extract_df', 'compilation_export_df', [sujet, monopol]) 
        
        
