

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




def load_surrogates(sujet, conditions, monopol):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    #### Cxy
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    surrogates_allcond = {'Cxy' : {}, 'cyclefreq' : {}, 'MVL' : {}}

    for cond in conditions:

        surrogates_Cxy = np.zeros((session_count[cond], len(chan_list_ieeg), len(hzCxy)))
        surrogates_cyclefreq = np.zeros((session_count[cond], 3, len(chan_list_ieeg), stretch_point_surrogates_MVL_Cxy))
        surrogates_MVL = np.zeros(( session_count[cond], len(chan_list_ieeg), stretch_point_surrogates_MVL_Cxy ))

        for session_i in range(session_count[cond]):

            if monopol:

                surrogates_Cxy[session_i,:,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_Coh.npy')[:len(chan_list_ieeg),:]
                surrogates_cyclefreq[session_i,:,:,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_wb.npy')[:,:len(chan_list_ieeg),:]
                surrogates_MVL[session_i,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_wb.npy')[:len(chan_list_ieeg),:]

            else:
                
                surrogates_Cxy[session_i,:,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_Coh_bi.npy')[:len(chan_list_ieeg),:]
                surrogates_cyclefreq[session_i,:,:,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_wb_bi.npy')[:,:len(chan_list_ieeg),:]
                surrogates_MVL[session_i,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_wb_bi.npy')[:len(chan_list_ieeg)]

        #### median
        surrogates_allcond['Cxy'][cond] = np.median(surrogates_Cxy, axis=0)
        surrogates_allcond['cyclefreq'][cond] = np.median(surrogates_cyclefreq, axis=0)
        surrogates_allcond['MVL'][cond] = np.median(surrogates_MVL, axis=0)

    return surrogates_allcond









################################
######## Cxy & MVL ########
################################



def export_Cxy_MVL_in_df(sujet, monopol):

    if sujet in sujet_list:

        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

    else:

        conditions = ['FR_CV']

    #### verif computation
    if monopol:

        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_Cxy_MVL.xlsx')) and os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_Pxx.xlsx')):
            print('Pxx Cxy MVL : ALREADY COMPUTED', flush=True)
            return

    else:

        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_Cxy_MVL_bi.xlsx')) and os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_Pxx_bi.xlsx')):
            print('Pxx Cxy MVL : ALREADY COMPUTED', flush=True)
            return

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    if monopol:  
        with open(f'allcond_{sujet}_metrics.pkl', 'rb') as f:
            metrics_allcond = pickle.load(f)

    else:   
        with open(f'allcond_{sujet}_metrics_bi.pkl', 'rb') as f:
            metrics_allcond = pickle.load(f)

    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)
    surrogates_allcond = load_surrogates(sujet, conditions, monopol)
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet, monopol)
    respi_i = chan_list.index('nasal')

    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### identify chan params
    if sujet[:3] != 'pat':
        if monopol:
            chan_list, chan_list_keep = modify_name(chan_list_ieeg)
        else:
            chan_list = chan_list_ieeg
    else:
        chan_list = chan_list_ieeg

    #### prepare df
    df_export_Cxy_MVL = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'Cxy', 'Cxy_surr', 'MVL', 'MVL_surr'])
    df_export_Pxx = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'Pxx', 'band', 'phase'])
    
    #### fill df
    #chan_i, chan_name = 0, chan_list[0]
    for chan_i, chan_name in enumerate(chan_list):

        print_advancement(chan_i, len(chan_list), steps=[10, 25, 50, 75])

        ROI_i = df_loca['ROI'][df_loca['name'] == chan_name].values[0]
        Lobe_i = df_loca['lobes'][df_loca['name'] == chan_name].values[0]
        if chan_name.find('p') or chan_name.find("'"):
            side_i = 'l'
        else:
            side_i = 'r' 

        #### identify normalize params
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        if monopol:
            tf_raw = np.median(np.load(f'{sujet}_tf_raw_FR_CV.npy')[chan_i,:,:,:], axis=0)
        else:
            tf_raw = np.median(np.load(f'{sujet}_tf_raw_FR_CV_bi.npy')[chan_i,:,:,:], axis=0)

        _mean = tf_raw.mean(axis=1).reshape(-1,1)
        _std = tf_raw.std(axis=1).reshape(-1,1)

        #cond = conditions[0]
        for cond in conditions:
                
            #### Pxx
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            phase_list = ['whole', 'inspi', 'expi']

            if monopol:
                tf_raw = np.median(np.load(f'{sujet}_tf_raw_{cond}.npy')[chan_i,:,:,:], axis=0)
            else:
                tf_raw = np.median(np.load(f'{sujet}_tf_raw_{cond}_bi.npy')[chan_i,:,:,:], axis=0)

            #### normalize
            tf_raw[:] = (tf_raw - _mean) / _std

            #band, freq = 'theta', [4,8]
            for band, freq in freq_band_dict_FC_function['wb'].items():

                frex_sel = (frex >= freq[0]) & (frex <= freq[-1])
                Pxx_whole = np.median(tf_raw[frex_sel,:])
                Pxx_inspi = np.median(tf_raw[frex_sel,:int(stretch_point_TF/2)])
                Pxx_expi = np.median(tf_raw[frex_sel,int(stretch_point_TF/2):])

                data_export_i =   {'sujet' : [sujet]*len(phase_list), 'cond' : [cond]*len(phase_list), 'chan' : [chan_name]*len(phase_list), 
                                    'ROI' : [ROI_i]*len(phase_list), 'Lobe' : [Lobe_i]*len(phase_list), 'side' : [side_i]*len(phase_list), 
                                    'Pxx' : [Pxx_whole, Pxx_inspi, Pxx_expi], 'band' : [band]*len(phase_list), 'phase' : phase_list}
                
                df_export_i = pd.DataFrame.from_dict(data_export_i)
        
                df_export_Pxx = pd.concat([df_export_Pxx, df_export_i])

            #### Cxy MVL
            respi_median = np.array([])
            for session_i in range(session_count[cond]):
                respi_median = np.append(respi_median, respfeatures_allcond[cond][session_i]['cycle_freq'].values)
            respi_median = np.median(respi_median)

            hzCxy_sel = (hzCxy >= respi_median-df_extraction_Cxy) & (hzCxy <= respi_median+df_extraction_Cxy)
            Cxy_i = np.median(metrics_allcond[cond]['Cxy'][chan_i,hzCxy_sel])
            Cxy_surr_i = np.median(surrogates_allcond['Cxy'][cond][chan_i,hzCxy_sel])

            MVL_i = metrics_allcond[cond]['MVL'][chan_i]
            MVL_surr_i = np.percentile(surrogates_allcond['MVL'][cond][chan_i,:], 99)

            data_export_i =   {'sujet' : [sujet], 'cond' : [cond], 'chan' : [chan_name], 'ROI' : [ROI_i], 'Lobe' : [Lobe_i], 'side' : [side_i], 
                            'Cxy' : [Cxy_i], 'Cxy_surr' : [Cxy_surr_i], 'MVL' : [MVL_i], 'MVL_surr' : [MVL_surr_i]}
            
            df_export_i = pd.DataFrame.from_dict(data_export_i)
            
            df_export_Cxy_MVL = pd.concat([df_export_Cxy_MVL, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))

    if monopol:
        df_export_Pxx.to_excel(f'{sujet}_df_Pxx.xlsx')
        df_export_Cxy_MVL.to_excel(f'{sujet}_df_Cxy_MVL.xlsx')
    else:
        df_export_Pxx.to_excel(f'{sujet}_df_Pxx_bi.xlsx')
        df_export_Cxy_MVL.to_excel(f'{sujet}_df_Cxy_MVL_bi.xlsx')
        
    print('done', flush=True)












################################
######## TF & ITPC ########
################################




def export_ITPC_in_df(sujet, respfeatures_allcond, prms, monopol):

    #### verif computation
    if monopol:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_ITPC.xlsx')):
            print('ITPC : ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_ITPC_bi.xlsx')):
            print('ITPC : ALREADY COMPUTED', flush=True)
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
            print('FC : ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_graph_DFC_bi.xlsx')):
            print('FC : ALREADY COMPUTED', flush=True)
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

            x_median = np.median(np.vstack([x, x_rev]), axis=0)

            #### identify pair name mean
            try:
                pair_position = np.where(pair_unique == pair_to_find)[0][0]
            except:
                pair_position = np.where(pair_unique == pair_to_find_rev)[0][0]

            dfc_mean_pair[pair_position, :, :] = x_median

    return dfc_mean_pair, pair_unique





def compute_fc_values(sujet, monopol):

    if sujet in sujet_list:

        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

    else:

        conditions = ['FR_CV']

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    #### verif computation
    if monopol:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_FC.xlsx')):
            print('FC values : ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_FC_bi.xlsx')):
            print('FC values : ALREADY COMPUTED', flush=True)
            return
        
    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'pair', 'value'])

    #### get pairs of interest
    pairs_of_interest = generate_ROI_pairs()

    #### compute
    #cond = 'FR_CV'
    for cond in conditions:

        #band, freq = 'theta', [4,8]
        for band, freq in freq_band_dict_FC_function['wb'].items():

            #cf_metric_i, cf_metric = 0, 'ispc'
            for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                #### extract data
                if monopol:
                    xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{cond}_allpairs.nc')
                else:
                    xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{cond}_allpairs_bi.nc')

                #### sel freq
                band_sel = (frex >= freq[0]) & (frex <= freq[-1])

                #### mean across pairs
                dfc_mean_pair, pairs_unique = dfc_pairs_mean(xr_dfc[cf_metric_i, :, :, band_sel].data, xr_dfc['pairs'].data)

                #### identify pairs of interest
                #pair_i, pair_name = 0, pairs_of_interest[0]
                for pair_i, pair_name in enumerate(pairs_of_interest):

                    pairs_of_interest_i_list = np.where((pairs_unique == pair_name) | (pairs_unique == f"{pair_name.split('-')[1]}-{pair_name.split('-')[0]}"))[0]
                    
                    if pairs_of_interest_i_list.shape[0] == 0:
                        continue

                    for pair_i in pairs_of_interest_i_list:

                        value_list =    [np.median(dfc_mean_pair[pair_i, 0, :]),
                                        np.median(dfc_mean_pair[pair_i, 1, :]),
                                        np.median(dfc_mean_pair[pair_i, 2, :])
                                        ]

                        data_export_i =    {'sujet' : [sujet]*3, 'cond' : [cond]*3, 'band' : [band]*3, 'metric' : [cf_metric]*3, 
                                'phase' : ['whole', 'inspi', 'expi'], 'pair' : [pairs_unique[pair_i]]*3, 'value' : value_list}

                        df_export_i = pd.DataFrame.from_dict(data_export_i)

                        df_export = pd.concat([df_export, df_export_i])

    #### export
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if monopol:
        df_export.to_excel(f'{sujet}_df_FC.xlsx')
    else:
        df_export.to_excel(f'{sujet}_df_FC_bi.xlsx')










########################################
######## COMPILATION FUNCTION ########
########################################




def compilation_export_df(sujet, monopol):

    # #### export
    print(f'COMPUTE Pxx Cxy MVL {sujet} {monopol}', flush=True)
    export_Cxy_MVL_in_df(sujet, monopol)

    # print('COMPUTE ITPC', flush=True)
    # export_ITPC_in_df(sujet, monopol)

    # print('COMPUTE GRAPH DFC', flush=True)
    # compute_graph_metric_fc(sujet, monopol)

    print(f'COMPUTE DFC VALUES {sujet} {monopol}', flush=True)
    compute_fc_values(sujet, monopol)





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #sujet = sujet_list_FR_CV[3]
    for sujet in sujet_list_FR_CV:

        #monopol = True
        for monopol in [True, False]:
                
            #### export df
            # compilation_export_df(sujet, monopol)
            execute_function_in_slurm_bash_mem_choice('n11_res_extract_df', 'compilation_export_df', [sujet, monopol], '20G') 
        
        
