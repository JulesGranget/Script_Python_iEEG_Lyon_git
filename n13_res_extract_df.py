

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import networkx as nx
import xarray as xr

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

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_Cxy_MVL.xlsx')):
        print('Cxy MVL : ALREADY COMPUTED')
        return

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

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_TF_IE.xlsx')):
        print('TF : ALREADY COMPUTED')
        return

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
                    Pxx_inspi = np.trapz(Pxx[0:int(stretch_point_TF*ratio_stretch_TF)])
                    Pxx_expi = np.trapz(Pxx[int(stretch_point_TF*ratio_stretch_TF):])
                    Pxx_IE = np.trapz(Pxx[stretch_point_IE[0]:stretch_point_IE[1]])
                    Pxx_EI = np.trapz(Pxx[stretch_point_EI[0]:]) + np.trapz(Pxx[:stretch_point_EI[1]])

                    data_export_i =   {'sujet' : [sujet]*4, 'cond' : [cond]*4, 'chan' : [chan_name]*4, 'ROI' : [ROI_i]*4, 'Lobe' : [Lobe_i]*4, 'side' : [side_i]*4, 
                                    'band' : [band]*4, 'phase' : ['inspi', 'expi'], 'Pxx' : [Pxx_inspi, Pxx_expi, Pxx_IE, Pxx_EI]}
                    df_export_i = pd.DataFrame.from_dict(data_export_i)
                    
                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_export.to_excel(f'{sujet}_df_TF_IE.xlsx')







########################################
######## COMPUTE GRAPH METRICS ########
########################################


#mat = dfc_data['inspi']
def from_dfc_to_mat_conn_trpz(mat, pairs, roi_in_data):

    #### mean over pairs
    pairs_unique = np.unique(pairs)

    pairs_unique_mat = np.zeros(( pairs_unique.shape[0], mat.shape[1] ))
    #pair_name_i = pairs_unique[0]
    for pair_name_i, pair_name in enumerate(pairs_unique):
        pairs_to_mean = np.where(pairs == pair_name)[0]
        pairs_unique_mat[pair_name_i, :] = np.mean(mat[pairs_to_mean,:], axis=0)

    #### fill mat
    mat_cf = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue
            val_to_place, pair_count = 0, 0
            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            if np.where(pairs_unique == pair_to_find)[0].shape[0] != 0:
                x = mat[np.where(pairs_unique == pair_to_find)[0]]
                val_to_place += np.trapz(x)
                pair_count += 1
            if np.where(pairs_unique == pair_to_find_rev)[0].shape[0] != 0:
                x = mat[np.where(pairs_unique == pair_to_find_rev)[0]]
                val_to_place += np.trapz(x)
                pair_count += 1
            val_to_place /= pair_count

            mat_cf[x_i, y_i] = val_to_place

    if debug:
        plt.matshow(mat_cf)
        plt.show()

    return mat_cf




def compute_graph_metric(sujet):

    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_DFC.xlsx')):
        print('DFC : ALREADY COMPUTED')
        return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'CPL', 'GE', 'SWN'])

    #### compute
    #cond = 'FR_CV'
    for cond in ['FR_CV']:
        #band_prep = 'hf'
        for band_prep in band_prep_list:
            #band, freq = 'l_gamma', [50,80]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:
                    #cf_metric = 'ispc'
                    for cf_metric in ['ispc', 'wpli']:

                        roi_in_data = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_reducedpairs.nc')['x'].data
                        xr_graph = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                        pairs = xr_graph['pairs'].data
                        cf_metric_i = np.where(xr_graph['mat_type'].data == cf_metric)[0]

                        #### separate inspi / expi
                        dfc_data = {'inspi' : xr_graph[cf_metric_i, :, :int(stretch_point_TF*ratio_stretch_TF)].data.reshape(pairs.shape[0], -1), 'expi' : xr_graph[cf_metric_i, :, int(stretch_point_TF*ratio_stretch_TF):].data.reshape(pairs.shape[0], -1)}
                        
                        #### transform dfc into connectivity matrices
                        mat_cf = {'inspi' : from_dfc_to_mat_conn_trpz(dfc_data['inspi'], pairs, roi_in_data), 'expi' : from_dfc_to_mat_conn_trpz(dfc_data['expi'], pairs, roi_in_data)}

                        if debug:
                            plt.plot(xr_graph[cf_metric_i, 0, :].data.reshape(-1))
                            plt.show()

                            plt.matshow(mat_cf['inspi'])
                            plt.matshow(mat_cf['expi'])
                            plt.show()

                        #respi_phase = 'expi'
                        for respi_phase in ['inspi', 'expi']:

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
    df_export.to_excel(f'{sujet}_df_DFC.xlsx')












########################################
######## COMPILATION FUNCTION ########
########################################




def compilation_export_df(sujet):

    #### load params
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
        
    surrogates_allcond = load_surrogates(sujet, respfeatures_allcond, prms)

    #### export
    export_Cxy_MVL_in_df(sujet, respfeatures_allcond, surrogates_allcond, prms)
    export_TF_in_df(sujet, respfeatures_allcond, prms)
    compute_graph_metric(sujet)





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    for sujet in sujet_list_FR_CV:
        
        print(sujet)
        
        #### export df
        compilation_export_df(sujet)
    
    
