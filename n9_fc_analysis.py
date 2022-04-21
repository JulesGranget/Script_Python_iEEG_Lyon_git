

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib
import pickle

from n4_respi_analysis import analyse_resp

from n0_config import *
from n0bis_analysis_functions import *


debug = False







#######################################
############# ISPC & PLI #############
#######################################

#data = data_tmp
def compute_fc_metrics_mat(band_prep, data, freq, band, cond, session_i, prms):
    
    #### check if already computed
    pli_mat = np.array([0])
    ispc_mat = np.array([0])

    os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'matrix'))
    if os.path.exists(sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1) + '.npy'):
        print('ALREADY COMPUTED : ' + sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1))
        ispc_mat = np.load(sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1) + '.npy')

    os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'matrix'))
    if os.path.exists(sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1) + '.npy'):
        print('ALREADY COMPUTED : ' + sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1))
        pli_mat = np.load(sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1) + '.npy')

    if len(ispc_mat) != 1 and len(pli_mat) != 1:
        return pli_mat, ispc_mat 
    

    #### select wavelet parameters
    wavelets, nfrex = get_wavelets(band_prep, freq)

    data = data[:len(prms['chan_list_ieeg']),:]

    #### compute all convolution
    os.chdir(path_memmap)
    convolutions = np.memmap(sujet + '_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

    print('CONV')

    def convolution_x_wavelets_nchan(nchan):

        if nchan/np.size(data,0) % .25 <= .01:
            print("{:.2f}".format(nchan/len(prms['chan_list_ieeg'])))
        
        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

        x = data[nchan,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan,:,:] = nchan_conv

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan) for nchan in range(np.size(data,0)))

    #### compute metrics
    pli_mat = np.zeros((np.size(data,0),np.size(data,0)))
    ispc_mat = np.zeros((np.size(data,0),np.size(data,0)))

    print('COMPUTE')

    for seed in range(np.size(data,0)) :

        if seed/len(prms['chan_list_ieeg']) % .25 <= .01:
            print("{:.2f}".format(seed/len(prms['chan_list_ieeg'])))

        def compute_ispc_pli(nchan):

            if nchan == seed : 
                return
                
            else :

                # initialize output time-frequency data
                ispc = np.zeros((nfrex))
                pli  = np.zeros((nfrex))

                # compute metrics
                for fi in range(nfrex):
                    
                    as1 = convolutions[seed][fi,:]
                    as2 = convolutions[nchan][fi,:]

                    # collect "eulerized" phase angle differences
                    cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                    
                    # compute ISPC and PLI (and average over trials!)
                    ispc[fi] = np.abs(np.mean(cdd))
                    pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

            # compute mean
            mean_ispc = np.mean(ispc,0)
            mean_pli = np.mean(pli,0)

            return mean_ispc, mean_pli

        compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(nchan) for nchan in range(np.size(data,0)))
        
        for nchan in range(np.size(data,0)) :
                
            if nchan == seed:

                continue

            else:
                    
                ispc_mat[seed,nchan] = compute_ispc_pli_res[nchan][0]
                pli_mat[seed,nchan] = compute_ispc_pli_res[nchan][1]

    #### save matrix
    os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'matrix'))
    np.save(sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1) + '.npy', ispc_mat)

    os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'matrix'))
    np.save(sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1) + '.npy', pli_mat)

    #### supress mmap
    os.chdir(path_memmap)
    os.remove(sujet + '_fc_convolutions.dat')
    
    return pli_mat, ispc_mat



################################
######## LOAD TF & ITPC ########
################################



#session_eeg=0
def compute_pli_ispc_allband(sujet):

    #### get params
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band = 'theta'
        for band in freq_band_list[band_prep_i]:

            if band == 'whole' :

                continue

            else: 

                freq = freq_band_fc_analysis[band]

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, conditions[0]
                #session_i = 0
                for cond_i, cond in enumerate(prms['conditions']) :

                    print(band, cond)

                    if len(respfeatures_allcond[cond]) == 1:

                        data_tmp = load_data(band_prep, cond, 0)
                        pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, data_tmp, freq, band, cond, 0, prms)
                        pli_allcond[cond] = [pli_mat]
                        ispc_allcond[cond] = [ispc_mat]

                    elif len(respfeatures_allcond[cond]) > 1:

                        load_ispc = []
                        load_pli = []

                        for session_i in range(len(respfeatures_allcond[cond])):
                            
                            data_tmp = load_data(band_prep, cond, session_i)
                            pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, data_tmp, freq, band, cond, session_i, prms)
                            load_ispc.append(ispc_mat)
                            load_pli.append(pli_mat)

                        pli_allcond[cond] = load_pli
                        ispc_allcond[cond] = load_ispc

                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond

    #### verif

    if debug == True:

        for band, freq in freq_band_fc_analysis.items():

            for cond_i, cond in enumerate(prms['conditions']) :

                print(band, cond, len(pli_allband[band][cond]))
                print(band, cond, len(ispc_allband[band][cond]))


    print('done')







def compute_TF_ITPC(prms):

    #tf_mode = 'ITPC'
    for tf_mode in ['TF', 'ITPC']:
    
        if tf_mode == 'TF':
            print('######## LOAD TF ########')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'{sujet}_tf_stretch_allcond.pkl')):
                print('ALREADY COMPUTED')
                continue
            
        elif tf_mode == 'ITPC':
            print('######## LOAD ITPC ########')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'{sujet}_itpc_stretch_allcond.pkl')):
                print('ALREADY COMPUTED')
                continue

        #### generate str to search file
        freq_band_str = {}

        for band_prep in band_prep_list:

            freq_band = freq_band_dict_FC[band_prep]

            for band, freq in freq_band.items():
                freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])


        #### load file with reducing to one TF

        tf_stretch_allcond = {}

        for band_prep in band_prep_list:

            tf_stretch_allcond[band_prep] = {}

            for cond in prms['conditions']:

                tf_stretch_allcond[band_prep][cond] = {}

                #### generate file to load
                load_file = []
                for file in os.listdir(): 
                    if file.find(cond) != -1:
                        load_file.append(file)
                    else:
                        continue

                #### impose good order in dict
                for band, freq in freq_band_dict_FC[band_prep].items():
                    tf_stretch_allcond[band_prep][cond][band] = 0

                #### file load
                for file in load_file:

                    for i, (band, freq) in enumerate(freq_band_dict_FC[band_prep].items()):

                        if file.find(freq_band_str[band]) != -1:
                            tf_stretch_allcond[band_prep][cond][band] = np.load(file)
                        else:
                            continue
                            

        #### verif
        for band_prep in band_prep_list:
            for cond in prms['conditions']:
                for band, freq in freq_band_dict_FC[band_prep].items():
                    if tf_stretch_allcond[band_prep][cond][band].shape[0] != len(prms['chan_list_ieeg']) :
                        print('ERROR FREQ BAND : ' + band)
                    
        #### save
        if tf_mode == 'TF':
            with open(f'{sujet}_tf_stretch_allcond.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)
        elif tf_mode == 'ITPC':
            with open(f'{sujet}_itpc_stretch_allcond.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)

    print('done')
    










def get_pli_ispc_allsession(sujet):

    #### get params
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band = 'theta'
        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, conditions[0]
                #session_i = 0
                for cond_i, cond in enumerate(prms['conditions']) :

                    print(band, cond)

                    if len(respfeatures_allcond[cond]) == 1:

                        data_tmp = load_data(band_prep, cond, 0)
                        pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, data_tmp, freq, band, cond, 0, prms)
                        pli_allcond[cond] = [pli_mat]
                        ispc_allcond[cond] = [ispc_mat]

                    elif len(respfeatures_allcond[cond]) > 1:

                        load_ispc = []
                        load_pli = []

                        for session_i in range(len(respfeatures_allcond[cond])):
                            
                            data_tmp = load_data(band_prep, cond, session_i)
                            pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, data_tmp, freq, band, cond, session_i, prms)
                            load_ispc.append(ispc_mat)
                            load_pli.append(pli_mat)

                        pli_allcond[cond] = load_pli
                        ispc_allcond[cond] = load_ispc

                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond


    #### verif conv
    if debug == True:
        for band in pli_allband.keys():
            for cond in pli_allband[band].keys():
                plt.matshow(pli_allband[band][cond][0])
                plt.show()

    #### verif

    if debug == True:
                
        for band_prep in band_prep_list:
            
            for band, freq in freq_band_dict_FC[band_prep].items():

                for cond_i, cond in enumerate(prms['conditions']) :

                    print(band, cond, len(pli_allband[band][cond]))
                    print(band, cond, len(ispc_allband[band][cond]))



    #### reduce to one cond
    #### generate dict to fill
    ispc_allband_reduced = {}
    pli_allband_reduced = {}

    for band, freq in freq_band_fc_analysis.items():

        ispc_allband_reduced[band] = {}
        pli_allband_reduced[band] = {}

        for cond_i, cond in enumerate(prms['conditions']) :

            ispc_allband_reduced[band][cond] = []
            pli_allband_reduced[band][cond] = []

    #### fill
    for band_prep_i, band_prep in enumerate(band_prep_list):

        for band, freq in freq_band_list[band_prep_i].items():

            if band == 'whole' :

                continue

            else:

                for cond_i, cond in enumerate(prms['conditions']) :

                    if len(respfeatures_allcond.get(cond)) == 1:

                        ispc_allband_reduced[band][cond] = ispc_allband[band][cond][0]
                        pli_allband_reduced[band][cond] = pli_allband[band][cond][0]

                    elif len(respfeatures_allcond[cond]) > 1:

                        load_ispc = []
                        load_pli = []

                        for session_i in range(len(respfeatures_allcond[cond])):

                            if session_i == 0 :

                                load_ispc.append(ispc_allband[band][cond][session_i])
                                load_pli.append(pli_allband[band][cond][session_i])

                            else :
                            
                                load_ispc = (load_ispc[0] + ispc_allband.get(band).get(cond)[session_i]) / 2
                                load_pli = (load_pli[0] + pli_allband.get(band).get(cond)[session_i]) / 2

                        pli_allband_reduced.get(band)[cond] = load_pli
                        ispc_allband_reduced.get(band)[cond] = load_ispc




    return pli_allband_reduced, ispc_allband_reduced





########################################################
######## SORTING AND MATRIX MANIPULATIONS ########
########################################################


def get_sorting(df_loca):

    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    index_sorted = df_sorted.index.values
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    chan_name_sorted_no_rep = []
    rep_count = 0
    for i, name_i in enumerate(chan_name_sorted):
        if i == 0:
            chan_name_sorted_no_rep.append(name_i)
            continue
        else:
            if name_i == chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_no_rep.append('')
                rep_count += 1
                continue
            if name_i != chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_no_rep.append(name_i)
                rep_count = 0
                continue

    return df_sorted, index_sorted, chan_name_sorted_no_rep



def sort_mat(mat, index_new):

    mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
    for i_before_sort_r, i_sort_r in enumerate(index_new):
        for i_before_sort_c, i_sort_c in enumerate(index_new):
            mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

    #### verify sorting
    if debug:
        plt.matshow(mat_sorted)
        plt.show()

    return mat_sorted


#mat = ispc_allband_reduced[band][cond]
def get_mat_mean(mat, df_sorted):

    #### extract infos
    index_new = df_sorted.index.values
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    roi_in_data = []
    rep_count = 0
    for i, name_i in enumerate(chan_name_sorted):
        if i == 0:
            roi_in_data.append(name_i)
            continue
        else:
            if name_i == chan_name_sorted[i-(rep_count+1)]:
                rep_count += 1
                continue
            if name_i != chan_name_sorted[i-(rep_count+1)]:
                roi_in_data.append(name_i)
                rep_count = 0
                continue

    #### sort mat
    mat_sorted = sort_mat(mat, index_new)
    
    #### mean mat
    indexes_to_compute = {}
    for roi_i, roi_name in enumerate(roi_in_data):        
        i_to_mean = [i for i, roi in enumerate(chan_name_sorted) if roi == roi_name]
        indexes_to_compute[roi_name] = i_to_mean
        
    mat_mean = np.zeros(( len(roi_in_data), len(roi_in_data) ))
    for roi_i_x, roi_name_x in enumerate(roi_in_data):        
        roi_chunk_dfc = mat_sorted[indexes_to_compute[roi_name_x],:]
        roi_chunk_dfc_mean = np.mean(roi_chunk_dfc, 0)
        coeff_i = []
        for roi_i_y, roi_name_y in enumerate(roi_in_data):
            if roi_name_x == roi_name_y:
                coeff_i.append(0)
                continue
            else:
                coeff_i.append( np.mean(roi_chunk_dfc_mean[indexes_to_compute[roi_name_y]]) )
        coeff_i = np.array(coeff_i)
        mat_mean[roi_i_x,:] = coeff_i

    #### verif
    if debug:
        plt.matshow(mat_mean)
        plt.show()

    return mat_mean




def mat_tresh(mat, percentile_thresh):

    thresh_value = np.percentile(mat.reshape(-1), percentile_thresh)

    for x in range(mat.shape[1]):
        for y in range(mat.shape[1]):
            if mat[x, y] < thresh_value:
                mat[x, y] = 0
            if mat[y, x] < thresh_value:
                mat[y, x] = 0

    #### verif
    if debug:
        plt.matshow(mat)
        plt.show()

    return mat

    


################################
######## SAVE FIGURES ######## 
################################



def save_fig_FC(pli_allband_reduced, ispc_allband_reduced, df_loca, prms):



    print('######## SAVEFIG FC ########')

    df_sorted, index_sorted, chan_name_sorted_no_rep = get_sorting(df_loca)
    roi_in_data = df_sorted.drop_duplicates(subset=['ROI'])['ROI'].values.tolist()
            

    #### count for cond subpolt
    if len(prms['conditions']) == 1:
        nrows, ncols = 0, 1
    elif len(prms['conditions']) == 2:
        nrows, ncols = 0, 2
    elif len(prms['conditions']) == 3:
        nrows, ncols = 0, 3
    elif len(prms['conditions']) == 4:
        nrows, ncols = 2, 3
    elif len(prms['conditions']) == 5:
        nrows, ncols = 2, 3
    elif len(prms['conditions']) == 6:
        nrows, ncols = 2, 3 


    #### identify scale
    scale = {'ispc' : {'min' : {}, 'max' : {}}, 'pli' : {'min' : {}, 'max' : {}}}
    for band, freq in freq_band_fc_analysis.items():

        band_ispc = {'min' : [], 'max' : []}
        band_pli = {'min' : [], 'max' : []}

        for cond_i, cond in enumerate(prms['conditions']):
            mat_to_plot = mat_tresh(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted), percentile_thresh)
            band_ispc['max'].append(np.max(mat_to_plot))
            band_ispc['min'].append(np.min(mat_to_plot))
            
            mat_to_plot = mat_tresh(get_mat_mean(pli_allband_reduced[band][cond], df_sorted), percentile_thresh)
            band_pli['max'].append(np.max(mat_to_plot))
            band_pli['min'].append(np.min(mat_to_plot))

        scale['ispc']['max'][band] = np.max(band_ispc['max'])
        scale['ispc']['min'][band] = np.min(band_ispc['min'])
        scale['pli']['max'][band] = np.max(band_pli['max'])
        scale['pli']['min'][band] = np.min(band_pli['min'])


    #### ISPC

    os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'figures'))

    #band_prep_i, band_prep, nchan, band, freq = 0, 'lf', 0, 'theta', [2, 10]
    for band, freq in freq_band_fc_analysis.items():

        if nrows < 1:
            nrows = 0

        #### graph
        fig = plt.figure(facecolor='black')
        if nrows == 0:
            for cond_i, cond in enumerate(prms['conditions']):
                mat_to_plot = mat_tresh(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted), percentile_thresh)
                mne.viz.plot_connectivity_circle(mat_to_plot, node_names=roi_in_data, n_lines=None, 
                                                title=cond, show=False, padding=7, fig=fig, vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
        else:
            for cond_i, cond in enumerate(prms['conditions']):
                mat_to_plot = mat_tresh(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted), percentile_thresh)
                mne.viz.plot_connectivity_circle(mat_to_plot, node_names=roi_in_data, n_lines=None, 
                                                title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1),
                                                vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
        plt.suptitle('ISPC_' + band, color='w')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        #fig.show()

        fig.savefig(sujet + '_ISPC_' + band + '_graph', dpi = 100)

        plt.close()

        #### matrix
        if len(prms['conditions']) == 1:
            fig, ax = plt.subplots(figsize=(20,10))
        else:
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

        if nrows < 1:
            nrows += 1
        for row_i in range(nrows):
            for col_i in range(ncols):
                if (((col_i+1) + (row_i+1)) > len(prms['conditions'])) and (len(prms['conditions']) != 1 ):
                    continue
                cond = prms['conditions'][col_i + (row_i*(col_i+1))]
                if len(prms['conditions']) == 1:
                    mat_to_plot = mat_tresh(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted), percentile_thresh)
                    ax.matshow(mat_to_plot, vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
                    ax.set_title(cond)
                else :
                    mat_to_plot = mat_tresh(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted), percentile_thresh)
                    ax = axs[row_i, col_i]
                    ax.matshow(mat_to_plot, vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
                    ax.set_title(cond)
                ax.set_yticks(range(len(roi_in_data)))
                ax.set_yticklabels(roi_in_data)
                    
        plt.suptitle('ISPC_' + band)
        #plt.show()
                    
        fig.savefig(sujet + '_ISPC_' + band + '_mat', dpi = 100)

        plt.close()


    #### PLI

    os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'figures'))

    #band_prep_i, band_prep, nchan, band, freq = 0, 'lf', 0, 'theta', [2, 10]
    for band, freq in freq_band_fc_analysis.items():

        if nrows < 1:
            nrows = 0

        fig = plt.figure(facecolor='black')
        if nrows == 0:
            for cond_i, cond in enumerate(prms['conditions']):
                mat_to_plot = mat_tresh(get_mat_mean(pli_allband_reduced[band][cond], df_sorted), percentile_thresh)
                mne.viz.plot_connectivity_circle(mat_to_plot, node_names=roi_in_data, n_lines=None, 
                                                            title=cond, show=False, padding=7, fig=fig,
                                                            vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
        else:
            for cond_i, cond in enumerate(prms['conditions']):
                mat_to_plot = mat_tresh(get_mat_mean(pli_allband_reduced[band][cond], df_sorted), percentile_thresh)
                mne.viz.plot_connectivity_circle(mat_to_plot, node_names=roi_in_data, n_lines=None, 
                                                title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1),
                                                vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
        plt.suptitle('PLI_' + band, color='w')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        #fig.show()

        fig.savefig(sujet + '_PLI_' + band + '_graph', dpi = 100)

        plt.close()

        #### matrix
        if len(prms['conditions']) == 1:
            fig, ax = plt.subplots(figsize=(20,10))
        else:
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

        if nrows < 1:
            nrows += 1
        for row_i in range(nrows):
            for col_i in range(ncols):
                if (((col_i+1) + (row_i+1)) > len(prms['conditions'])) and (len(prms['conditions']) != 1 ):
                    continue
                cond = prms['conditions'][col_i + (row_i*(col_i+1))]
                if len(prms['conditions']) == 1:
                    mat_to_plot = mat_tresh(get_mat_mean(pli_allband_reduced[band][cond], df_sorted), percentile_thresh)
                    ax.matshow(mat_to_plot, vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
                    ax.set_title(cond)
                else :
                    mat_to_plot = mat_tresh(get_mat_mean(pli_allband_reduced[band][cond], df_sorted), percentile_thresh)
                    ax = axs[row_i, col_i]
                    ax.matshow(mat_to_plot, vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
                    ax.set_title(cond)
                ax.set_yticks(range(len(roi_in_data)))
                ax.set_yticklabels(roi_in_data)
                    
        plt.suptitle('PLI_' + band)
        #plt.show()
                    
        fig.savefig(sujet + '_PLI_' + band + '_mat', dpi = 100)

        plt.close()







def save_fig_for_allsession(sujet):

    prms = get_params(sujet)

    df_loca = get_loca_df(sujet)

    pli_allband_reduced, ispc_allband_reduced = get_pli_ispc_allsession(sujet)

    save_fig_FC(pli_allband_reduced, ispc_allband_reduced, df_loca, prms)









################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    #### params
    compute_metrics = False
    plot_fig = True

    #### compute fc metrics
    if compute_metrics:
        #compute_pli_ispc_allband(sujet)
        execute_function_in_slurm_bash('n10_fc_analysis', 'compute_pli_ispc_allband', [sujet])

    #### save fig
    if plot_fig:

        save_fig_for_allsession(sujet)