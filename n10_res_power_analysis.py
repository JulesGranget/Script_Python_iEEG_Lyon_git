
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import pickle
import cv2

import pickle
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







########################################
######## PSD & COH PRECOMPUTE ########
########################################



def load_surrogates(sujet, monopol):

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
                surrogates_cyclefreq[session_i,:,:,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_wb_bi.npy')[:len(chan_list_ieeg),:,:]
                surrogates_MVL[session_i,:] = np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_wb_bi.npy')[:len(chan_list_ieeg)]

        #### median
        surrogates_allcond['Cxy'][cond] = np.median(surrogates_Cxy, axis=0)
        surrogates_allcond['cyclefreq'][cond] = np.median(surrogates_cyclefreq, axis=0)
        surrogates_allcond['MVL'][cond] = np.median(surrogates_MVL, axis=0)

    return surrogates_allcond






def get_metrics_for_sujet(sujet, monopol):

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)

    if sujet in sujet_list:

        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

    else:

        conditions = ['FR_CV']

    #### params
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    band_prep = 'wb'

    #### MA
    metrics_allcond = {}

    #### compute
    for cond in conditions:

        #### MA
        metrics_allcond[cond] = {}

        Cxy_for_cond = np.zeros(( len(chan_list_ieeg), session_count[cond], len(hzCxy) ))
        Pxx_for_cond = np.zeros(( len(chan_list_ieeg), session_count[cond], len(hzPxx) ))
        cyclefreq_for_cond = np.zeros(( len(chan_list_ieeg), session_count[cond], stretch_point_surrogates_MVL_Cxy ))
        # MI_for_cond = np.zeros(( data_tmp.shape[0] ))
        MVL_for_cond = np.zeros(( len(chan_list_ieeg), session_count[cond] ))
        # cyclefreq_binned_for_cond = np.zeros(( data_tmp.shape[0], MI_n_bin))

        for session_i in range(session_count[cond]):
    
            #### extract data
            chan_i = chan_list.index('nasal')
            respi = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[chan_i,:]
            data_tmp = load_data_sujet(sujet, band_prep, cond, session_i, monopol)

            #### compute
            #n_chan = 0
            for n_chan, _ in enumerate(chan_list_ieeg):

                #### Pxx, Cxy, CycleFreq
                x = data_tmp[n_chan,:]
                hzPxx, Pxx = scipy.signal.welch(x, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

                y = respi
                hzPxx, Cxy = scipy.signal.coherence(x, y, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

                x_stretch, trash = stretch_data(respfeatures_allcond[cond][session_i], stretch_point_surrogates_MVL_Cxy, x, srate)
                x_stretch_mean = np.mean(x_stretch, 0)

                Cxy_for_cond[n_chan,session_i,:] = Cxy[mask_hzCxy]
                Pxx_for_cond[n_chan,session_i,:] = Pxx
                cyclefreq_for_cond[n_chan,session_i,:] = x_stretch_mean

                #### MVL
                x_zscore = zscore(x)
                x_stretch, trash = stretch_data(respfeatures_allcond[cond][session_i], stretch_point_surrogates_MVL_Cxy, x_zscore, srate)

                MVL_for_cond[n_chan,session_i] = get_MVL(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())

                # #### MI
                # x = x_stretch_mean

                # x_bin = np.zeros(( MI_n_bin ))

                # for bin_i in range(MI_n_bin):
                #     x_bin[bin_i] = np.mean(x[MI_bin_i*bin_i:MI_bin_i*(bin_i+1)])

                # cyclefreq_binned_for_cond[n_chan,:] = x_bin

                # x_bin += np.abs(x_bin.min())*2 #supress zero values
                # x_bin = x_bin/np.sum(x_bin) #transform into probabilities
                    
                # MI_for_cond[n_chan] = Shannon_MI(x_bin)

        #### mean
        metrics_allcond[cond]['Cxy'] = Cxy_for_cond.mean(axis=1)
        metrics_allcond[cond]['Pxx'] = Pxx_for_cond.mean(axis=1)
        metrics_allcond[cond]['cyclefreq'] = cyclefreq_for_cond.mean(axis=1)
        metrics_allcond[cond]['MVL'] = MVL_for_cond.mean(axis=1)

    return metrics_allcond




def compute_and_save_metrics_sujet(sujet, monopol):

    #### verify computation
    if monopol:
        
        if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'allcond_{sujet}_metrics.pkl')):
            print('ALREADY COMPUTED', flush=True)
            return

    else:
        
        if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'allcond_{sujet}_metrics_bi.pkl')):
            print('ALREADY COMPUTED', flush=True)
            return

    #### compute metrics
    metrics_allcond = get_metrics_for_sujet(sujet, monopol)

    #### save 
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    if monopol:

        with open(f'allcond_{sujet}_metrics.pkl', 'wb') as f:
            pickle.dump(metrics_allcond, f)

    else:

        with open(f'allcond_{sujet}_metrics_bi.pkl', 'wb') as f:
            pickle.dump(metrics_allcond, f)

    print('done', flush=True) 




################################################
######## PLOT & SAVE PSD AND COH ########
################################################



#n_chan = 0
def plot_save_PSD_Cxy_CF_MVL(n_chan, monopol):

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
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet, monopol)
    surrogates_allcond = load_surrogates(sujet, monopol)

    #### identify chan params
    if sujet[:3] != 'pat':
        if monopol:
            chan_list_modified, chan_list_keep = modify_name(chan_list_ieeg)
            chan_name = chan_list_modified[n_chan]    
        else:
            chan_name = chan_list_ieeg[n_chan]
    else:
        chan_name = chan_list_ieeg[n_chan]

    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

    if sujet in sujet_list:
        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']
    else:
        conditions = ['FR_CV']

    #### plot
    print_advancement(n_chan, len(chan_list_ieeg), steps=[25, 50, 75])

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    band_prep = 'wb'

    fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
    plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')
    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV'   
    for c, cond in enumerate(conditions):

        #### identify respi mean
        respi_median = []
        for session_i in range(session_count[cond]):
            respi_median.append(np.round(respfeatures_allcond[cond][session_i]['cycle_freq'].median(), 3))
        respi_mean = np.round(np.median(respi_median),3)
                
        #### plot
        if len(conditions) == 1:
            ax = axs[0]
        else:      
            ax = axs[0, c]
        ax.set_title(cond, fontweight='bold', rotation=0)
        ax.semilogy(hzPxx, metrics_allcond[cond]['Pxx'][n_chan,:], color='k')
        ax.vlines(respi_mean, ymin=0, ymax=metrics_allcond[cond]['Pxx'][n_chan,:].max(), color='r')
        ax.set_xlim(0,60)

        if len(conditions) == 1:
            ax = axs[1]
        else:      
            ax = axs[1, c]
        Pxx_sel_min = metrics_allcond[cond]['Pxx'][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].min()
        Pxx_sel_max = metrics_allcond[cond]['Pxx'][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].max()
        ax.semilogy(hzPxx[remove_zero_pad:], metrics_allcond[cond]['Pxx'][n_chan,remove_zero_pad:], color='k')
        ax.set_xlim(0, 2)
        ax.set_ylim(Pxx_sel_min, Pxx_sel_max)
        ax.vlines(respi_mean, ymin=0, ymax=metrics_allcond[cond]['Pxx'][n_chan,remove_zero_pad:].max(), color='r')

        if len(conditions) == 1:
            ax = axs[2]
        else:      
            ax = axs[2, c]
        ax.plot(hzCxy,metrics_allcond[cond]['Cxy'][n_chan,:], color='k')
        ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][n_chan,:], color='c')
        ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

        if len(conditions) == 1:
            ax = axs[3]
        else:      
            ax = axs[3, c]
        MVL_i = np.round(metrics_allcond[cond]['MVL'][n_chan], 5)
        MVL_surr = np.percentile(surrogates_allcond['MVL'][cond][n_chan,:], 99)
        if MVL_i > MVL_surr:
            MVL_p = f'MVL : {MVL_i}, *** {int(MVL_i * 100 / MVL_surr)}%'
        else:
            MVL_p = f'MVL : {MVL_i}, NS {int(MVL_i * 100 / MVL_surr)}%'
        # ax.set_title(MVL_p, rotation=0)
        ax.set_xlabel(MVL_p)

        ax.plot(metrics_allcond[cond]['cyclefreq'][n_chan,:], color='k')
        ax.plot(surrogates_allcond['cyclefreq'][cond][0, n_chan,:], color='b')
        ax.plot(surrogates_allcond['cyclefreq'][cond][1, n_chan,:], color='c', linestyle='dotted')
        ax.plot(surrogates_allcond['cyclefreq'][cond][2, n_chan,:], color='c', linestyle='dotted')
        ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq'][cond][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq'][cond][1, n_chan,:].max(), colors='r')
        #plt.show() 

    #### save
    os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))
    fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_{band_prep}.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()



    





########################################
######## PLOT & SAVE TF & ITPC ########
########################################




def get_tf_stats(tf_plot, pixel_based_distrib, tf_stats_type):

    tf_thresh = np.zeros(tf_plot.shape)
    
    phase_list = ['inspi', 'expi']
    phase_point = int(stretch_point_TF/len(phase_list))

    if tf_stats_type == 'inter':

        start = 0
        stop = stretch_point_TF

    else:

        start = phase_point
        stop = stretch_point_TF

    #wavelet_i = 0
    for wavelet_i in range(nfrex):

        mask = np.logical_or(tf_plot[wavelet_i, start:stop] < pixel_based_distrib[wavelet_i, 0], tf_plot[wavelet_i, start:stop] > pixel_based_distrib[wavelet_i, 1])
        tf_thresh[wavelet_i, start:stop] = mask*1

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### if empty return
    if tf_thresh.sum() == 0:

        return tf_thresh

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf_plot.shape[-1])

        plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh






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



    





########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet, monopol):

    #### compute & reduce surrogates
    print('######## COMPUTE & REDUCE PSD AND COH ########', flush=True)
    compute_and_save_metrics_sujet(sujet, monopol)
    
    #### compute joblib
    print('######## PLOT & SAVE PSD AND COH ########', flush=True)
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Cxy_CF_MVL)(n_chan, monopol) for n_chan, _ in enumerate(chan_list_ieeg))

    print('done', flush=True)

    


def compilation_compute_TF_ITPC(sujet, monopol):
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########', flush=True)
        if tf_mode == 'ITPC':
            print('######## PLOT & SAVE ITPC ########', flush=True)
            continue

        #### load prms
        if sujet in sujet_list:
            conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']
        else:
            conditions = ['FR_CV']

        chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
        df_loca = get_loca_df(sujet, monopol)

        if tf_mode == 'TF':
            os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
        elif tf_mode == 'ITPC':
            os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

        #### load data
        print('LOAD', flush=True)
        os.chdir(os.path.join(path_precompute, sujet, tf_mode))

        data_allcond = {}

        for cond in conditions:

            if monopol:
                data_allcond[cond] = np.median(np.load(f'{sujet}_{tf_mode.lower()}_conv_{cond}.npy'), axis=1)
            else:
                data_allcond[cond] = np.median(np.load(f'{sujet}_{tf_mode.lower()}_conv_{cond}_bi.npy'), axis=1)

        #### scale
        vals = np.array([])

        for cond in conditions:

            vals = np.append(vals, data_allcond[cond].reshape(-1))

        if debug:

            vals_diff = np.abs(vals - np.median(vals))

            count, _, _ = plt.hist(vals_diff, bins=500)
            thresh = np.percentile(vals_diff, 100-tf_plot_percentile_scale)
            val_max = vals_diff.max()
            plt.vlines([thresh, val_max], ymin=0, ymax=count.max(), color='r')
            plt.show()

        # median_diff = np.max([np.abs(np.median(vals) - vals.min()), np.abs(np.median(vals) + vals.max())])
        median_diff = np.percentile(np.abs(vals - np.median(vals)), 100-tf_plot_percentile_scale)

        vmin = np.median(vals) - median_diff
        vmax = np.median(vals) + median_diff

        del vals

        #### inspect stats
        if debug:

            tf_stats_type = 'intra'
            n_chan, chan_name = 0, chan_list_ieeg[0]
            c, cond = 1, conditions[1]

            tf_plot = data_allcond[cond][n_chan,:,:]

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_intra.npy')[n_chan]

            plt.pcolormesh(time, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            plt.yscale('log')
            plt.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')

            plt.show()

            plt.pcolormesh(get_tf_stats(tf_plot, pixel_based_distrib))
            plt.show()

            #wavelet_i = 0
            for wavelet_i in range(tf_plot.shape[0]):

                plt.plot(tf_plot[wavelet_i,:], color='b')
                plt.hlines([pixel_based_distrib[wavelet_i,0], pixel_based_distrib[wavelet_i,1]], xmin=0, xmax=tf_plot.shape[-1] ,color='r')

                plt.title(f'{np.round(wavelet_i/tf_plot.shape[0],2)}')

                plt.show()


        #### plot
        if tf_mode == 'TF':
            os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
        elif tf_mode == 'ITPC':
            os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

        print('COMPUTE', flush=True)

        #tf_stats_type = 'intra'
        for tf_stats_type in ['inter', 'intra']:

            if sujet not in sujet_list and tf_stats_type == 'inter':
                continue

            #n_chan, chan_name = 0, chan_list_ieeg[0]
            for n_chan, chan_name in enumerate(chan_list_ieeg):

                if sujet[:3] != 'pat' and monopol:
                    chan_list_modified, chan_list_keep = modify_name(chan_list_ieeg)
                    chan_name = chan_list_modified[n_chan]
                else:
                    chan_name = chan_list_ieeg[n_chan]

                chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

                print_advancement(n_chan, len(chan_list_ieeg), steps=[25, 50, 75])

                #### plot
                fig, axs = plt.subplots(ncols=len(conditions))

                plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')

                fig.set_figheight(10)
                fig.set_figwidth(15)

                time = range(stretch_point_TF)

                #c, cond = 0, conditions[0]
                for c, cond in enumerate(conditions):

                    tf_plot = data_allcond[cond][n_chan,:,:]
                
                    if len(conditions) == 1:
                        ax = axs
                    else:
                        ax = axs[c]

                    ax.set_title(cond, fontweight='bold', rotation=0)

                    ax.pcolormesh(time, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
                    ax.set_yscale('log')

                    if tf_mode == 'TF' and cond != 'FR_CV' and tf_stats_type == 'inter':
                        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                        if monopol:
                            pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_inter.npy')[n_chan]
                        else:
                            pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_inter_bi.npy')[n_chan]

                        if get_tf_stats(tf_plot, pixel_based_distrib, tf_stats_type).sum() != 0:
                            ax.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib, tf_stats_type), levels=0, colors='g')

                    if tf_mode == 'TF' and tf_stats_type == 'intra':
                        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                        if monopol:
                            pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_intra.npy')[n_chan]
                        else:
                            pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_intra_bi.npy')[n_chan]    

                        if get_tf_stats(tf_plot, pixel_based_distrib, tf_stats_type).sum() != 0:
                            ax.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib, tf_stats_type), levels=0, colors='g')

                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
                    ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

                #plt.show()

                if tf_mode == 'TF':
                    os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
                elif tf_mode == 'ITPC':
                    os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

                #### save
                if monopol:
                    if tf_stats_type == 'inter':
                        fig.savefig(f'{sujet}_{chan_name}_inter.jpeg', dpi=150)
                    if tf_stats_type == 'intra':
                        fig.savefig(f'{sujet}_{chan_name}_intra.jpeg', dpi=150)
                else:
                    if tf_stats_type == 'inter':
                        fig.savefig(f'{sujet}_{chan_name}_inter_bi.jpeg', dpi=150)
                    if tf_stats_type == 'intra':
                        fig.savefig(f'{sujet}_{chan_name}_intra_bi.jpeg', dpi=150)
                    
                fig.clf()
                plt.close('all')
                gc.collect()

    print('done', flush=True)






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #sujet = sujet_list_FR_CV
    for sujet in sujet_list_FR_CV:

        #monopol = True
        for monopol in [True, False]:

            print(sujet, flush=True)

            #### Pxx Cxy CycleFreq
            # compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet, monopol)
            # execute_function_in_slurm_bash_mem_choice('n10_res_power_analysis', 'compilation_compute_Pxx_Cxy_Cyclefreq_MVL', [sujet, monopol], '15G')

            #### TF & ITPC
            compilation_compute_TF_ITPC(sujet, monopol)
            # execute_function_in_slurm_bash_mem_choice('n10_res_power_analysis', 'compilation_compute_TF_ITPC', [sujet, monopol], '15G')