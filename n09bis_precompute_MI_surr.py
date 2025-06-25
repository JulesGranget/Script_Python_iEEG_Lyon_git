

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import networkx as nx
import xarray as xr

import pickle
import joblib

from n00_config_params import *
from n00bis_config_analysis_functions import *


debug = False


#sujet, monopol = sujet_list[0], True
def export_surr_MI(sujet, monopol):

    if sujet in sujet_list:

        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

    else:

        conditions = ['FR_CV']

    #### verif computation
    # if monopol:

    #     if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'xr_MI_{sujet}.nc')):
    #         print('MI : ALREADY COMPUTED', flush=True)
    #         return

    # else:

    #     if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'xr_MI_{sujet}_bi.nc')):
    #         print('MI : ALREADY COMPUTED', flush=True)
    #         return

    #### identify chan params
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    if sujet[:3] != 'pat':
        if monopol:
            chan_list_sel, chan_list_keep = modify_name(chan_list_ieeg)
        else:
            chan_list_sel = chan_list_ieeg
    else:
        chan_list_sel = chan_list_ieeg

    #### prepare df
    data_xr = np.zeros((len(conditions), len(chan_list_sel), 3, len(freq_band_dict_FC_lmm['wb']), stretch_point_TF)) 
    dict_xr = {'cond' : conditions, 'chan' : chan_list_sel, 'thresh' : ['down', 'up' ,'obs'], 'band' : list(freq_band_dict_FC_lmm['wb']), 'time' : np.arange(stretch_point_TF)}

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    #cond = conditions[-1]
    for cond_i, cond in enumerate(conditions):

        print(cond, flush=True)
            
        #### Pxx
        if monopol:
            tf = np.load(f'{sujet}_tf_conv_{cond}.npy')
        else:
            tf = np.load(f'{sujet}_tf_conv_{cond}_bi.npy')

        if debug:

            plt.pcolormesh(np.median(tf[0], axis=0))
            plt.show()

            band, freq = 'theta', [4,8]
            frex_sel = (frex >= freq[0]) & (frex <= freq[-1])
            for cycle_i in range(tf.shape[1]):
                plt.plot(np.median(tf[0,cycle_i,frex_sel], axis=0))
            plt.show()

            plt.plot(np.median(np.median(tf[0], axis=0)[frex_sel], axis=0))
            plt.show()

        #### fill df
        #chan_i, chan_name = 0, chan_list_sel[0]
        # for chan_i, chan_name in enumerate(chan_list_sel):

        def get_surr_MI_for_chan(chan_i):

            print_advancement(chan_i, len(chan_list_sel), steps=[25, 50, 75])

            data_chan_xr = np.zeros((3, len(freq_band_dict_FC_lmm['wb']), stretch_point_TF)) 

            tf_chan = tf[chan_i,:,:]

            #band, freq = 'theta', [4,8]
            for band_i, (band, freq) in enumerate(freq_band_dict_FC_lmm['wb'].items()):

                frex_sel = (frex >= freq[0]) & (frex <= freq[-1])
                tf_chan_frex = np.median(tf_chan[:,frex_sel], axis=1)
                flat_suffle_sig = tf_chan_frex.reshape(-1)
                n_cycle = tf_chan_frex.shape[0]

                _surr = np.zeros((n_surrogates_tf, tf_chan_frex.shape[-1]))

                for surr_i in np.arange(n_surrogates_tf):

                    shuffle_win_start = np.random.choice(np.arange(tf_chan_frex.shape[-1], flat_suffle_sig.size-tf_chan_frex.shape[-1]), n_cycle, replace=False)
                    shuffle_win_stop = shuffle_win_start + tf_chan_frex.shape[-1]

                    _shuffle_win = np.zeros((n_cycle, tf_chan_frex.shape[-1]))
                    for win_i, (_win_start, _win_stop) in enumerate(zip(shuffle_win_start, shuffle_win_stop)):
                        _shuffle_win[win_i] = flat_suffle_sig[_win_start:_win_stop]

                    _surr[surr_i] = np.median(_shuffle_win, axis=0)

                    if debug:

                        plt.plot(np.median(tf_chan_frex, axis=0))
                        plt.plot(np.median(_shuffle_win, axis=0), color='r')
                        plt.legend()
                        plt.show()
                    
                if debug:

                    count, _, _ = plt.hist(_surr, bins=50)
                    plt.vlines([np.median(tf_chan_frex, axis=0).max() - np.median(tf_chan_frex, axis=0).min()], ymin=0, ymax=count.max(), color='r', label='obs')
                    plt.vlines([np.percentile(_surr, 99)], ymin=0, ymax=count.max(), color='g', label='99th')
                    plt.legend()
                    plt.show()

                    plt.plot(np.median(tf_chan_frex, axis=0))
                    plt.plot(np.percentile(_surr, 99, axis=0), color='r')
                    plt.plot(np.percentile(_surr, 1, axis=0), color='r')
                    plt.legend()
                    plt.show()

                data_chan_xr[0, band_i, :], data_chan_xr[1, band_i, :] = np.percentile(_surr, percentile_MI[0], axis=0), np.percentile(_surr, percentile_MI[1], axis=0)
                data_chan_xr[2, band_i, :] = np.median(tf_chan_frex, axis=0)

            return data_chan_xr

        allchan_MI_surr = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_surr_MI_for_chan)(chan_i) for chan_i, _ in enumerate(chan_list_sel))

        #### extract 
        for chan_i, _ in enumerate(chan_list_sel):

            data_xr[cond_i, chan_i] = allchan_MI_surr[chan_i]

    #### save
    xr_MI_surr = xr.DataArray(data=data_xr, dims=dict_xr.keys(), coords=dict_xr)

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    if monopol:
        xr_MI_surr.to_netcdf(f'xr_MI_{sujet}.nc')
    else:
        xr_MI_surr.to_netcdf(f'xr_MI_{sujet}_bi.nc')
        
    print('done', flush=True)






################################
######## EXECUTE ########
################################

if __name__ == '__main__':



    ######## MI ########
    list_params = []
    for sujet in sujet_list_FR_CV:    
        for monopol in [True, False]:
            list_params.append([sujet, monopol])

    execute_function_in_slurm_bash('n09bis_precompute_MI_surr', 'export_surr_MI', list_params)
    #sync_folders__push_to_crnldata()

