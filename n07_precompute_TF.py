

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import joblib

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False



################################
######## STRETCH TF ########
################################




#tf, respfeatures_i = tf_allchan.copy(), respfeatures_allcond[cond][session_i]
# def compute_stretch_tf(sujet, tf, cond, respfeatures_i, stretch_point_TF, srate, monopol):

#     #n_chan = 0
#     def stretch_tf_db_n_chan(n_chan):

#         tf_mean_allchan[n_chan,:,:,:] = stretch_data_tf(respfeatures_i, stretch_point_TF, tf[n_chan,:,:], srate)[0]
    
#     #### raw
#     n_cycle_stretch = stretch_data_tf(respfeatures_i, stretch_point_TF, tf[0,:,:], srate)[0].shape[0]
#     os.chdir(path_memmap)
#     tf_mean_allchan = np.memmap(f'{sujet}_stretch_tf_{cond}_{monopol}.dat', dtype=np.float32, mode='w+', shape=(tf.shape[0], n_cycle_stretch, tf.shape[1], stretch_point_TF))
#     joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

#     print('SAVE RAW', flush=True)
#     os.chdir(os.path.join(path_precompute, sujet, 'TF'))
#     if monopol:
#         np.save(f'{sujet}_tf_raw_{cond}.npy', tf_mean_allchan)
#     else:
#         np.save(f'{sujet}_tf_raw_{cond}_bi.npy', tf_mean_allchan)

#     #### norm
#     print('COMPUTE STRETCH', flush=True)
#     tf[:] = norm_tf(sujet, tf, monopol, norm_method)

#     joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

#     return tf_mean_allchan











################################
######## PRECOMPUTE TF ########
################################

#sujet, cond, monopol = sujet_list_FR_CV[13], 'FR_CV', True
def precompute_tf_allconv(sujet, cond, monopol):

    #### verify il already computed
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    if monopol:
        if os.path.exists(f'{sujet}_tf_conv_{cond}.npy'):
            print('ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(f'{sujet}_tf_conv_{cond}_bi.npy'):
            print('ALREADY COMPUTED', flush=True)
            return

    print(f'TF PRECOMPUTE {sujet} {cond}', flush=True)

    #### get params
    band_prep = 'wb'

    respfeatures_allcond = load_respfeatures(sujet)
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    
    #### select wavelet parameters
    wavelets = get_wavelets()

    #### compute
    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'count_cycle'))
    cycle_counts = pd.read_excel(f"{sujet}_count_cycles.xlsx").query(f"cond == '{cond}'")['count'].values

    tf_allconv = np.zeros((len(chan_list_ieeg), cycle_counts.sum(), nfrex, stretch_point_TF), dtype=np.float32)

    #session_i = 0
    for session_i in range(session_count[cond]):

        #### load data
        #chan_list_ieeg = chan_list_ieeg[:10]
        data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[:len(chan_list_ieeg),:]

        if debug:

            os.chdir(os.path.join(path_results, 'allplot'))
            plt.plot(data[10,:10000])
            plt.savefig('test.png')
            plt.close()
            
        print(f'CONV ses{session_i+1}', flush=True)

        #### conv
        #n_chan = len(chan_list_ieeg)
        def compute_tf_convolution_nchan(chan_i):

            print_advancement(chan_i, data.shape[0], steps=[25, 50, 75])

            _tf_allchan = np.zeros((nfrex, data.shape[1]), dtype=np.float32)

            for fi in range(nfrex):
                
                _tf_allchan[fi,:] = abs(scipy.signal.fftconvolve(data[chan_i,:], wavelets[fi,:], 'same'))**2

            return _tf_allchan

        tf_allchan_session = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(chan_i) for chan_i, _ in enumerate(chan_list_ieeg))

        #### extract and norm
        tf_allchan = np.zeros((len(chan_list_ieeg), nfrex, data.shape[1]), dtype=np.float32)

        print(f'NORM ses{session_i+1}', flush=True)

        for chan_i, _ in enumerate(chan_list_ieeg):

            print_advancement(chan_i, data.shape[0], steps=[25, 50, 75])

            tf_allchan[chan_i,:,:] = rscore_mat(tf_allchan_session[chan_i])

        if debug:

            os.chdir(os.path.join(path_results, 'allplot'))
            plt.pcolormesh(tf_allchan[chan_i,:,:])
            plt.savefig('test.png')
            plt.close()

        #### stretch
        print(f'STRETCH ses{session_i+1}', flush=True)

        select_vec = respfeatures_allcond[cond][session_i]['select'].values.astype('bool')

        if session_i == 0:
            pre_cycle_i, post_cycle_i = 0, cycle_counts[0]  
        else:
            pre_cycle_i, post_cycle_i = cycle_counts[:session_i].sum(), cycle_counts[:session_i].sum()+cycle_counts[session_i]

        for chan_i, _ in enumerate(chan_list_ieeg):

            print_advancement(chan_i, data.shape[0], steps=[25, 50, 75])

            tf_allconv[chan_i,pre_cycle_i:post_cycle_i,:,:] = stretch_data_tf(respfeatures_allcond[cond][session_i], stretch_point_TF, tf_allchan[chan_i,:,:], srate)[0].astype(np.float32)[select_vec]


    if debug:

        os.chdir(os.path.join(path_results, 'allplot'))
        plt.pcolormesh(np.median(tf_allconv[chan_i,61:,:,:], axis=0))
        plt.savefig('test.png')
        plt.close()

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))
        test = np.load(f'{sujet}_tf_conv_{cond}.npy')

    #### save
    print('SAVE', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if monopol:
        np.save(f'{sujet}_tf_conv_{cond}.npy', tf_allconv)
    else:
        np.save(f'{sujet}_tf_conv_{cond}_bi.npy', tf_allconv)    

    print('done', flush=True)







########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    list_params = []
    for sujet in sujet_list_FR_CV:    
        if sujet in sujet_list:
            conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']
        else:
            conditions = ['FR_CV']
        for monopol in [True, False]:
            for cond in conditions:
                list_params.append([sujet, cond, monopol])

    execute_function_in_slurm_bash('n07_precompute_TF', 'precompute_tf_allconv', list_params, mem='40G')
    #sync_folders__push_to_crnldata()







