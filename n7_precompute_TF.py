

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False



################################
######## STRETCH TF ########
################################




#tf, respfeatures_i = tf_allchan.copy(), respfeatures_allcond[cond][session_i]
def compute_stretch_tf(sujet, tf, cond, respfeatures_i, stretch_point_TF, srate, monopol):

    #n_chan = 0
    def stretch_tf_db_n_chan(n_chan):

        tf_mean = stretch_data_tf(respfeatures_i, stretch_point_TF, tf[n_chan,:,:], srate)[0]

        return tf_mean
    
    #### raw
    n_cycle_stretch = stretch_data_tf(respfeatures_i, stretch_point_TF, tf[0,:,:], srate)[0].shape[0]
    tf_mean_allchan = np.zeros((tf.shape[0], n_cycle_stretch, tf.shape[1], stretch_point_TF))
    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))
    
    for n_chan in range(tf.shape[0]):
        tf_mean_allchan[n_chan,:,:,:] = stretch_tf_db_nchan_res[n_chan]

    print('SAVE RAW', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if monopol:
        np.save(f'{sujet}_tf_raw_{cond}.npy', tf_mean_allchan)
    else:
        np.save(f'{sujet}_tf_raw_{cond}_bi.npy', tf_mean_allchan)

    del stretch_tf_db_nchan_res

    #### norm
    tf[:] = norm_tf(sujet, tf, monopol, norm_method)

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    #### extract
    for n_chan in range(tf.shape[0]):
        tf_mean_allchan[n_chan,:,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan





def compute_stretch_tf_itpc(tf, cond, respfeatures_allcond, stretch_point_TF, srate):
    
    tf_stretch, ratio = stretch_data(respfeatures_allcond[cond][0], stretch_point_TF, tf, srate)

    return tf_stretch







################################
######## PRECOMPUTE TF ########
################################


def precompute_tf_allconv(sujet, cond, monopol):

    #### verify il already computed
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    if monopol:
        if os.path.exists(f'{sujet}_tf_conv_{cond}.npy') and os.path.exists(f'{sujet}_tf_raw_{cond}.npy'):
            print('ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(f'{sujet}_tf_conv_{cond}_bi.npy') and os.path.exists(f'{sujet}_tf_raw_{cond}_bi.npy'):
            print('ALREADY COMPUTED', flush=True)
            return

    print(f'TF PRECOMPUTE {sujet} {cond}', flush=True)

    #### get params
    band_prep = 'wb'

    respfeatures_allcond = load_respfeatures(sujet)
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    #### MA
    n_cycle_stretch = 0
    for session_i in range(session_count[cond]):
        data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)
        _n_cycle_stretch = stretch_data_tf(respfeatures_allcond[cond][session_i], stretch_point_TF, np.ones((nfrex, data.shape[1])), srate)[0].shape[0]
        n_cycle_stretch += _n_cycle_stretch

    os.chdir(path_memmap)
    tf_allsession = np.memmap(f'{sujet}_tf_conv_{cond}_{monopol}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), n_cycle_stretch, nfrex, stretch_point_TF))

    #### select wavelet parameters
    wavelets = get_wavelets()

    #### compute
    cycle_stretch_pre = 0

    #session_i = 0
    for session_i in range(session_count[cond]):

        #### load data
        data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)
            
        print(f'COMPUTE ses{session_i+1}', flush=True)

        #### conv
        os.chdir(path_memmap)
        data_r = np.memmap(f'{sujet}_tf_{cond}_data_read_{monopol}.dat', dtype=np.float32, mode='w+', shape=(data.shape))
        data_r[:] = data
        del data
        tf_allchan = np.memmap(f'{sujet}_tf_{cond}_precompute_convolutions_{monopol}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), nfrex, data_r.shape[1]))

        #n_chan = len(chan_list_ieeg)
        def compute_tf_convolution_nchan(n_chan):

            print_advancement(n_chan, data_r.shape[0], steps=[25, 50, 75])

            for fi in range(nfrex):
                
                tf_allchan[n_chan,fi,:] = abs(scipy.signal.fftconvolve(data_r[n_chan,:], wavelets[fi,:], 'same'))**2 

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan, _ in enumerate(chan_list_ieeg))

        #### stretch or chunk
        n_cycle_stretch = stretch_data_tf(respfeatures_allcond[cond][session_i], stretch_point_TF, tf_allchan[0,:,:], srate)[0].shape[0]

        print(f'STRETCH ses{session_i+1}', flush=True)
        cycle_pre = cycle_stretch_pre
        cycle_post = cycle_stretch_pre + n_cycle_stretch
        tf_allsession[:,cycle_pre:cycle_post,:,:] = compute_stretch_tf(sujet, tf_allchan, cond, respfeatures_allcond[cond][session_i], stretch_point_TF, srate, monopol)

        cycle_stretch_pre += n_cycle_stretch

        os.chdir(path_memmap)
        os.remove(f'{sujet}_tf_{cond}_precompute_convolutions_{monopol}.dat')
        os.remove(f'{sujet}_tf_{cond}_data_read_{monopol}.dat')

    #### save
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if monopol:
        np.save(f'{sujet}_tf_conv_{cond}.npy', tf_allsession)
    else:
        np.save(f'{sujet}_tf_conv_{cond}_bi.npy', tf_allsession)    

    os.chdir(path_memmap)
    os.remove(f'{sujet}_tf_conv_{cond}_{monopol}.dat')

    print('done', flush=True)







################################
######## PRECOMPUTE ITPC ########
################################



# def precompute_itpc(sujet, cond, band_prep_list, electrode_recording_type):

#     print('ITPC PRECOMPUTE', flush=True)

#     respfeatures_allcond = load_respfeatures(sujet)
#     chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    
#     #### select prep to load
#     for band_prep_i, band_prep in enumerate(band_prep_list):

#         #### select data without aux chan
#         data = load_data(sujet, cond, electrode_recording_type)

#         #### remove aux chan
#         data = data[:len(chan_list_ieeg),:]

#         freq_band = freq_band_list_precompute[band_prep_i]

#         #band, freq = list(freq_band.items())[0]
#         for band, freq in freq_band.items():

#             os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

#             if electrode_recording_type == 'monopolaire':
#                 if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy') :
#                     print('ALREADY COMPUTED', flush=True)
#                     continue
#             if electrode_recording_type == 'bipolaire':
#                 if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy') :
#                     print('ALREADY COMPUTED', flush=True)
#                     continue
            
#             print(band, ' : ', freq, flush=True)

#             #### select wavelet parameters
#             wavelets = get_wavelets()

#             #### compute itpc
#             print('COMPUTE, STRETCH & ITPC', flush=True, flush=True)
#             #n_chan = 0
#             def compute_itpc_n_chan(n_chan):

#                 print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

#                 x = data[n_chan,:]

#                 tf = np.zeros((nfrex, x.shape[0]), dtype='complex')

#                 for fi in range(nfrex):
                    
#                     tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

#                 #### stretch
#                 if cond == 'FR_CV':
#                     tf_stretch = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, tf, srate)[0]

#                 elif cond == 'AC':
#                     ac_starts = get_ac_starts(sujet)
#                     tf_stretch = chunk_stretch_tf_itpc_ac(sujet, tf, cond, ac_starts, srate)

#                 elif cond == 'SNIFF':
#                     sniff_starts = get_sniff_starts(sujet)
#                     tf_stretch = chunk_stretch_tf_itpc_sniff(sujet, tf, cond, sniff_starts, srate)

#                 #### ITPC
#                 tf_angle = np.angle(tf_stretch)
#                 tf_cangle = np.exp(1j*tf_angle) 
#                 itpc = np.abs(np.mean(tf_cangle,0))

#                 if debug == True:
#                     time = range(stretch_point_TF)
#                     frex = range(nfrex)
#                     plt.pcolormesh(time,frex,itpc,vmin=np.min(itpc),vmax=np.max(itpc))
#                     plt.show()

#                 return itpc 

#             compute_itpc_n_chan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_itpc_n_chan)(n_chan) for n_chan in range(data.shape[0]))
            
#             if cond == 'FR_CV':
#                 itpc_allchan = np.zeros((data.shape[0],nfrex,stretch_point_TF))

#             elif cond == 'AC':
#                 stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
#                 itpc_allchan = np.zeros((data.shape[0], nfrex, stretch_point_TF_ac))

#             elif cond == 'SNIFF':
#                 stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
#                 itpc_allchan = np.zeros((data.shape[0], nfrex, stretch_point_TF_sniff))

#             for n_chan in range(data.shape[0]):

#                 itpc_allchan[n_chan,:,:] = compute_itpc_n_chan_res[n_chan]

#             #### save
#             print('SAVE', flush=True)
#             os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
#             if electrode_recording_type == 'monopolaire':
#                 np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy', itpc_allchan)
#             if electrode_recording_type == 'bipolaire':
#                 np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy', itpc_allchan)
            

#             del itpc_allchan














########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list[1]
    for sujet in sujet_list_FR_CV:    

        if sujet in sujet_list:

            conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

        else:

            conditions = ['FR_CV']

        #monopol = True
        for monopol in [True, False]:

            #### load dataprecompute_tf_allconv(sujet, cond, monopol)

            #### compute all
            print('######## PRECOMPUTE TF & ITPC ########', flush=True)

            #### compute and save tf
            #cond = 'RD_FV'
            for cond in conditions:

                precompute_tf_allconv(sujet, cond, monopol)
                # execute_function_in_slurm_bash_mem_choice('n7_precompute_TF', 'precompute_tf_allconv', [sujet, cond, monopol], '40G')
                
                # precompute_itpc(sujet, cond, session_i, freq_band_list, band_prep_list, monopol)
                # execute_function_in_slurm_bash_mem_choice('n7_precompute_TF', 'precompute_itpc', [sujet, cond, session_i, freq_band_list, band_prep_list, monopol], '15G')









