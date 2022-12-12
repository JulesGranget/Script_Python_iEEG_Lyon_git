
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False



################################
######## STRETCH TF ########
################################


#tf = tf_allchan
#condition, resp_features, freq_band, stretch_point_TF = conditions[0], list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf_dB(sujet, tf, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate, monopol):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
    if monopol:
        baselines = np.load(f'{sujet}_{band}_baselines.npy')
    else:
        baselines = np.load(f'{sujet}_{band}_baselines_bi.npy')

    #### apply baseline
    for n_chan in range(tf.shape[0]):
        
        for fi in range(tf.shape[1]):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        tf_mean = np.mean(stretch_data_tf(respfeatures_allcond[cond][session_i], stretch_point_TF, tf[n_chan,:,:], srate)[0], axis=0)

        return tf_mean

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    #### extarct
    tf_mean_allchan = np.zeros((tf.shape[0], tf.shape[1], stretch_point_TF))

    for n_chan in range(tf.shape[0]):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan








################################
######## PRECOMPUTE TF ########
################################

# _freq_band_list = freq_band_list_precompute
def precompute_tf(sujet, cond, session_i, _freq_band_list, band_prep_list, monopol):

    print(f'{sujet} {cond} {session_i+1}')
    print('TF PRECOMPUTE')

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, monopol)
    respfeatures_allcond = load_respfeatures(sujet)

    #### select prep to load
    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[:len(chan_list_ieeg),:]

        freq_band = _freq_band_list[band_prep_i] 

        #band, freq = list(freq_band.items())[1]
        for band, freq in freq_band.items():

            #### supress indice
            band = band[:-2]

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            if monopol:
                if os.path.exists(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy') :
                    print('ALREADY COMPUTED')
                    continue
            else:
                if os.path.exists(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}_bi.npy') :
                    print('ALREADY COMPUTED')
                    continue
            
            print(band, ' : ', freq)
            print('COMPUTE')

            #### select wavelet parameters
            wavelets, nfrex = get_wavelets(sujet, band_prep, freq, monopol)

            os.chdir(path_memmap)
            if monopol:
                tf_allchan = np.memmap(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
            else:
                tf_allchan = np.memmap(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))

            def compute_tf_convolution_nchan(n_chan):

                # print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                x = data[n_chan,:]

                tf = np.zeros((nfrex, x.shape[0]))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                tf_allchan[n_chan,:,:] = tf

                return

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

            #### stretch
            print('STRETCH')
            tf_allband_stretched = compute_stretch_tf_dB(sujet, tf_allchan, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate, monopol)
            
            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            if monopol:
                np.save(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy', tf_allband_stretched)
            else:
                np.save(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}_bi.npy', tf_allband_stretched)
                        
            os.chdir(path_memmap)
            if monopol:
                os.remove(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
            else:
                os.remove(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat')

    print('done')
    


########################################
######## PRECOMPUTE ITPC ########
########################################



def precompute_itpc(sujet, cond, session_i, _freq_band_list, band_prep_list, monopol):

    print(f'{sujet} {cond} {session_i+1}')
    print('ITPC PRECOMPUTE')

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, monopol)
    respfeatures_allcond = load_respfeatures(sujet)
    
    #### select prep to load
    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[:len(chan_list_ieeg),:]

        freq_band = _freq_band_list[band_prep_i]

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            if monopol:
                if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy') :
                    print('ALREADY COMPUTED')
                    continue
            else:
                if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}_bi.npy') :
                    print('ALREADY COMPUTED')
                    continue
            
            print(band, ' : ', freq)

            #### select wavelet parameters
            wavelets, nfrex = get_wavelets(sujet, band_prep, freq, monopol)

            #### compute itpc
            print('COMPUTE, STRETCH & ITPC')
            def compute_itpc_n_chan(n_chan):
                    
                x = data[n_chan,:]

                tf = np.zeros((nfrex, x.shape[0]), dtype='complex')

                for fi in range(nfrex):
                    
                    tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

                #### stretch
                tf_stretch = stretch_data_tf(respfeatures_allcond[cond][session_i], stretch_point_TF, tf, srate)[0]

                #### ITPC
                tf_angle = np.angle(tf_stretch)
                tf_cangle = np.exp(1j*tf_angle) 
                itpc = np.abs(np.mean(tf_cangle,0))

                if debug == True:
                    plt.pcolormesh(itpc)
                    plt.show()

                return itpc 

            compute_itpc_n_chan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_itpc_n_chan)(n_chan) for n_chan in range(data.shape[0]))
            
            #### extract
            itpc_allchan = np.zeros((data.shape[0],nfrex,stretch_point_TF))

            for n_chan in range(data.shape[0]):

                itpc_allchan[n_chan,:,:] = compute_itpc_n_chan_res[n_chan]

            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            if monopol:
                np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy', itpc_allchan)
            else:
                np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}_bi.npy', itpc_allchan)

            del itpc_allchan

    print('done')






########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:    

        #monopol = True
        for monopol in [True, False]:

            #### load data
            conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, monopol)
            respfeatures_allcond = load_respfeatures(sujet)

            #### compute all
            print('######## PRECOMPUTE TF & ITPC ########')

            #### compute and save tf
            #cond = 'FR_CV'
            #session_i = 0
            for cond in conditions:

                if len(respfeatures_allcond[cond]) == 1:
            
                    # precompute_tf(sujet, cond, 0, freq_band_list_precompute, band_prep_list, monopol)
                    execute_function_in_slurm_bash_mem_choice('n7_precompute_TF', 'precompute_tf', [sujet, cond, 0, freq_band_list_precompute, band_prep_list, monopol], '15G')
                    # precompute_itpc(sujet, cond, 0, freq_band_list, band_prep_list, monopol)
                    execute_function_in_slurm_bash_mem_choice('n7_precompute_TF', 'precompute_itpc', [sujet, cond, 0, freq_band_list, band_prep_list, monopol], '15G')
                
                elif len(respfeatures_allcond[cond]) > 1:

                    for session_i in range(len(respfeatures_allcond[cond])):

                        # precompute_tf(sujet, cond, session_i, freq_band_list_precompute, band_prep_list, monopol)
                        execute_function_in_slurm_bash_mem_choice('n7_precompute_TF', 'precompute_tf', [sujet, cond, session_i, freq_band_list_precompute, band_prep_list, monopol], '15G')
                        # precompute_itpc(sujet, cond, session_i, freq_band_list, band_prep_list, monopol)
                        execute_function_in_slurm_bash_mem_choice('n7_precompute_TF', 'precompute_itpc', [sujet, cond, session_i, freq_band_list, band_prep_list, monopol], '15G')









