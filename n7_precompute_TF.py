
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib

from n0_config import *
from n0bis_analysis_functions import *

debug = False



################################
######## STRETCH TF ########
################################



#condition, resp_features, freq_band, stretch_point_TF = 'CV', list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf(tf, cond, session_i, respfeatures_allcond, stretch_point_TF):

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    for n_chan in range(np.size(tf,0)):

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x, srate)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        tf_mean_allchan[n_chan,:,:] = tf_mean

    return tf_mean_allchan

#tf = tf_allchan
#condition, resp_features, freq_band, stretch_point_TF = conditions[0], list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf_dB(tf, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
    baselines = np.load(f'{sujet}_{band}_baselines.npy')

    #### apply baseline
    for n_chan in range(np.size(tf,0)):
        
        for fi in range(np.size(tf,1)):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        print_advancement(n_chan, np.size(tf,0), steps=[25, 50, 75])

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x, srate)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        return tf_mean

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(np.size(tf,0)))

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    for n_chan in range(np.size(tf,0)):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan


#condition, resp_features, freq_band, stretch_point_TF = conditions[0], list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf_itpc(tf, cond, session_i, respfeatures_allcond, stretch_point_TF, srate):
    
    #### identify number stretch
    x = tf[0,:]
    x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x, srate)
    nb_cycle = np.size(x_stretch, 0)
    
    #### compute tf
    tf_stretch = np.zeros((nb_cycle, np.size(tf,0), int(stretch_point_TF)), dtype='complex')

    for fi in range(np.size(tf,0)):

        x = tf[fi,:]
        x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x, srate)
        tf_stretch[:,fi,:] = x_stretch

    return tf_stretch






################################
######## PRECOMPUTE TF ########
################################


def precompute_tf(cond, session_i, freq_band_list, band_prep_list):

    print('TF PRECOMPUTE')

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    respfeatures_allcond = load_respfeatures(sujet)

    #### select prep to load
    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data = load_data(band_prep, cond, session_i)[:len(chan_list_ieeg),:]

        freq_band = freq_band_list[band_prep_i] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            if os.path.exists(sujet + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)
            print('COMPUTE')

            #### select wavelet parameters
            wavelets, nfrex = get_wavelets(band_prep, freq)

            os.chdir(path_memmap)
            tf_allchan = np.memmap(f'{sujet}_{cond}_{session_i}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(np.size(data,0), nfrex, np.size(data,1)))

            def compute_tf_convolution_nchan(n_chan):

                print_advancement(n_chan, np.size(data,0), steps=[25, 50, 75])

                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                tf_allchan[n_chan,:,:] = tf

                return

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(np.size(data,0)))

            #### stretch
            print('STRETCH')
            tf_allband_stretched = compute_stretch_tf_dB(tf_allchan, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate)
            
            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            np.save(sujet + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy', tf_allband_stretched)
            
            os.chdir(path_memmap)
            os.remove(f'{sujet}_{cond}_{session_i}_precompute_convolutions.dat')


    print('done')


################################
######## PRECOMPUTE ITPC ########
################################



def precompute_tf_itpc(cond, session_i, freq_band_list, band_prep_list):

    print('ITPC PRECOMPUTE')

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
    
    #### select prep to load
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data = load_data(band_prep, cond, session_i)[:len(chan_list_ieeg),:]

        freq_band = freq_band_list[band_prep_i]

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            if os.path.exists(sujet + '_itpc_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)

            #### select wavelet parameters
            wavelets, nfrex = get_wavelets(band_prep, freq)

            #### compute itpc
            print('COMPUTE, STRETCH & ITPC')
            def compute_itpc_n_chan(n_chan):

                if n_chan/np.size(data,0) % .2 <= .01:
                    print("{:.2f}".format(n_chan/np.size(data,0)))
                    
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)), dtype='complex')

                for fi in range(nfrex):
                    
                    tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

                #### stretch
                tf_stretch = compute_stretch_tf_itpc(tf, cond, session_i, respfeatures_allcond, stretch_point_TF, srate)

                #### ITPC
                tf_angle = np.angle(tf_stretch)
                tf_cangle = np.exp(1j*tf_angle) 
                itpc = np.abs(np.mean(tf_cangle,0))

                if debug == True:
                    time = range(stretch_point_TF)
                    frex = range(nfrex)
                    plt.pcolormesh(time,frex,itpc,vmin=np.min(itpc),vmax=np.max(itpc))
                    plt.show()

                return itpc 

            compute_itpc_n_chan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_itpc_n_chan)(n_chan) for n_chan in range(np.size(data,0)))
            
            itpc_allchan = np.zeros((np.size(data,0),nfrex,stretch_point_TF))

            for n_chan in range(np.size(data,0)):

                itpc_allchan[n_chan,:,:] = compute_itpc_n_chan_res[n_chan]

            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            np.save(sujet + '_itpc_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy', itpc_allchan)

            del itpc_allchan


    print('done')


########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':


    #### load data

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    respfeatures_allcond = load_respfeatures(sujet)

    #### compute all

    print('######## PRECOMPUTE TF & ITPC ########')

    #### compute and save tf
    #cond = 'FR_CV'
    #session_i = 0
    for cond in conditions:

        print(cond)

        if len(respfeatures_allcond[cond]) == 1:
    
            # precompute_tf(cond, 0, freq_band_list, band_prep_list)
            execute_function_in_slurm_bash('n7_precompute_TF', 'precompute_tf', [cond, 0, freq_band_list, band_prep_list])
            # precompute_tf_itpc(cond, 0, freq_band_list, band_prep_list)
            execute_function_in_slurm_bash('n7_precompute_TF', 'precompute_tf_itpc', [cond, 0, freq_band_list, band_prep_list])
        
        elif len(respfeatures_allcond[cond]) > 1:

            for session_i in range(len(respfeatures_allcond[cond])):

                # precompute_tf(cond, session_i, freq_band_list, band_prep_list)
                execute_function_in_slurm_bash('n7_precompute_TF', 'precompute_tf', [cond, session_i, freq_band_list, band_prep_list])
                # precompute_tf_itpc(cond, session_i, freq_band_list, band_prep_list)
                execute_function_in_slurm_bash('n7_precompute_TF', 'precompute_tf_itpc', [cond, session_i, freq_band_list, band_prep_list])








