
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from n0_config import *
from n0bis_analysis_functions import *

import joblib

debug = False



################################################
######## PRECOMPUTE AND SAVE SURROGATES ########
################################################


def get_shuffle(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift


def precompute_surrogates_coh(band_prep, cond, session_i):
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    
    print(cond)

    data_tmp = load_data(band_prep, cond, session_i)

    if os.path.exists(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy') == True :
        print('ALREADY COMPUTED')
        return

    if cond == 'FR_MV':
        respi_i = chan_list.index('ventral')
    else:
        respi_i = chan_list.index('nasal')

    respi = data_tmp[respi_i,:]

    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    surrogates_n_chan = np.zeros((np.size(data_tmp,0),len(hzCxy)))

    def compute_surrogates_coh_n_chan(n_chan):

        if n_chan/np.size(data_tmp,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]
        y = respi

        surrogates_val_tmp = np.zeros((n_surrogates_coh,len(hzCxy)))
        for surr_i in range(n_surrogates_coh):
            
            #if surr_i%100 == 0:
            #    print(surr_i) 

            x_shift = get_shuffle(x)
            #y_shift = get_shuffle(y)
            hzCxy_tmp, Cxy = scipy.signal.coherence(x_shift, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

            surrogates_val_tmp[surr_i,:] = Cxy[mask_hzCxy]

        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i = int(np.floor(n_surrogates_coh*percentile_coh))
        compute_surrogates_coh_tmp = surrogates_val_tmp_sorted[percentile_i,:]

        return compute_surrogates_coh_tmp
    
    compute_surrogates_coh_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_coh_n_chan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

    for n_chan in range(np.size(data_tmp,0)):

        surrogates_n_chan[n_chan,:] = compute_surrogates_coh_results[n_chan]

    np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy', surrogates_n_chan)




def precompute_surrogates_cyclefreq(band_prep, cond, session_i, respfeatures_allcond):
    
    print(cond)

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    data_tmp = load_data(band_prep, cond, session_i)

    if os.path.exists(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy') == True :
        print('ALREADY COMPUTED')
        return

    surrogates_n_chan = np.zeros((3,np.size(data_tmp,0), stretch_point_surrogates))

    respfeatures_i = respfeatures_allcond[cond][session_i]

    def compute_surrogates_cyclefreq_nchan(n_chan):

        if n_chan/np.size(data_tmp,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]

        surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq,stretch_point_surrogates))
        for surr_i in range(n_surrogates_cyclefreq):
            
            #if surr_i%100 == 0:
            #    print(surr_i)

            x_shift = get_shuffle(x)
            #y_shift = get_shuffle(y)

            x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_shift, srate)

            x_stretch_mean = np.mean(x_stretch, axis=0)

            surrogates_val_tmp[surr_i,:] = x_stretch_mean

        mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
        percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

        up_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_up,:]
        dw_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_dw,:]

        return mean_surrogate_tmp, up_percentile_values_tmp, dw_percentile_values_tmp

    compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

    for n_chan in range(np.size(data_tmp,0)):

        surrogates_n_chan[0,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][0]
        surrogates_n_chan[1,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][1]
        surrogates_n_chan[2,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][2]
    
    np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy', surrogates_n_chan)



if enable_big_execute:
    __name__ = '__main__'


if __name__ == '__main__':


    #### load data

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(conditions_allsubjects)
    respfeatures_allcond = load_respfeatures(conditions_allsubjects)

    #### params surrogates

    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    #### compute and save

    print('######## COMPUTE SURROGATES ########')

    for band_prep in band_prep_list:

        print('COMPUTE FOR ' + band_prep)

        for cond in conditions:

            if len(respfeatures_allcond.get(cond)) == 1:

                precompute_surrogates_cyclefreq(band_prep, cond, 0, respfeatures_allcond)

                if band_prep == 'lf':
                    precompute_surrogates_coh(band_prep, cond, 0)

            elif len(respfeatures_allcond.get(cond)) > 1:

                for session_i in range(len(respfeatures_allcond.get(cond))):

                    precompute_surrogates_cyclefreq(band_prep, cond, session_i, respfeatures_allcond)

                    if band_prep == 'lf':
                        precompute_surrogates_coh(band_prep, cond, session_i)







