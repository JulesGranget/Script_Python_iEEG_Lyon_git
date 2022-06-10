
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from n0_config_params import *
from n0bis_config_analysis_functions import *

import joblib

debug = False







########################################
######## ANALYSIS FUNCTIONS ########
########################################



def shuffle_CycleFreq(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift
    

def shuffle_Cxy(x):
   half_size = x.shape[0]//2
   ind = np.random.randint(low=0, high=half_size)
   x_shift = x.copy()
   
   x_shift[ind:ind+half_size] *= -1
   if np.random.rand() >=0.5:
       x_shift *= -1

   return x_shift


def Kullback_Leibler_Distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def Shannon_Entropy(a):
    a = np.asarray(a, dtype=float)
    return - np.sum(np.where(a != 0, a * np.log(a), 0))

def Modulation_Index(distrib, show=False, verbose=False):
    distrib = np.asarray(distrib, dtype = float)
    
    if verbose:
        if np.sum(distrib) != 1:
            print(f'(!)  The sum of all bins is not 1 (sum = {round(np.sum(distrib), 2)})  (!)')
        
    N = distrib.size
    uniform_distrib = np.ones(N) * (1/N)
    mi = Kullback_Leibler_Distance(distrib, uniform_distrib) / np.log(N)
    
    if show:
        bin_width_deg = 360 / N
        
        doubled_distrib = np.concatenate([distrib,distrib] )
        x = np.arange(0, doubled_distrib.size*bin_width_deg, bin_width_deg)
        fig, ax = plt.subplots(figsize = (8,4))
        
        doubled_uniform_distrib = np.concatenate([uniform_distrib,uniform_distrib] )
        ax.scatter(x, doubled_uniform_distrib, s=2, color='r')
        
        ax.bar(x=x, height=doubled_distrib, width = bin_width_deg/1.1, align = 'edge')
        ax.set_title(f'Modulation Index = {round(mi, 4)}')
        ax.set_xlabel(f'Phase (Deg)')
        ax.set_ylabel(f'Amplitude (Normalized)')
        ax.set_xticks([0,360,720])

    return mi

def Shannon_MI(a):
    a = np.asarray(a, dtype = float)
    N = a.size
    kl_divergence_shannon = np.log(N) - Shannon_Entropy(a)
    return kl_divergence_shannon / np.log(N)








################################################
######## CXY CYCLE FREQ SURROGATES ########
################################################




def precompute_surrogates_coh(sujet, band_prep, cond, session_i):
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    
    print(cond)

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

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

        print_advancement(n_chan, np.size(data_tmp,0), steps=[25, 50, 75])

        x = data_tmp[n_chan,:]
        y = respi

        surrogates_val_tmp = np.zeros((n_surrogates_coh,len(hzCxy)))

        for surr_i in range(n_surrogates_coh):

            x_shift = shuffle_Cxy(x)
            #y_shift = shuffle_Cxy(y)
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

    print('done')




def precompute_surrogates_cyclefreq(sujet, band_prep, cond, session_i):
    
    print(cond)

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    if os.path.exists(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy') == True :
        print('ALREADY COMPUTED')
        return

    #### compute surrogates
    surrogates_n_chan = np.zeros((3, data_tmp.shape[0], stretch_point_surrogates))
    MI_surrogates = np.zeros((data_tmp.shape[0], n_surrogates_cyclefreq))

    respfeatures_i = respfeatures_allcond[cond][session_i]

    def compute_surrogates_cyclefreq_nchan(n_chan):

        print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

        x = data_tmp[n_chan,:]

        surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates))

        for surr_i in range(n_surrogates_cyclefreq):

            # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

            x_shift = shuffle_CycleFreq(x)
            #y_shift = shuffle_CycleFreq(y)

            x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_shift, srate)

            x_stretch_mean = np.mean(x_stretch, axis=0)

            surrogates_val_tmp[surr_i,:] = x_stretch_mean

        mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
        percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

        up_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_up,:]
        dw_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_dw,:]

        #### compute MI
        MI_surrogates = np.array([])
        for surr_i in range(n_surrogates_cyclefreq):

            x = surrogates_val_tmp[surr_i,:]

            x += np.abs(x.min())*2 #supress zero values
            x = x/np.sum(x) #transform into probabilities
            
            MI_surrogates = np.append(MI_surrogates, Shannon_MI(x))

        return mean_surrogate_tmp, up_percentile_values_tmp, dw_percentile_values_tmp, MI_surrogates

    compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

    #### fill results
    for n_chan in range(np.size(data_tmp,0)):

        surrogates_n_chan[0, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][0]
        surrogates_n_chan[1, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][1]
        surrogates_n_chan[2, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][2]
        MI_surrogates[n_chan:] = compute_surrogates_cyclefreq_results[n_chan][3]
    
    #### save
    np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy', surrogates_n_chan)
    np.save(f'{sujet}_{cond}_{str(session_i+1)}_MI_{band_prep}.npy', MI_surrogates)

    print('done')





if __name__ == '__main__':


    #### load data
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    respfeatures_allcond = load_respfeatures(sujet)

    #### compute and save
    print('######## COMPUTE SURROGATES ########')

    #band_prep = band_prep_list[0]
    for band_prep in band_prep_list:

        print('COMPUTE FOR ' + band_prep)

        #cond = 'FR_CV'
        for cond in conditions:

            if len(respfeatures_allcond[cond]) == 1:

                # precompute_surrogates_cyclefreq(sujet, band_prep, cond, 0)
                execute_function_in_slurm_bash('n6_precompute_surrogates', 'precompute_surrogates_cyclefreq', [sujet, band_prep, cond, 0])

                if band_prep == 'lf':
                    # precompute_surrogates_coh(sujet, band_prep, cond, 0)
                    execute_function_in_slurm_bash('n6_precompute_surrogates', 'precompute_surrogates_coh', [sujet, band_prep, cond, 0])

            elif len(respfeatures_allcond[cond]) > 1:

                for session_i in range(len(respfeatures_allcond[cond])):

                    # precompute_surrogates_cyclefreq(sujet, band_prep, cond, session_i)
                    execute_function_in_slurm_bash('n6_precompute_surrogates', 'precompute_surrogates_cyclefreq', [sujet, band_prep, cond, session_i])

                    if band_prep == 'lf':
                        # precompute_surrogates_coh(sujet, band_prep, cond, session_i)
                        execute_function_in_slurm_bash('n6_precompute_surrogates', 'precompute_surrogates_coh', [sujet, band_prep, cond, session_i])







