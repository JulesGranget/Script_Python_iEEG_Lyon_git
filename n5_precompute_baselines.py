
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import joblib
import glob
import neo

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





########################################
######## COMPUTE BASELINE ######## 
########################################

#sujet, band_prep = 'pat_03083_1527', 'lf'
def compute_and_save_baseline(sujet, band_prep, monopol):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    verif_band_compute = []
    for band in list(freq_band_dict[band_prep].keys()):
        if monopol:
            if os.path.exists(os.path.join(path_precompute, sujet, 'baselines', f'{sujet}_{band}_baselines.npy')):
                verif_band_compute.append(True)
        else:
            if os.path.exists(os.path.join(path_precompute, sujet, 'baselines', f'{sujet}_{band}_baselines_bi.npy')):
                verif_band_compute.append(True)

    if np.sum(verif_band_compute) > 0:
        print(f'{sujet} : BASELINES ALREADY COMPUTED')
        return
            
    #### open data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    if monopol:
        raw = mne.io.read_raw_fif(f'{sujet}_allcond_{band_prep}.fif', preload=True)
    else:
        raw = mne.io.read_raw_fif(f'{sujet}_allcond_{band_prep}_bi.fif', preload=True)

    data = raw.get_data()
    srate = raw.info['sfreq']

    #### open raw and section if sujet from paris
    if sujet[:3] == 'pat' and sujet not in sujet_list_paris_only_FR_CV:
        os.chdir(os.path.join(path_data, sujet))
        if monopol:
            raw = mne.io.read_raw_eeglab(f'{sujet}_allchan.set', preload=True)
        else:
            raw = mne.io.read_raw_fif(f'{sujet}_allchan_bi.set', preload=True)

        data, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw(sujet, raw, monopol)

    #### generate all wavelets to conv
    wavelets_to_conv = {}

    #band, freq = 'theta', [2, 10]
    for band, freq in freq_band_dict[band_prep].items():

        #### compute wavelets
        wavelets_to_conv[band], nfrex = get_wavelets(sujet, band_prep, freq, monopol)

    # plot all the wavelets
    if debug == True:
        for band in list(wavelets_to_conv.keys()):
            wavelets2plot = wavelets_to_conv[band]
            plt.pcolormesh(np.arange(wavelets2plot.shape[1]),np.arange(wavelets2plot.shape[0]),np.real(wavelets2plot))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(band)
            plt.show()

    #### compute convolutions
    n_band_to_compute = len(list(freq_band_dict[band_prep].keys()))

    os.chdir(path_memmap)
    if monopol:
        baseline_allchan = np.memmap(f'{sujet}_baseline_convolutions_{band_prep}.dat', dtype=np.float64, mode='w+', shape=(n_band_to_compute, data.shape[0], nfrex))
    else:
        baseline_allchan = np.memmap(f'{sujet}_baseline_convolutions_{band_prep}_bi.dat', dtype=np.float64, mode='w+', shape=(n_band_to_compute, data.shape[0], nfrex))

    #### load trig
    os.chdir(os.path.join(path_prep, sujet, 'info'))
        
    trig = pd.read_excel(sujet + '_trig.xlsx')

    #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        print_advancement(n_chan, np.size(data,0), steps=[25, 50, 75])

        x = data[n_chan,:]
        #band_i, band = 0, list(wavelets_to_conv.keys())[0]
        for band_i, band in enumerate(list(wavelets_to_conv.keys())):

            baseline_coeff_band = np.array(())
            #fi = 0
            for fi in range(nfrex):
                
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2

                #### chunk data
                fi_conv_chunked = np.array([])

                for condition, trig_cond in conditions_trig.items():

                    cond_i = np.where(trig.name.values == trig_cond[0])[0]
                    #export_i, trig_i = 0, cond_i[0]
                    for trig_i in cond_i:

                        fi_conv_chunked_i = fi_conv[trig.iloc[trig_i,:].time:trig.iloc[trig_i+1,:].time]
                        fi_conv_chunked = np.concatenate((fi_conv_chunked, fi_conv_chunked_i))

                baseline_coeff_band = np.append(baseline_coeff_band, np.median(fi_conv_chunked))
        
            baseline_allchan[band_i, n_chan,:] = baseline_coeff_band

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data,0)))

    #### save baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

    for band_i, band in enumerate(list(freq_band_dict[band_prep].keys())):
    
        if monopol:
            np.save(f'{sujet}_{band}_baselines.npy', baseline_allchan[band_i, :, :])
        else:
            np.save(f'{sujet}_{band}_baselines_bi.npy', baseline_allchan[band_i, :, :])    

    #### remove memmap
    os.chdir(path_memmap)
    if monopol:
        os.remove(f'{sujet}_baseline_convolutions_{band_prep}.dat')
    else:
        os.remove(f'{sujet}_baseline_convolutions_{band_prep}_bi.dat')

    print('done')
    print(sujet)




################################
######## EXECUTE ########
################################


if __name__== '__main__':

    #sujet = sujet_list[2]
    for sujet in sujet_list:

        #monopol = False
        for monopol in [True, False]:

            #band_prep = 'lf'
            for band_prep in band_prep_list:
                # compute_and_save_baseline(sujet, band_prep, monopol)
                execute_function_in_slurm_bash('n5_precompute_baselines', 'compute_and_save_baseline', [sujet, band_prep, monopol])

