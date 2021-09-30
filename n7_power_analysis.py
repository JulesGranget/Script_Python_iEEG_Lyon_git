

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config import *
from n0bis_analysis_functions import *


debug = False




################################
######## LOAD DATA ########
################################


conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(conditions_allsubjects)
respfeatures_allcond = load_respfeatures(conditions)
respi_ratio_allcond = get_all_respi_ratio(conditions, respfeatures_allcond)

dict_loca = get_electrode_loca()







########################################
######## PSD AND COH PARAMS ########
########################################


nwind = int( 20*srate ) # window length in seconds*srate
nfft = nwind*5 # if no zero padding nfft = nwind
noverlap = np.round(nwind/2) # number of points of overlap here 50%
hannw = scipy.signal.windows.hann(nwind) # hann window







################################################
######## PSD & COH ANALYSIS FUNCTIONS ########
################################################



#### load surrogates
def load_surrogates():
    Cxy_surrogates_allcond = {}
    cyclefreq_surrogates_allcond_lf = {}
    cyclefreq_surrogates_allcond_hf = {}
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    for cond in conditions:

        if len(respfeatures_allcond.get(cond)) == 1:

            data_load = []
            data_load.append(np.load(sujet + '_' + cond + '_' + str(1) + '_Coh.npy'))
            Cxy_surrogates_allcond[cond] = data_load

            data_load = []
            data_load.append(np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_lf.npy'))
            cyclefreq_surrogates_allcond_lf[cond] = data_load

            data_load = []
            data_load.append(np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_hf.npy'))
            cyclefreq_surrogates_allcond_hf[cond] = data_load

        elif len(respfeatures_allcond.get(cond)) > 1:

            data_load = []

            for session_i in range(len(respfeatures_allcond.get(cond))):

                data_load.append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy'))
            
            Cxy_surrogates_allcond[cond] = data_load

            data_load = []

            for session_i in range(len(respfeatures_allcond.get(cond))):

                data_load.append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_lf.npy'))
            
            cyclefreq_surrogates_allcond_lf[cond] = data_load

            data_load = []

            for session_i in range(len(respfeatures_allcond.get(cond))):

                data_load.append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_hf.npy'))
            
            cyclefreq_surrogates_allcond_hf[cond] = data_load

    return Cxy_surrogates_allcond, cyclefreq_surrogates_allcond_lf, cyclefreq_surrogates_allcond_hf


#### compute Pxx & Cxy & Cyclefreq
def compute_PxxCxyCyclefreq_for_cond(band_prep, cond, session_i, nb_point_by_cycle):
    
    print(cond)

    chan_i = chan_list.index('nasal')
    respi = load_data(band_prep, cond, session_i)[chan_i,:]
    data_tmp = load_data(band_prep, cond, session_i)

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    Cxy_for_cond = np.zeros(( np.size(data_tmp,0), len(hzCxy)))
    Pxx_for_cond = np.zeros(( np.size(data_tmp,0), len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( np.size(data_tmp,0), nb_point_by_cycle))

    for n_chan in range(np.size(data_tmp,0)):

        #### script avancement
        if n_chan/np.size(data_tmp,0) % .2 <= 0.01:
            print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

        x_stretch, trash = stretch_data(respfeatures_allcond.get(cond)[session_i], nb_point_by_cycle, x, srate)
        x_stretch_mean = np.mean(x_stretch, 0)

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond

        


def compute_all_PxxCxyCyclefreq():

    Pxx_allcond_lf = {}
    Pxx_allcond_hf = {}
    Cxy_allcond = {}
    cyclefreq_allcond_lf = {}
    cyclefreq_allcond_hf = {}

    for band_prep in band_prep_list:

        print(band_prep)

        for cond in conditions:

            if ( len(respfeatures_allcond.get(cond)) == 1 ) & (band_prep == 'lf'):

                Pxx_load = []
                Cxy_load = []
                cyclefreq_load = []

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep ,cond, 0, stretch_point_surrogates)

                Pxx_load.append(Pxx_for_cond)
                Cxy_load.append(Cxy_for_cond)
                cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond_lf[cond] = Pxx_load
                Cxy_allcond[cond] = Cxy_load
                cyclefreq_allcond_lf[cond] = cyclefreq_load

            elif ( len(respfeatures_allcond.get(cond)) == 1 ) & (band_prep == 'hf') :

                Pxx_load = []
                cyclefreq_load = []

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep ,cond, 0, stretch_point_surrogates)

                Pxx_load.append(Pxx_for_cond)
                cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond_hf[cond] = Pxx_load
                cyclefreq_allcond_hf[cond] = cyclefreq_load


            elif (len(respfeatures_allcond.get(cond)) > 1) & (band_prep == 'lf'):

                Pxx_load = []
                Cxy_load = []
                cyclefreq_load = []

                for session_i in range(len(respfeatures_allcond.get(cond))):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep ,cond, session_i, stretch_point_surrogates)

                    Pxx_load.append(Pxx_for_cond)
                    Cxy_load.append(Cxy_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond_lf[cond] = Pxx_load
                Cxy_allcond[cond] = Cxy_load
                cyclefreq_allcond_lf[cond] = cyclefreq_load

            elif (len(respfeatures_allcond.get(cond)) > 1) & (band_prep == 'hf'):

                Pxx_load = []
                cyclefreq_load = []

                for session_i in range(len(respfeatures_allcond.get(cond))):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep ,cond, session_i, stretch_point_surrogates)

                    Pxx_load.append(Pxx_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond_hf[cond] = Pxx_load
                cyclefreq_allcond_hf[cond] = cyclefreq_load

    return Pxx_allcond_lf, Pxx_allcond_hf, Cxy_allcond, cyclefreq_allcond_lf, cyclefreq_allcond_hf








########################################
######## EXECUTE PSD CXY CF ########
########################################


print('######## COMPUTE PxxCxyCyclefreq ########')

Cxy_surrogates_allcond, cyclefreq_surrogates_allcond_lf, cyclefreq_surrogates_allcond_hf = load_surrogates()
Pxx_allcond_lf, Pxx_allcond_hf, Cxy_allcond, cyclefreq_allcond_lf, cyclefreq_allcond_hf = compute_all_PxxCxyCyclefreq()


#### reduce data to one session
respfeatures_allcond_adjust = {} # to conserve respi_allcond for TF

for cond in conditions:

    if len(respfeatures_allcond.get(cond)) == 1:

        respfeatures_allcond_adjust[cond] = respfeatures_allcond[cond].copy()

    elif len(respfeatures_allcond.get(cond)) > 1:

        data_to_short = []

        for session_i in range(len(respfeatures_allcond.get(cond))):
            
            
            if session_i == 0 :

                data_to_short = [
                                respfeatures_allcond.get(cond)[session_i], 
                                Pxx_allcond_lf.get(cond)[session_i],
                                Pxx_allcond_hf.get(cond)[session_i], 
                                Cxy_allcond.get(cond)[session_i], 
                                Cxy_surrogates_allcond.get(cond)[session_i], 
                                cyclefreq_allcond_lf.get(cond)[session_i],
                                cyclefreq_allcond_hf.get(cond)[session_i], 
                                cyclefreq_surrogates_allcond_lf.get(cond)[session_i]
                                ]

            elif session_i > 0 :

                data_replace = [
                                (data_to_short[0] + respfeatures_allcond.get(cond)[session_i]) / 2, 
                                (data_to_short[1] + Pxx_allcond_lf.get(cond)[session_i]) / 2, 
                                (data_to_short[2] + Pxx_allcond_hf.get(cond)[session_i]) / 2, 
                                (data_to_short[3] + Cxy_allcond.get(cond)[session_i]) / 2, 
                                (data_to_short[4] + Cxy_surrogates_allcond.get(cond)[session_i]) / 2,   
                                (data_to_short[5] + cyclefreq_allcond_lf.get(cond)[session_i]) / 2,
                                (data_to_short[6] + cyclefreq_allcond_hf.get(cond)[session_i]) / 2,  
                                (data_to_short[7] + cyclefreq_surrogates_allcond_lf.get(cond)[session_i]) / 2
                                ]

                data_to_short = data_replace.copy()
        
        # to put in list
        data_load = []
        data_load.append(data_to_short[0])
        respfeatures_allcond_adjust[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[1])
        Pxx_allcond_lf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[2])
        Pxx_allcond_hf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[3])
        Cxy_allcond[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[4])
        Cxy_surrogates_allcond[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[5])
        cyclefreq_allcond_lf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[6])
        cyclefreq_allcond_hf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[7])
        cyclefreq_surrogates_allcond_lf[cond] = data_load 



#### verif if one session only
for cond in conditions :

    verif_size = []

    verif_size.append(len(respfeatures_allcond_adjust[cond]) == 1)
    verif_size.append(len(Pxx_allcond_lf[cond]) == 1)
    verif_size.append(len(Pxx_allcond_hf[cond]) == 1)
    verif_size.append(len(Cxy_allcond[cond]) == 1)
    verif_size.append(len(Cxy_surrogates_allcond[cond]) == 1)
    verif_size.append(len(cyclefreq_allcond_lf[cond]) == 1)
    verif_size.append(len(cyclefreq_allcond_hf[cond]) == 1)
    verif_size.append(len(cyclefreq_surrogates_allcond_lf[cond]) == 1)

    if verif_size.count(False) != 0 :
        print('!!!! PROBLEM VERIF !!!!')
        exit()

    elif verif_size.count(False) == 0 :
        print('Verif OK')






################################################
######## PLOT & SAVE PSD AND COH ########
################################################

os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))

print('######## PLOT & SAVE PSD AND COH ########')

#### def functions
def plot_save_PSD_Coh_lf(n_chan):

    session_i = 0       
    
    chan_name_init = chan_list_ieeg[n_chan]
    chan_name_modif, trash = modify_name([chan_name_init])
    chan_name = str(chan_name_modif[0])

    if n_chan/len(chan_list_ieeg) % .2 <= 0.01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### plot

    if len(conditions) == 1:

        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
        plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

        cond = conditions[0]

        #### supress NaN
        keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
        cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
        respi_mean = round(np.mean(cycle_for_mean), 2)
        
        #### plot
        ax = axs[0]
        ax.set_title(cond, fontweight='bold', rotation=0)
        ax.semilogy(hzPxx,Pxx_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
        ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_lf.get(cond)[session_i][n_chan,:]), color='r')
        ax.set_xlim(0,60)

        ax = axs[1]
        ax.plot(hzPxx,Pxx_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
        ax.set_xlim(0, 2)
        ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_lf.get(cond)[session_i][n_chan,:]), color='r')

        ax = axs[2]
        ax.plot(hzCxy,Cxy_allcond.get(cond)[session_i][n_chan,:], color='k')
        ax.plot(hzCxy,Cxy_surrogates_allcond.get(cond)[session_i][n_chan,:], color='c')
        ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

        ax = axs[3]
        ax.plot(cyclefreq_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][0, n_chan,:], color='b')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
        ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:] ), ymax=np.max( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:] ), colors='r')

    else:

        for c, cond in enumerate(conditions):

            fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            #### supress NaN
            keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
            cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
            respi_mean = round(np.mean(cycle_for_mean), 2)
            
            #### plot
            ax = axs[0,c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx,Pxx_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_lf.get(cond)[session_i][n_chan,:]), color='r')
            ax.set_xlim(0,60)

            ax = axs[1,c]
            ax.plot(hzPxx,Pxx_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
            ax.set_xlim(0, 2)
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_lf.get(cond)[session_i][n_chan,:]), color='r')

            ax = axs[2,c]
            ax.plot(hzCxy,Cxy_allcond.get(cond)[session_i][n_chan,:], color='k')
            ax.plot(hzCxy,Cxy_surrogates_allcond.get(cond)[session_i][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3,c]
            ax.plot(cyclefreq_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
            ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][0, n_chan,:], color='b')
            ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
            ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:] ), ymax=np.max( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:] ), colors='r')


    #### save
    fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
    plt.close()

    return



def plot_save_PSD_Coh_hf(n_chan):    
    
    session_i = 0

    chan_name_init = chan_list_ieeg[n_chan]
    chan_name_modif, trash = modify_name([chan_name_init])
    chan_name = str(chan_name_modif[0])

    if n_chan/len(chan_list_ieeg) % .2 <= 0.01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))

    #### plot

    fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
    plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

    if len(conditions) == 1 :

        for c, cond in enumerate(conditions):

            #### supress NaN
            keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
            cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
            respi_mean = round(np.mean(cycle_for_mean), 2)
            
            #### plot
            ax = axs[0]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx,Pxx_allcond_hf.get(cond)[session_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_hf.get(cond)[session_i][n_chan,:]), color='r')
            ax.set_xlim(45,120)

            ax = axs[1]
            ax.plot(cyclefreq_allcond_hf.get(cond)[session_i][n_chan,:], color='k')
            ax.plot(cyclefreq_surrogates_allcond_hf.get(cond)[session_i][0, n_chan,:], color='b')
            ax.plot(cyclefreq_surrogates_allcond_hf.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(cyclefreq_surrogates_allcond_hf.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
            ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( cyclefreq_surrogates_allcond_hf.get(cond)[session_i][2, n_chan,:] ), ymax=np.max( cyclefreq_surrogates_allcond_hf.get(cond)[session_i][1, n_chan,:] ), colors='r')

    else:

        for c, cond in enumerate(conditions):

            #### supress NaN
            keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
            cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
            respi_mean = round(np.mean(cycle_for_mean), 2)
            
            #### plot
            ax = axs[0,c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx,Pxx_allcond_hf.get(cond)[session_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_hf.get(cond)[session_i][n_chan,:]), color='r')
            ax.set_xlim(45,120)

            ax = axs[1,c]
            ax.plot(cyclefreq_allcond_hf.get(cond)[session_i][n_chan,:], color='k')
            ax.plot(cyclefreq_surrogates_allcond_hf.get(cond)[session_i][0, n_chan,:], color='b')
            ax.plot(cyclefreq_surrogates_allcond_hf.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(cyclefreq_surrogates_allcond_hf.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
            ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( cyclefreq_surrogates_allcond_hf.get(cond)[session_i][2, n_chan,:] ), ymax=np.max( cyclefreq_surrogates_allcond_hf.get(cond)[session_i][1, n_chan,:] ), colors='r')


    #### save
    fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
    plt.close()

    return





#### compute joblib

joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Coh_lf)(n_chan) for n_chan in range(len(chan_list_ieeg)))
joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Coh_hf)(n_chan) for n_chan in range(len(chan_list_ieeg)))







################################
######## LOAD TF ########
################################

#### load and reduce to all cond
os.chdir(os.path.join(path_precompute, sujet, 'TF'))

#### generate str to search file
freq_band_str = {}

for band_prep_i, band_prep in enumerate(band_prep_list):

    freq_band = freq_band_list[band_prep_i]

    for band, freq in freq_band.items():
        freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])


#### load file with reducing to one TF

tf_stretch_allcond = {}

for cond in conditions:

    tf_stretch_onecond = {}

    if len(respfeatures_allcond.get(cond)) == 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### impose good order in dict
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### file load
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:
                        tf_stretch_onecond[band] = np.load(file)
                    else:
                        continue
                    
        tf_stretch_allcond[cond] = tf_stretch_onecond

    elif len(respfeatures_allcond.get(cond)) > 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### implement count
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### load file
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:

                        if np.sum(tf_stretch_onecond.get(band)) != 0:

                            session_load_tmp = ( np.load(file) + tf_stretch_onecond.get(band) ) /2
                            tf_stretch_onecond[band] = session_load_tmp

                        else:
                            
                            tf_stretch_onecond[band] = np.load(file)

                    else:

                        continue

        tf_stretch_allcond[cond] = tf_stretch_onecond




#### verif

for cond in conditions:
    if len(tf_stretch_allcond.get(cond)) != 6:
        print('ERROR COND : ' + cond)

    for freq_band in freq_band_list:

        for band, freq in freq_band.items():
            if len(tf_stretch_allcond.get(cond).get(band)) != len(chan_list_ieeg) :
                print('ERROR FREQ BAND : ' + band)
            






################################
######## SAVE TF ########
################################



os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))

print('######## SAVE TF ########')


def save_TF_n_chan(n_chan):

    chan_name_init = chan_list_ieeg[n_chan]
    chan_name_modif, trash = modify_name([chan_name_init])
    chan_name = str(chan_name_modif[0])

    if n_chan/len(chan_list_ieeg) % .2 <= .01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    time = range(stretch_point_TF)
    frex = np.size(tf_stretch_allcond.get(conditions[0]).get(list(freq_band.keys())[0]),1)

    if len(conditions) == 1:

        if freq_band_i == 0:

            #### plot
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
            plt.close()


        elif freq_band_i == 1:

            #### plot
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
            plt.close()



    else:

        if freq_band_i == 0:

            #### plot
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
            plt.close()


        elif freq_band_i == 1:

            #### plot
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
            plt.close()

    return


#### compute
for freq_band_i, freq_band in enumerate(freq_band_list): 

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_n_chan)(n_chan) for n_chan in range(len(chan_list_ieeg)))




################################
######## LOAD ITPC ########
################################


#### load and reduce to all cond
os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

#### generate str to search file
freq_band_str = {}

for band_prep_i, band_prep in enumerate(band_prep_list):

    freq_band = freq_band_list[band_prep_i]

    for band, freq in freq_band.items():
        freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])

#### load file with reducing to one TF

tf_itpc_allcond = {}

for cond in conditions:

    tf_itpc_onecond = {}

    if len(respfeatures_allcond.get(cond)) == 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### impose good order in dict
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### file load
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):
                    if file.find(freq_band_str.get(band)) != -1:
                        tf_itpc_onecond[ band ] = np.load(file)
                    else:
                        continue
                    
        tf_itpc_allcond[cond] = tf_itpc_onecond

    elif len(respfeatures_allcond.get(cond)) > 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### implement count
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_itpc_onecond[band] = 0

        #### load file
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:

                        if np.sum(tf_itpc_onecond.get(band)) != 0:

                            session_load_tmp = ( np.load(file) + tf_itpc_onecond.get(band) ) /2
                            tf_itpc_onecond[band] = session_load_tmp

                        else:
                            
                            tf_itpc_onecond[band] = np.load(file)

                    else:

                        continue

        tf_itpc_allcond[cond] = tf_itpc_onecond


#### verif

for cond in conditions:
    if len(tf_itpc_allcond.get(cond)) != 6:
        print('ERROR COND : ' + cond)

    for freq_band in freq_band_list:

        for band, freq in freq_band.items():
            if len(tf_itpc_allcond.get(cond).get(band)) != len(chan_list_ieeg) :
                print('ERROR FREQ BAND : ' + band)
            






################################
######## SAVE ITPC ########
################################



os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

print('######## SAVE ITPC ########')

def save_itpc_n_chan(n_chan):       
    
    chan_name_init = chan_list_ieeg[n_chan]
    chan_name_modif, trash = modify_name([chan_name_init])
    chan_name = str(chan_name_modif[0])

    if n_chan/len(chan_list_ieeg) % .2 <= .01:
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    time = range(stretch_point_TF)
    frex = np.size(tf_itpc_allcond.get(conditions[0]).get(list(freq_band.keys())[0]),1)

    if len(conditions) == 1:

        if freq_band_i == 0:

            #### plot
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
            plt.close()

        elif freq_band_i == 1:

            #### plot
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
            plt.close()

    else:

        if freq_band_i == 0:

            #### plot
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)
            plt.close()

        elif freq_band_i == 1:

            #### plot
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
            plt.close()

    return

for freq_band_i, freq_band in enumerate(freq_band_list): 

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_itpc_n_chan)(n_chan) for n_chan in range(len(chan_list_ieeg)))









































####################################################################

if debug == True:

    #### fig PSD COh all cond

    def PSD_Coh_fig_allcond(cond, session_i, respfeatures_allcond, Pxx_allcond, Cxy_allcond, Cxy_surrogates_allcond, cyclefreq_allcond, cyclefreq_surrogates_allcond):

        for cond in conditions:

            for n_chan in range(np.size(chan_list,0)):       
                
                chan_name = chan_list[n_chan]
                print('{:.2f}'.format(n_chan/np.size(chan_list,0)))

                hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
                hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
                mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
                hzCxy = hzCxy[mask_hzCxy]

                respi_mean = round(np.mean(respfeatures_allcond.get(cond)[session_i]['cycle_freq'].values), 2)

                #### plot
                fig, axs = plt.subplots(nrows=2, ncols=2)
                plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))
                        
                ax = axs[0,0]
                ax.set_title('Welch_full', fontweight='bold', rotation=0)
                ax.semilogy(hzPxx,Pxx_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.set_xlim(0,50)

                ax = axs[0,1]
                ax.set_title('Welch_Respi', fontweight='bold', rotation=0)
                ax.plot(hzPxx,Pxx_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.set_xlim(0, 2)
                ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond.get(cond)[session_i][n_chan,:]), color='b')

                ax = axs[1,0]
                ax.set_title('Cxy', fontweight='bold', rotation=0)
                ax.plot(hzCxy,Cxy_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.plot(hzCxy,Cxy_surrogates_allcond.get(cond)[session_i][n_chan,:], color='c')
                ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

                ax = axs[1,1]
                ax.set_title('CycleFreq', fontweight='bold', rotation=0)
                ax.plot(cyclefreq_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.plot(cyclefreq_surrogates_allcond.get(cond)[session_i][0, n_chan,:], color='c')
                ax.plot(cyclefreq_surrogates_allcond.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
                ax.plot(cyclefreq_surrogates_allcond.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')


    ####################################################################################


