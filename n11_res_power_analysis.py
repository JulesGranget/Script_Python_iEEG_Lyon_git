

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

import pickle

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







########################################
######## PSD & COH PRECOMPUTE ########
########################################




#dict2reduce = Cxy_allcond
def reduce_data(dict2reduce, prms):

    #### for Pxx and Cyclefreq
    if np.sum([True for i in list(dict2reduce.keys()) if i in band_prep_list]) > 0:
    
        #### generate dict
        dict_reduced = {}
        for band_prep in band_prep_list:
            dict_reduced[band_prep] = {}
            for cond in prms['conditions']:
                dict_reduced[band_prep][cond] = []

        for band_prep in band_prep_list:

            for cond in prms['conditions']:

                dict_reduced[band_prep][cond].append(dict2reduce[band_prep][cond][0])


        #### verify
        for band_prep in band_prep_list:
            for cond in prms['conditions']:
                if len(dict_reduced[band_prep][cond][0].shape) != 2:
                    raise ValueError(f'reducing false for Pxx or Cyclefreq : {band_prep}, {cond}')

    #### for Cxy
    elif np.sum([True for i in list(dict2reduce.keys()) if i in prms['conditions']]) > 0:

        #### generate dict
        dict_reduced = {}
        for cond in prms['conditions']:
            dict_reduced[cond] = []

        for cond in prms['conditions']:

            dict_reduced[cond].append(dict2reduce[cond][0])

        #### verify
        for cond in prms['conditions']:
            if len(dict_reduced[cond][0].shape) != 2:
                raise ValueError(f'reducing false for Cxy :, {cond}')

    #### for surrogates
    else:
        
        #### generate dict
        dict_reduced = {}
        for key in list(dict2reduce.keys()):
            dict_reduced[key] = {}
            for cond in prms['conditions']:
                dict_reduced[key][cond] = []

        #key = 'Cxy'
        for key in list(dict2reduce.keys()):

            for cond in prms['conditions']:

                dict_reduced[key][cond].append(dict2reduce[key][cond][0])

        #### verify
        for key in list(dict2reduce.keys()):
            if key == 'Cxy':
                for cond in prms['conditions']:
                    if len(dict_reduced[key][cond][0].shape) != 2:
                        raise ValueError(f'reducing false for Surrogates : {key}, {cond}')
            else:
                for cond in prms['conditions']:
                    if len(dict_reduced[key][cond][0].shape) != 3:
                        raise ValueError(f'reducing false for Surrogates : {key}, {cond}')

    return dict_reduced






#### load surrogates
def load_surrogates(sujet, respfeatures_allcond, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {'Cxy' : {}, 'cyclefreq_lf' : {}, 'cyclefreq_hf' : {}}

    for cond in prms['conditions']:

        if len(respfeatures_allcond[cond]) == 1:

            surrogates_allcond['Cxy'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_Coh.npy')]
            surrogates_allcond['cyclefreq_lf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_lf.npy')]
            surrogates_allcond['cyclefreq_hf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_hf.npy')]

        elif len(respfeatures_allcond[cond]) > 1:

            data_load = {'Cxy' : [], 'cyclefreq_lf' : [], 'cyclefreq_hf' : []}

            for session_i in range(len(respfeatures_allcond[cond])):

                data_load['Cxy'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy'))
                data_load['cyclefreq_lf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_lf.npy'))
                data_load['cyclefreq_hf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_hf.npy'))
            
            surrogates_allcond['Cxy'][cond] = data_load['Cxy']
            surrogates_allcond['cyclefreq_lf'][cond] = data_load['cyclefreq_lf']
            surrogates_allcond['cyclefreq_hf'][cond] = data_load['cyclefreq_hf']


    return surrogates_allcond







#### compute Pxx & Cxy & Cyclefreq
def compute_PxxCxyCyclefreq_for_cond(sujet, band_prep, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms):
    
    print(cond)

    #### extract data
    chan_i = prms['chan_list'].index('nasal')
    respi = load_data_sujet(sujet, band_prep, cond, session_i)[chan_i,:]
    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    #### prepare analysis
    hzPxx = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### compute
    Cxy_for_cond = np.zeros(( np.size(data_tmp,0), len(hzCxy)))
    Pxx_for_cond = np.zeros(( np.size(data_tmp,0), len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( np.size(data_tmp,0), stretch_point_surrogates))

    for n_chan in range(np.size(data_tmp,0)):

        x = data_tmp[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        x_stretch, trash = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_surrogates, x, prms['srate'])
        x_stretch_mean = np.mean(x_stretch, 0)

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond

        





def compute_all_PxxCxyCyclefreq(sujet, respfeatures_allcond, prms):

    Pxx_allcond = {'lf' : {}, 'hf' : {}}
    Cxy_allcond = {}
    cyclefreq_allcond = {'lf' : {}, 'hf' : {}}

    #band_prep = band_prep_list[0]
    for band_prep in band_prep_list:

        print(band_prep)

        for cond in prms['conditions']:

            if ( len(respfeatures_allcond[cond]) == 1 ) & (band_prep == 'lf'):

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(sujet, band_prep, cond, 0, stretch_point_surrogates, respfeatures_allcond, prms)

                Pxx_allcond['lf'][cond] = [Pxx_for_cond]
                Cxy_allcond[cond] = [Cxy_for_cond]
                cyclefreq_allcond['lf'][cond] = [cyclefreq_for_cond]

            elif ( len(respfeatures_allcond[cond]) == 1 ) & (band_prep == 'hf') :

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(sujet, band_prep, cond, 0, stretch_point_surrogates, respfeatures_allcond, prms)

                Pxx_allcond['hf'][cond] = [Pxx_for_cond]
                cyclefreq_allcond['hf'][cond] = [cyclefreq_for_cond]

            elif (len(respfeatures_allcond[cond]) > 1) & (band_prep == 'lf'):

                Pxx_load = []
                Cxy_load = []
                cyclefreq_load = []

                for session_i in range(len(respfeatures_allcond[cond])):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(sujet, band_prep, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms)

                    Pxx_load.append(Pxx_for_cond)
                    Cxy_load.append(Cxy_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond['lf'][cond] = Pxx_load
                Cxy_allcond[cond] = Cxy_load
                cyclefreq_allcond['lf'][cond] = cyclefreq_load

            elif (len(respfeatures_allcond[cond]) > 1) & (band_prep == 'hf'):

                Pxx_load = []
                cyclefreq_load = []

                for session_i in range(len(respfeatures_allcond[cond])):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(sujet, band_prep, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms)

                    Pxx_load.append(Pxx_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond['hf'][cond] = Pxx_load
                cyclefreq_allcond['hf'][cond] = cyclefreq_load

    return Pxx_allcond, Cxy_allcond, cyclefreq_allcond






def reduce_PxxCxy_cyclefreq(Pxx_allcond, Cxy_allcond, cyclefreq_allcond, surrogates_allcond, prms):

    Pxx_allcond_red = reduce_data(Pxx_allcond, prms)
    cyclefreq_allcond_red = reduce_data(cyclefreq_allcond, prms)

    Cxy_allcond_red = reduce_data(Cxy_allcond, prms)
    surrogates_allcond_red = reduce_data(surrogates_allcond, prms)
    
    return Pxx_allcond_red, cyclefreq_allcond_red, Cxy_allcond_red, surrogates_allcond_red






def save_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet, Pxx_allcond, cyclefreq_allcond, Cxy_allcond, surrogates_allcond):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    with open(f'{sujet}_Pxx_allcond.pkl', 'wb') as f:
        pickle.dump(Pxx_allcond, f)

    with open(f'{sujet}_Cxy_allcond.pkl', 'wb') as f:
        pickle.dump(Cxy_allcond, f)

    with open(f'{sujet}_surrogates_allcond.pkl', 'wb') as f:
        pickle.dump(surrogates_allcond, f)

    with open(f'{sujet}_cyclefreq_allcond.pkl', 'wb') as f:
        pickle.dump(cyclefreq_allcond, f)




def compute_reduced_PxxCxyCyclefreqSurrogates(sujet, respfeatures_allcond, surrogates_allcond, prms):


    if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'{sujet}_Pxx_allcond.pkl')) == False:
    
        Pxx_allcond, Cxy_allcond, cyclefreq_allcond = compute_all_PxxCxyCyclefreq(sujet, respfeatures_allcond, prms)

        Pxx_allcond, cyclefreq_allcond, Cxy_allcond, surrogates_allcond = reduce_PxxCxy_cyclefreq(Pxx_allcond, Cxy_allcond, cyclefreq_allcond, surrogates_allcond, prms)

        save_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet, Pxx_allcond, cyclefreq_allcond, Cxy_allcond, surrogates_allcond)

        print('COMPUTE Pxx CF Cxy Surr')

    else:

        print('ALREADY COMPUTED')

    print('done') 







################################################
######## PLOT & SAVE PSD AND COH ########
################################################




def get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
        
    with open(f'{sujet}_Pxx_allcond.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'{sujet}_Cxy_allcond.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'{sujet}_surrogates_allcond.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'{sujet}_cyclefreq_allcond.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond



#n_chan = 0
def plot_save_PSD_Coh(sujet, n_chan):

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond = get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet)
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet)

    #### identify chan params
    chan_list_modified, chan_list_keep = modify_name(prms['chan_list_ieeg'])
    chan_name = chan_list_modified[n_chan]
    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]
    
    #### plot
    print_advancement(n_chan, len(prms['chan_list_ieeg']), steps=[25, 50, 75])

    hzPxx = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    band_prep = 'lf'

    #cond_i, cond = 0, 'FR_CV'
    for cond_i, cond in enumerate(prms['conditions']):

        fig, axs = plt.subplots(nrows=4, ncols=len(prms['conditions']))
        plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')

        if len(prms['conditions']) == 1:

            #### identify respi mean
            respi_mean = np.round(respfeatures_allcond[cond][0]['cycle_freq'].mean(), 3)
            
            #### plot
            ax = axs[0]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx, Pxx_allcond['lf'][cond][0][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond['lf'][cond][0][n_chan,:].max(), color='r')
            ax.set_xlim(0,60)

            ax = axs[1]
            ax.semilogy(hzPxx[remove_zero_pad:], Pxx_allcond['lf'][cond][0][n_chan,remove_zero_pad:], color='k')
            ax.set_xlim(0, 2)
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond['lf'][cond][0][n_chan,remove_zero_pad:].max(), color='r')

            ax = axs[2]
            ax.plot(hzCxy,Cxy_allcond[cond][0][n_chan,:], color='k')
            ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][0][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3]
            ax.plot(cyclefreq_allcond['lf'][cond][0][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_lf'][cond][0][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_lf'][cond][0][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:], color='c', linestyle='dotted')
            if stretch_TF_auto:
                ax.vlines(prms['respi_ratio_allcond'][cond][0]*stretch_point_surrogates, ymin=surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:].max(), colors='r')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:].max(), colors='r')
            #plt.show()

        else:
            
            for c, cond in enumerate(prms['conditions']):

                #### identify respi mean
                respi_mean = []
                for trial_i, _ in enumerate(respfeatures_allcond[cond]):
                    respi_mean.append(np.round(respfeatures_allcond[cond][trial_i]['cycle_freq'].mean(), 3))
                respi_mean = np.round(np.mean(respi_mean),3)
                     
                #### plot
                ax = axs[0, c]
                ax.set_title(cond, fontweight='bold', rotation=0)
                ax.semilogy(hzPxx, Pxx_allcond['lf'][cond][0][n_chan,:], color='k')
                ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond['lf'][cond][0][n_chan,:].max(), color='r')
                ax.set_xlim(0,60)

                ax = axs[1, c]
                ax.semilogy(hzPxx[remove_zero_pad:], Pxx_allcond['lf'][cond][0][n_chan,remove_zero_pad:], color='k')
                ax.set_xlim(0, 2)
                ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond['lf'][cond][0][n_chan,remove_zero_pad:].max(), color='r')

                ax = axs[2, c]
                ax.plot(hzCxy,Cxy_allcond[cond][0][n_chan,:], color='k')
                ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][0][n_chan,:], color='c')
                ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

                ax = axs[3, c]
                ax.plot(cyclefreq_allcond['lf'][cond][0][n_chan,:], color='k')
                ax.plot(surrogates_allcond['cyclefreq_lf'][cond][0][0, n_chan,:], color='b')
                ax.plot(surrogates_allcond['cyclefreq_lf'][cond][0][1, n_chan,:], color='c', linestyle='dotted')
                ax.plot(surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:], color='c', linestyle='dotted')
                if stretch_TF_auto:
                    ax.vlines(prms['respi_ratio_allcond'][cond][0]*stretch_point_surrogates, ymin=surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_lf'][cond][0][1, n_chan,:].max(), colors='r')
                else:
                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq_lf'][cond][0][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_lf'][cond][0][1, n_chan,:].max(), colors='r')
                #plt.show() 
        #plt.show()
        
    #### save
    os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))
    fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_{band_prep}.jpeg', dpi=150)
    plt.close('all')
    del fig

    return


    







################################
######## LOAD TF & ITPC ########
################################


def compute_TF_ITPC(sujet, prms):

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
    
        if tf_mode == 'TF':
            print('######## LOAD TF ########')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'{sujet}_tf_stretch_allcond.pkl')):
                print('ALREADY COMPUTED')
                continue
            
        elif tf_mode == 'ITPC':
            print('######## LOAD ITPC ########')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'{sujet}_itpc_stretch_allcond.pkl')):
                print('ALREADY COMPUTED')
                continue

        #### generate str to search file
        freq_band_str = {}

        for band_prep in band_prep_list:

            freq_band = freq_band_dict[band_prep]

            for band, freq in freq_band.items():
                freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])


        #### load file with reducing to one TF
        tf_stretch_allcond = {}

        #band_prep = 'lf'
        for band_prep in band_prep_list:

            tf_stretch_allcond[band_prep] = {}

            #cond = 'FR_CV'
            for cond in prms['conditions']:

                tf_stretch_allcond[band_prep][cond] = {}

                #### generate file to load
                load_file = []
                for file in os.listdir(): 
                    if file.find(cond) != -1:
                        load_file.append(file)
                    else:
                        continue

                #### impose good order in dict
                for band, freq in freq_band_dict[band_prep].items():
                    tf_stretch_allcond[band_prep][cond][band] = 0

                #### file load
                for file in load_file:

                    for i, (band, freq) in enumerate(freq_band_dict[band_prep].items()):

                        if file.find(freq_band_str[band]) != -1:
                            tf_stretch_allcond[band_prep][cond][band] = np.load(file)
                        else:
                            continue
                            

        #### verif
        for band_prep in band_prep_list:
            for cond in prms['conditions']:
                for band, freq in freq_band_dict[band_prep].items():
                    if tf_stretch_allcond[band_prep][cond][band].shape[0] != len(prms['chan_list_ieeg']) :
                        print('ERROR FREQ BAND : ' + band)
                    
        #### save
        if tf_mode == 'TF':
            with open(f'{sujet}_tf_stretch_allcond.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)
        elif tf_mode == 'ITPC':
            with open(f'{sujet}_itpc_stretch_allcond.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)

    print('done')








########################################
######## PLOT & SAVE TF & ITPC ########
########################################


def get_tf_itpc_stretch_allcond(sujet, tf_mode):

    source_path = os.getcwd()

    if tf_mode == 'TF':

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        with open(f'{sujet}_tf_stretch_allcond.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    elif tf_mode == 'ITPC':
        
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        with open(f'{sujet}_itpc_stretch_allcond.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond




#n_chan, tf_mode, band_prep = 0, 'TF', 'lf'
def save_TF_ITPC_n_chan(sujet, n_chan, tf_mode, band_prep):

    #### load prms
    prms = get_params(sujet)
    df_loca = get_loca_df(sujet)

    if tf_mode == 'TF':
        os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
    elif tf_mode == 'ITPC':
        os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))
    
    chan_list_modified, chan_list_keep = modify_name(prms['chan_list_ieeg'])
    chan_name = chan_list_modified[n_chan]
    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

    print_advancement(n_chan, len(prms['chan_list_ieeg']), steps=[25, 50, 75])

    freq_band = freq_band_dict[band_prep]

    #### determine plot scale
    vmaxs = {}
    vmins = {}
    for cond in prms['conditions']:

        scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

        for i, (band, freq) in enumerate(freq_band.items()) :

            if band == 'whole' or band == 'l_gamma':
                continue

            data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))

            scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            scales['median_val'] = np.append(scales['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

        vmin = np.median(scales['median_val']) - median_diff
        vmax = np.median(scales['median_val']) + median_diff

        vmaxs[cond] = vmax
        vmins[cond] = vmin

    del scales

    #### plot
    fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(prms['conditions']))
    plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')

    #### for plotting l_gamma down
    if band_prep == 'hf':
        keys_list_reversed = list(freq_band.keys())
        keys_list_reversed.reverse()
        freq_band_reversed = {}
        for key_i in keys_list_reversed:
            freq_band_reversed[key_i] = freq_band[key_i]
        freq_band = freq_band_reversed

    for c, cond in enumerate(prms['conditions']):

        #### plot
        for i, (band, freq) in enumerate(freq_band.items()) :

            data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))
        
            if len(conditions_allsubjects) == 1:
                ax = axs[i]
            else:
                ax = axs[i,c]

            if i == 0 :
                ax.set_title(cond, fontweight='bold', rotation=0)

            time = range(stretch_point_TF)

            if tf_mode == 'TF':
                ax.pcolormesh(time, frex, data, vmin=vmins[cond], vmax=vmaxs[cond], shading='gouraud', cmap=plt.get_cmap('seismic'))
                # ax.pcolormesh(time, frex, data, shading='gouraud', cmap=plt.get_cmap('seismic'))
            if tf_mode == 'ITPC':
                ax.pcolormesh(time, frex, data, vmin=vmins[cond], vmax=vmaxs[cond], shading='gouraud', cmap=plt.get_cmap('seismic'))
                # ax.pcolormesh(time, frex, data, shading='gouraud', cmap=plt.get_cmap('seismic'))

            if c == 0:
                ax.set_ylabel(band)

            
            if stretch_TF_auto:
                ax.vlines(prms['respi_ratio_allcond'][cond][0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')

    #plt.show()

    del data

    #### save
    fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_{band_prep}.jpeg', dpi=150)
    plt.close('all')
    del fig











########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq(sujet):
    
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
        
    surrogates_allcond = load_surrogates(sujet, respfeatures_allcond, prms)

    compute_reduced_PxxCxyCyclefreqSurrogates(sujet, respfeatures_allcond, surrogates_allcond, prms)
    
    #### compute joblib
    print('######## PLOT & SAVE PSD AND COH ########')

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Coh)(sujet, n_chan) for n_chan in range(len(prms['chan_list_ieeg'])))

    print('done')

    

def compilation_compute_TF_ITPC(sujet):

    prms = get_params(sujet)

    compute_TF_ITPC(sujet, prms)
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########')
        if tf_mode == 'ITPC':
            print('######## PLOT & SAVE ITPC ########')
        
        #band_prep = 'lf'
        for band_prep in band_prep_list: 

            print(band_prep)

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan)(sujet, n_chan, tf_mode, band_prep) for n_chan, tf_mode, band_prep in zip(range(len(prms['chan_list_ieeg'])), [tf_mode]*len(prms['chan_list_ieeg']), [band_prep]*len(prms['chan_list_ieeg'])))

    print('done')




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    
    #### Pxx Cxy CycleFreq
    # compilation_compute_Pxx_Cxy_Cyclefreq(sujet)
    execute_function_in_slurm_bash('n11_res_power_analysis', 'compilation_compute_Pxx_Cxy_Cyclefreq', [sujet])
    # execute_function_in_slurm_bash_mem_choice('n8_res_power_analysis', 'compilation_compute_Pxx_Cxy_Cyclefreq', [sujet], 15)


    #### TF & ITPC
    # compilation_compute_TF_ITPC(sujet)
    execute_function_in_slurm_bash('n11_res_power_analysis', 'compilation_compute_TF_ITPC', [sujet])
    # execute_function_in_slurm_bash_mem_choice('n8_res_power_analysis', 'compilation_compute_TF_ITPC', [sujet], 15)






