

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.plts import plot_cyclepoints_array

from n0_config import *

debug = False




########################################
######## COMPUTE RESPI FEATURES ########
########################################

#resp_sig, sr, t_start, condition = raw_allcond[cond][session_i].get_data()[respi_i, :], srate, 0, cond
def analyse_resp(resp_sig, sr, t_start, condition):
    
    # compute signal features
        # indicate if inspiration is '+' or '-'
        # for abdominal belt iEEG inspi = '-'
        # for abdominal belt EEG inspi = '-'
        # for nasal thermistance inspi = '+'
    cycle_indexes = respirationtools.detect_respiration_cycles(resp_sig, sr, t_start=t_start, output = 'index',
                                                    inspiration_sign = '+',
                                                    # baseline
                                                    #baseline_with_average = False,
                                                    baseline_with_average = True,
                                                    manual_baseline = 0.,

                                                    high_pass_filter = None,
                                                    constrain_frequency = None,
                                                    median_windows_filter = None,

                                                    # clean
                                                    eliminate_time_shortest_ratio = 8,
                                                    eliminate_amplitude_shortest_ratio = 4,
                                                    eliminate_mode = 'OR', ) # 'AND')
    


    resp_sig_mc = resp_sig.copy()
    resp_sig_mc -= np.mean(resp_sig_mc)
    resp_features = respirationtools.get_all_respiration_features(resp_sig_mc, sr, cycle_indexes, t_start = 0.)
    #print(resp_features.columns)
    
    cycle_amplitudes = resp_features['total_amplitude'].values
    cycle_durations = resp_features['cycle_duration'].values # idem as : cycle_durations = np.diff(cycle_indexes[:, 0])/sr
    cycle_freq = resp_features['cycle_freq'].values
    
    # figure 
    
    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(condition)
    times = np.arange(resp_sig.size)/sr
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, resp_sig)
    ax.plot(times[cycle_indexes[:, 0]], resp_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r', label='inspi')
    ax.plot(times[cycle_indexes[:, 1]], resp_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g', label='expi')
    #ax.set_xlim(0,120)
    ax.set_ylabel('resp')
    ax.legend()
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(condition)

        # histogram cycle freq
    ax = axs[0]
    count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
    ax.plot(bins[:-1], count)
    ax.set_xlim(0,.6)
    ax.set_ylabel('n')
    ax.set_xlabel('freq')
    ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
    W, pval = scipy.stats.shapiro(cycle_freq)
    ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
    ax.legend()
    
        # histogram inspi/expi ratio
    ax = axs[1]
    ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()

    plt.close()
   
    return resp_features, fig0, fig1
    







def analyse_resp_debug(resp_sig, sr, t_start, condition, params):

    if params['mean_smooth'] :
        if (2*sr)%2 != 1:
            win = 2*sr + 1
        else:
            win = 2*sr
        resp_sig = scipy.signal.savgol_filter(resp_sig, win, 5)
    
    # compute signal features
        # indicate if inspiration is '+' or '-'
        # for abdominal belt inspi = '-'
        # for nasal thermistance inspi = '+'
    cycle_indexes = respirationtools.detect_respiration_cycles(resp_sig, sr, t_start=t_start, output = 'index',
                                                    inspiration_sign = '+',
                                                    # baseline
                                                    #baseline_with_average = False,
                                                    baseline_with_average = params.get('baseline_with_average'),
                                                    manual_baseline = params.get('manual_baseline'),

                                                    high_pass_filter = params.get('high_pass_filter'),
                                                    constrain_frequency = params.get('constrain_frequency'),
                                                    median_windows_filter = params.get('median_windows_filter'),

                                                    # clean
                                                    eliminate_time_shortest_ratio = params.get('eliminate_time_shortest_ratio'),
                                                    eliminate_amplitude_shortest_ratio = params.get('eliminate_amplitude_shortest_ratio'),
                                                    eliminate_mode = params.get('eliminate_mode') )
    


    resp_sig_mc = resp_sig.copy()
    resp_sig_mc -= np.mean(resp_sig_mc)
    resp_features = respirationtools.get_all_respiration_features(resp_sig_mc, sr, cycle_indexes, t_start = 0.)
    #print(resp_features.columns)
    
    cycle_amplitudes = resp_features['total_amplitude'].values
    cycle_durations = resp_features['cycle_duration'].values # idem as : cycle_durations = np.diff(cycle_indexes[:, 0])/sr
    cycle_freq = resp_features['cycle_freq'].values
    
    # figure 
    
    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(condition)
    times = np.arange(resp_sig.size)/sr
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, resp_sig)
    ax.plot(times[cycle_indexes[:, 0]], resp_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r')
    ax.plot(times[cycle_indexes[:, 1]], resp_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g')
    #ax.set_xlim(0,120)
    ax.set_ylabel('resp')
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(condition)

        # histogram cycle freq
    ax = axs[0]
    count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
    ax.plot(bins[:-1], count)
    ax.set_xlim(0,.6)
    ax.set_ylabel('n')
    ax.set_xlabel('freq')
    ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
    W, pval = scipy.stats.shapiro(cycle_freq)
    ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
    ax.legend()
    
        # histogram inspi/expi ratio
    ax = axs[1]
    ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()

    plt.close()
   
    return resp_features, fig0, fig1
    





#sig = respi_sig
def detection_bycycle(sig, srate):

    if debug:
        plt.plot(sig)
        plt.show()

    #### filter
    sig_low = mne.filter.filter_data(sig, srate, l_freq, h_freq, filter_length='auto', verbose='CRITICAL')

    if debug:
        plt.plot(sig)
        plt.plot(sig_low)
        plt.show()

    #### detect
    peaks, troughs = find_extrema(sig_low, srate, f_theta)
    rises, decays = find_zerox(sig_low, peaks, troughs)

    #### adjust detection
    if np.where(decays < rises[0])[0].shape[0] != 0:
        point_delete = np.where(decays < rises[0])[0][-1]
        decays = decays[point_delete+1:]

    if np.where(peaks < rises[0])[0].shape[0] != 0:
        point_delete = np.where(peaks < rises[0])[0][-1]
        peaks = peaks[point_delete+1:]

    if np.where(troughs < rises[0])[0].shape[0] != 0:
        point_delete = np.where(troughs < rises[0])[0][-1]
        troughs = troughs[point_delete+1:]

    if np.where(rises > troughs[-1])[0].shape[0] != 0:
        point_delete = np.where(rises > troughs[-1])[0][0]
        rises = rises[:point_delete]

    if np.where(peaks > troughs[-1])[0].shape[0] != 0:
        point_delete = np.where(peaks > troughs[-1])[0][0]
        peaks = peaks[:point_delete]

    if np.where(decays > troughs[-1])[0].shape[0] != 0:
        point_delete = np.where(decays > troughs[-1])[0][0]
        decays = decays[:point_delete]

    #### verif
    shapes_verif = [rises.shape[0], decays.shape[0], peaks.shape[0], troughs.shape[0]]
    if np.mean(shapes_verif) != int(np.mean(shapes_verif)):
        raise ValueError('incorrect detection')

    #### generate df
    data_detection = {'cycle_num' : range(rises.shape[0]), 'inspi_index' : rises, 'expi_index' : decays, 
    'inspi_time' : rises/srate, 'expi_time' : decays/srate, 'peaks' : peaks, 'troughs' : troughs, 'select' : [1]*rises.shape[0]}
    
    df_detection = pd.DataFrame(data_detection, columns=['cycle_num', 'inspi_index', 'expi_index', 'inspi_time', 'expi_time', 'peaks', 'troughs', 'select'])

    df_detection['cycle_duration'] = np.append(np.diff(df_detection['inspi_time']), np.round( np.mean( np.diff(df_detection['inspi_time'])), 2 ) )
    df_detection['insp_duration'] = df_detection['expi_time'] - df_detection['inspi_time']
    df_detection['exp_duration'] = df_detection['cycle_duration'] - df_detection['insp_duration']
    df_detection['cycle_freq'] = 1/df_detection['cycle_duration']

    df_detection['insp_amplitude'] = sig[df_detection['peaks'].values] - sig[df_detection['inspi_index'].values]
    df_detection['exp_amplitude'] = np.abs(sig[df_detection['troughs'].values] - sig[df_detection['expi_index'].values])
    df_detection['total_amplitude'] = df_detection['insp_amplitude'] + df_detection['exp_amplitude']

    #### verif
    if debug:
        plot_cyclepoints_array(sig_low, srate, peaks=df_detection['peaks'], troughs=df_detection['troughs'], 
        rises=df_detection['inspi_index'], decays=df_detection['expi_index'])
        plt.show()

    #### supress cycle freq based
    mean_freq = np.mean(df_detection['cycle_freq'].values)
    std_freq = np.std(df_detection['cycle_freq'].values)

    delete_i_freq = np.where( (df_detection['cycle_freq'].values > mean_freq + SD_delete_cycles_freq*std_freq) | (df_detection['cycle_freq'].values < mean_freq - SD_delete_cycles_freq*std_freq) )[0]

    df_detection['select'][delete_i_freq] = 0
    df_detection_deleted = df_detection.loc[delete_i_freq]

    #### supress cycle amp based
    mean_amp = np.mean(df_detection['total_amplitude'].values)
    std_amp = np.std(df_detection['total_amplitude'].values)

    delete_i_amp = np.where( (df_detection['total_amplitude'].values > mean_amp + SD_delete_cycles_amp*std_amp) | (df_detection['total_amplitude'].values < mean_amp - SD_delete_cycles_amp*std_amp) )[0]

    df_detection_deleted = pd.concat([df_detection_deleted, df_detection.loc[delete_i_amp]])
    df_detection['select'][delete_i_amp] = 0

    #### verif
    if debug:
        plot_cyclepoints_array(sig_low, srate, peaks=df_detection['peaks'], troughs=df_detection['troughs'], 
        rises=df_detection['inspi_index'], decays=df_detection['expi_index'])
        plt.show()
        
        plot_cyclepoints_array(sig_low, srate, peaks=df_detection_deleted['peaks'], troughs=df_detection_deleted['troughs'], 
        rises=df_detection_deleted['inspi_index'], decays=df_detection_deleted['expi_index'])
        plt.show()

    

    return df_detection





#df_detection, respi_sig = detection_bycycle(respi_sig, srate), respi_sig
def correct_resp_features(respi_sig, df_detection, srate):

    cycle_indexes = np.concatenate((df_detection['inspi_index'][df_detection['select'] == 1].values.reshape(-1,1), 
                                    df_detection['expi_index'][df_detection['select'] == 1].values.reshape(-1,1)), axis=1)
    cycle_freq = df_detection['cycle_freq'][df_detection['select'] == 1].values
    cycle_amplitudes = df_detection['total_amplitude'][df_detection['select'] == 1].values

    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(cond)
    times = np.arange(respi_sig.size)/srate
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, respi_sig)
    ax.plot(times[cycle_indexes[:, 0]], respi_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r', label='inspi')
    ax.plot(times[cycle_indexes[:, 1]], respi_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g', label='expi')
    ax.set_ylabel('resp')
    ax.legend()
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()
    # plt.show()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(cond)

        # histogram cycle freq
    ax = axs[0]
    count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
    ax.plot(bins[:-1], count)
    ax.set_xlim(0,.6)
    ax.set_ylabel('n')
    ax.set_xlabel('freq')
    ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
    W, pval = scipy.stats.shapiro(cycle_freq)
    ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
    ax.legend()
    
        # histogram inspi/expi ratio
    ax = axs[1]
    ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()
    # plt.show()

    plt.close()

    return [df_detection, fig0, fig1]











########################################
######## VERIF RESPIFEATURES ########
########################################



if __name__ == '__main__':

    ############################
    ######## PARAMETERS ########
    ############################

    from n0_config import *
    #### whole protocole
    sujet = 'CHEe'
    # sujet = 'GOBc' 
    #sujet = 'MAZm' 
    #sujet = 'TREt' 

    #### FR_CV only
    #sujet = 'MUGa'
    #sujet = 'BANc'
    #sujet = 'LEMl'
    #sujet = 'pat_02459_0912'
    #sujet = 'pat_02476_0929'
    #sujet = 'pat_02495_0949'

    ############################
    ######## LOAD DATA ########
    ############################


    #### adjust conditions
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    dirlist_subject = os.listdir()

    cond_keep = []
    for cond in conditions_allsubjects:

        for file in dirlist_subject:

            if file.find(cond) != -1 : 
                cond_keep.append(cond)
                break

    conditions = cond_keep

    #### load data
    raw_allcond = {}

    for cond in conditions:

        load_i = []
        for session_i, session_name in enumerate(os.listdir()):
            if session_name.find(cond) > 0 and session_name.find('lf') != -1 :
                load_i.append(session_i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        data = []
        for load_name in load_list:
            data.append(mne.io.read_raw_fif(load_name, preload=True))

        raw_allcond[cond] = data


    srate = int(raw_allcond[os.listdir()[0][5:10]][0].info['sfreq'])
    chan_list = raw_allcond[os.listdir()[0][5:10]][0].info['ch_names']

    #### compute
    respi_allcond = {}
    for cond in conditions:
        
        data = []
        for session_i in range(len(raw_allcond[cond])):
            if cond == 'FR_MV' :
                respi_i = chan_list.index('ventral')
            else :
                respi_i = chan_list.index('nasal')

            data.append(analyse_resp(raw_allcond[cond][session_i].get_data()[respi_i, :], srate, 0, cond))

        respi_allcond[cond] = data



    respi_allcond_bybycle = {}
    for cond in conditions:
        
        data = []
        for session_i in range(len(raw_allcond[cond])):
            if cond == 'FR_MV' :
                respi_i = chan_list.index('ventral')
            else :
                respi_i = chan_list.index('nasal')
            
            respi_sig = raw_allcond[cond][session_i].get_data()[respi_i, :]
            resp_features_i = correct_resp_features(respi_sig, detection_bycycle(respi_sig, srate), srate)
            data.append(resp_features_i)

        respi_allcond_bybycle[cond] = data



    ########################################
    ######## VERIF RESPIFEATURES ########
    ########################################
    
    if debug == True :

        # info to debug
        cond_len = {}
        for cond in conditions:
            cond_len[cond] = len(respi_allcond[cond])
        
        cond_len
        cond = 'RD_CV' 
        # cond = 'RD_FV' 
        # cond = 'RD_SV'
        # cond = 'RD_AV'
        # cond = 'FR_CV'
        # cond = 'FR_MV'
        
        session_i = 0

        respi_allcond[cond][session_i][1].show()
        respi_allcond[cond][session_i][2].show()

        respi_allcond_bybycle[cond][session_i][1].show()
        respi_allcond_bybycle[cond][session_i][2].show()

        #### recompute
        params = {

        'mean_smooth' : True,

        'baseline_with_average' : True,
        'manual_baseline' : 0.,

        'high_pass_filter' : True,
        'constrain_frequency' : None,
        'median_windows_filter' : False,

        'eliminate_time_shortest_ratio' : 8,
        'eliminate_amplitude_shortest_ratio' : 10,
        'eliminate_mode' : 'OR'

        }

        #### adjust for MOUTH VENTILATION
        if cond == 'FR_MV':
            respi_i = chan_list.index('ventral')
        else:
            respi_i = chan_list.index('nasal')

        resp_features, fig0, fig1 = analyse_resp_debug(raw_allcond[cond][session_i].get_data()[respi_i, :], srate, 0, cond, params)
        fig0.show()
        fig1.show()

        #### changes
        # CHEe : 'eliminate_time_shortest_ratio' : 2, 'eliminate_amplitude_shortest_ratio' : 10, 'baseline_with_average' : False
        # TREt : 'median_windows_filter' : True, 'eliminate_time_shortest_ratio' : 8, 'eliminate_amplitude_shortest_ratio' : 10, 'mean_smooth' : True

        #### replace
        respi_allcond[cond][session_i] = [resp_features, fig0, fig1]



    ################################
    ######## SAVE FIG ########
    ################################


    #### when everything ok classic
    os.chdir(os.path.join(path_results, sujet, 'RESPI'))

    for cond_i in conditions:

        for i in range(len(respi_allcond[cond_i])):

            respi_allcond[cond_i][i][0].to_excel(sujet + '_' + cond_i + '_' + str(i+1) + '_respfeatures.xlsx')
            respi_allcond[cond_i][i][1].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig0.jpeg')
            respi_allcond[cond_i][i][2].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig1.jpeg')

    #### when everything ok bycycle
    os.chdir(os.path.join(path_results, sujet, 'RESPI'))

    for cond_i in conditions:

        for i in range(len(respi_allcond_bybycle[cond_i])):

            respi_allcond_bybycle[cond_i][i][0].to_excel(sujet + '_' + cond_i + '_' + str(i+1) + '_respfeatures.xlsx')
            respi_allcond_bybycle[cond_i][i][1].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig0.jpeg')
            respi_allcond_bybycle[cond_i][i][2].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig1.jpeg')

