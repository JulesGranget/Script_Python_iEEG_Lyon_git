


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import seaborn as sns

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







########################################
######## ALLPLOT ANATOMY ######## 
########################################

def get_all_ROI_and_Lobes_name():

    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')
    
    #### fill dict with anat names
    anat_loca_dict = {}
    anat_lobe_dict = {}
    anat_loca_list = nomenclature_df['Our correspondances'].values
    anat_lobe_list_non_sorted = nomenclature_df['Lobes'].values
    for i in range(len(anat_loca_list)):
        anat_loca_dict[anat_loca_list[i]] = {'TF' : {}, 'ITPC' : {}}
        anat_lobe_dict[anat_lobe_list_non_sorted[i]] = {'TF' : {}, 'ITPC' : {}}

    return anat_loca_dict, anat_lobe_dict




########################################
######## PREP ALLPLOT ANALYSIS ########
########################################



def get_ROI_Lobes_list_and_Plots(FR_CV_compute=False):

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = np.unique(nomenclature_df['Our correspondances'].values)
    lobe_list = np.unique(nomenclature_df['Lobes'].values)

    #### fill dict with anat names
    ROI_dict_count = {}
    ROI_dict_plots = {}
    for i, _ in enumerate(ROI_list):
        ROI_dict_count[ROI_list[i]] = 0
        ROI_dict_plots[ROI_list[i]] = []

    lobe_dict_count = {}
    lobe_dict_plots = {}
    for i, _ in enumerate(lobe_list):
        lobe_dict_count[lobe_list[i]] = 0
        lobe_dict_plots[lobe_list[i]] = []

    #### initiate for cond
    sujet_for_cond = []

    #### search for ROI & lobe that have been counted

    if FR_CV_compute:
        sujet_list_selected = sujet_list_FR_CV
    else:
        sujet_list_selected = sujet_list

    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')

        chan_list_ieeg = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        #nchan = chan_list_ieeg_csv[0]
        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            ROI_dict_count[ROI_tmp] = ROI_dict_count[ROI_tmp] + 1
            lobe_dict_count[lobe_tmp] = lobe_dict_count[lobe_tmp] + 1
            count_verif += 1

            ROI_dict_plots[ROI_tmp].append([sujet_i, nchan])
            lobe_dict_plots[lobe_tmp].append([sujet_i, nchan])

        #### verif count
        if count_verif != len(chan_list_ieeg):
            raise ValueError('ERROR : anatomical count is not correct, count != len chan_list')

    #### exclude ROi and Lobes with 0 counts
    ROI_to_include = [ROI_i for ROI_i in ROI_list if ROI_dict_count[ROI_i] > 0]
    lobe_to_include = [Lobe_i for Lobe_i in lobe_list if lobe_dict_count[Lobe_i] > 0]

    return ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots









################################
######## TF & ITPC ########
################################


#ROI_to_process = 'amygdala'
def get_TF_and_ITPC_for_ROI(ROI_to_process, cond):

    #### load anat
    ROI_list_allband, Lobe_list_allband = get_all_ROI_and_Lobes_name()
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(FR_CV_compute=True)

    #### identify stretch point
    stretch_point = stretch_point_TF

    #### identify if need to be proccessed
    if (ROI_to_process in ROI_to_include) == False:
        return

    print(ROI_to_process)

    #### plot to compute
    plot_to_process = ROI_dict_plots[ROI_to_process]

    #### identify sujet that participate
    sujet_that_participate = []
    for plot_sujet_i, plot_plot_i in plot_to_process:
        if plot_sujet_i in sujet_that_participate:
            continue
        else:
            sujet_that_participate.append(plot_sujet_i)

    #### generate dict for loading TF
    dict_TF_for_ROI_to_process = {}
    dict_ITPC_for_ROI_to_process = {}
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_ITPC_for_ROI_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_ITPC_for_ROI_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### initiate len recorded
    len_recorded = []
    
    #### compute TF
    #plot_to_process_i = plot_to_process[0]    
    for plot_to_process_num, plot_to_process_i in enumerate(plot_to_process):

        # print_advancement(plot_to_process_num, len(plot_to_process), steps=[25, 50, 75])
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        #### load subject params
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects)
        if sujet_tmp[:3] != 'pat':
            chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

        #### identify plot name
        plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### add length recorded
        len_recorded.append(load_data_sujet(sujet_tmp, 'lf', cond, 0)[plot_tmp_i,:].shape[0]/srate/60)

        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))

        #### identify trial number
        band, freq = list(dict_freq_band.items())[0]
        n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1])

        #### load TF and mean trial
        #band, freq = 'theta', [2, 10]
        for band, freq in dict_freq_band.items():
    
            #trial_i = 0
            for trial_i in range(n_trials):
                
                if trial_i == 0:

                    TF_load = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')

                else:

                    TF_load += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')
            
            #### average trials TF
            TF_load /= n_trials
            TF_load_zscore = zscore_mat(TF_load[plot_tmp_i,:,:])

            dict_TF_for_ROI_to_process[band] = (dict_TF_for_ROI_to_process[band] + TF_load_zscore)

            #### verif
            if debug:
                plt.pcolormesh(dict_TF_for_ROI_to_process[band])
                plt.show()

        #### load ITPC and mean trial
        os.chdir(os.path.join(path_precompute, sujet_tmp, 'ITPC'))
        for band, freq in dict_freq_band.items():
    
            for trial_i in range(n_trials):
                
                if trial_i == 0:

                    ITPC_load = np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')

                else:

                    ITPC_load += np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')
            
            #### average trials ITPC
            ITPC_load /= n_trials
            ITPC_load_zscore = zscore_mat(ITPC_load[plot_tmp_i,:,:])

            dict_ITPC_for_ROI_to_process[band] = (dict_ITPC_for_ROI_to_process[band] + ITPC_load_zscore)

    #### mean
    for band, freq in dict_freq_band.items():
        dict_TF_for_ROI_to_process[band] /= len(plot_to_process)
        dict_ITPC_for_ROI_to_process[band] /= len(plot_to_process)

    #### verif
    if debug:
        band = 'theta'
        plt.pcolormesh(dict_TF_for_ROI_to_process[band])
        plt.show()

    return dict_ITPC_for_ROI_to_process, dict_TF_for_ROI_to_process







# Lobe_to_process = 'Occipital'
def get_TF_and_ITPC_for_Lobe(Lobe_to_process, cond):

    #### load anat
    ROI_list_allband, Lobe_list_allband = get_all_ROI_and_Lobes_name()
    len_ROI, len_Lobes = len(list(ROI_list_allband.keys())), len(list(Lobe_list_allband.keys()))
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(FR_CV_compute=True)

    #### identify stretch point
    stretch_point = stretch_point_TF

    #### identify if proccessed
    if (Lobe_to_process in lobe_to_include) != True:
        return

    print(Lobe_to_process)

    #### plot to compute
    plot_to_process = lobe_dict_plots[Lobe_to_process]

    #### identify sujet that participate
    sujet_that_participate = []
    for plot_sujet_i, plot_plot_i in plot_to_process:
        if plot_sujet_i in sujet_that_participate:
            continue
        else:
            sujet_that_participate.append(plot_sujet_i)

    #### generate dict for loading TF
    dict_TF_for_Lobe_to_process = {}
    dict_ITPC_for_Lobe_to_process = {}
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_ITPC_for_Lobe_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_ITPC_for_Lobe_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### initiate len recorded
    len_recorded = []
    
    #### compute TF
    # plot_to_process_i = plot_to_process[0]    
    for plot_to_process_i in plot_to_process:
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        #### load subject params
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects)
        chan_list_modified, chan_list_keep = modify_name(chan_list_ieeg)

        #### identify plot name in trc
        if sujet_tmp[:3] != 'pat':
            list_mod, list_trc = modify_name(chan_list_ieeg)
            plot_tmp = list_trc[list_mod.index(plot_tmp_mod)]
        else:
            plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### add length recorded
        len_recorded.append(load_data_sujet(sujet_tmp, 'lf', cond, 0)[plot_tmp_i,:].shape[0]/srate/60)

        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))

        #### identify trial number
        band, freq = list(dict_freq_band.items())[0]
        n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1])

        #### load TF and mean trial
        for band, freq in dict_freq_band.items():
    
            for trial_i in range(n_trials):
                
                if trial_i == 0:

                   TF_load = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')

                else:

                    TF_load += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')
            
            #### average trials TF and normalize
            TF_load /= n_trials
            TF_load_zscore = zscore_mat(TF_load[plot_tmp_i,:,:])

            dict_TF_for_Lobe_to_process[band] = (dict_TF_for_Lobe_to_process[band] + TF_load_zscore)

        #### load ITPC and mean trial
        os.chdir(os.path.join(path_precompute, sujet_tmp, 'ITPC'))
        for band, freq in dict_freq_band.items():
    
            for trial_i in range(n_trials):
                
                if trial_i == 0:

                   ITPC_load = np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')

                else:

                    ITPC_load += np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')
            
            #### average trials ITPC
            ITPC_load /= n_trials
            ITPC_load_zscore = zscore_mat(ITPC_load[plot_tmp_i,:,:])

            dict_ITPC_for_Lobe_to_process[band] = (dict_ITPC_for_Lobe_to_process[band] + ITPC_load_zscore)

    #### mean
    for band, freq in dict_freq_band.items():
        dict_TF_for_Lobe_to_process[band] /= len(plot_to_process)
        dict_ITPC_for_Lobe_to_process[band] /= len(plot_to_process)


    return dict_ITPC_for_Lobe_to_process, dict_TF_for_Lobe_to_process

















################################
######## COMPILATION ########
################################



def compilation_allplot_analysis(cond):

    #### verify computation
    os.chdir(os.path.join(path_precompute, 'allplot'))
    band = list(freq_band_list[0].keys())[0]
    if os.path.exists(f'ROI_{cond}_{band}_allband.npy') and os.path.exists(f'Lobes_{cond}_{band}_allband.npy'):
        return

    #### load anat
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(FR_CV_compute=True)
        
    #### identify stretch point
    stretch_point = stretch_point_TF
    
    #### compute TF & ITPC for ROI
    print('#### TF and ITPC for ROI ####')
    res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_TF_and_ITPC_for_ROI)(ROI_to_process, cond) for ROI_to_process in ROI_to_include)

    #### extract & save
    os.chdir(os.path.join(path_precompute, 'allplot'))
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        for band in list(freq_band_dict.keys()):

            ROI_band = np.zeros((len(ROI_to_include), 2, nfrex_hf, stretch_point))

            #ROI_to_process_i = 5
            for ROI_to_process_i, ROI_to_process in enumerate(ROI_to_include):

                ROI_band[ROI_to_process_i, 0, :, :] = res[ROI_to_process_i][0][band]
                ROI_band[ROI_to_process_i, 1, :, :] = res[ROI_to_process_i][1][band]

                if debug:
                    plt.pcolormesh(ROI_band[1,1,:,:])
                    plt.show()

            np.save(f'ROI_TF_ITPC_{cond}_{band}_allband.npy', ROI_band)


    #### compute TF & ITPC for Lobes
    print('#### TF and ITPC for Lobe ####')
    res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_TF_and_ITPC_for_Lobe)(Lobe_to_process, cond) for Lobe_to_process in lobe_to_include)

    #### extract
    os.chdir(os.path.join(path_precompute, 'allplot'))
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        for band in list(freq_band_dict.keys()):

            Lobe_band = np.zeros((len(lobe_to_include), 2, nfrex_hf, stretch_point))

            for Lobe_to_process_i, Lobe_to_process in enumerate(lobe_to_include):

                Lobe_band[Lobe_to_process_i, 0, :, :] = res[Lobe_to_process_i][0][band]
                Lobe_band[Lobe_to_process_i, 1, :, :] = res[Lobe_to_process_i][1][band]

            np.save(f'Lobes_TF_ITPC_{cond}_{band}_allband.npy', Lobe_band)

    print('done')

    

        








################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    cond = 'FR_CV'

    # compilation_allplot_analysis(cond)
    execute_function_in_slurm_bash('n15_precompute_allplot_TF_FR_CV', 'compilation_allplot_analysis', [cond])
    



 