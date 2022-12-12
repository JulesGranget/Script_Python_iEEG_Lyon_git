
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import xarray as xr

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



def get_ROI_Lobes_list_and_Plots(monopol):

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = list(np.unique(nomenclature_df['Our correspondances'].values))
    lobe_list = list(np.unique(nomenclature_df['Lobes'].values))

    #### fill dict with anat names
    ROI_dict = {}
    ROI_dict_plots = {}

    for ROI_i in ROI_list:
        ROI_dict[ROI_i] = 0
        ROI_dict_plots[ROI_i] = []

    lobe_dict = {}
    lobe_dict_plots = {}

    for lobe_i in lobe_list:
        lobe_dict[lobe_i] = 0
        lobe_dict_plots[lobe_i] = []

    #### search for ROI & lobe that have been counted
    sujet_list_selected = sujet_list

    #sujet_i = sujet_list_selected[0]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        if monopol:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        else:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')
            
        chan_list_ieeg = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            ROI_dict[ROI_tmp] = ROI_dict[ROI_tmp] + 1
            lobe_dict[lobe_tmp] = lobe_dict[lobe_tmp] + 1
            count_verif += 1

            ROI_dict_plots[ROI_tmp].append([sujet_i, nchan])
            lobe_dict_plots[lobe_tmp].append([sujet_i, nchan])

        #### verif count
        if count_verif != len(chan_list_ieeg):
            raise ValueError('ERROR : anatomical count is not correct, count != len chan_list')

    ROI_to_include = [ROI_i for ROI_i in ROI_list if ROI_dict[ROI_i] > 0]
    lobe_to_include = [Lobe_i for Lobe_i in lobe_list if lobe_dict[Lobe_i] > 0]

    return ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots









########################
######## STATS ########
########################




def get_tf_stats(tf, nchan, pixel_based_distrib, nfrex):

    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(nfrex):
        mask = np.logical_or(tf_thresh[wavelet_i, :] >= pixel_based_distrib[nchan, wavelet_i, 0], tf_thresh[wavelet_i, :] <= pixel_based_distrib[nchan, wavelet_i, 1])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    return tf_thresh





#ROI_to_process = ROI_to_include[1]
def get_STATS_for_ROI(ROI_to_process, cond, monopol):

    #### load srate
    srate = get_params(sujet_list[0], monopol)['srate']

    #### load anat
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(monopol)

    #### identify stretch point
    stretch_point = stretch_point_TF

    #### identify if need to be proccessed
    if (ROI_to_process in ROI_to_include) == False:
        return

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
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict_i in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict_i.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict_i.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### initiate len recorded
    len_recorded = []
    
    #### compute TF
    #plot_to_process_i = plot_to_process[0]    
    for plot_to_process_num, plot_to_process_i in enumerate(plot_to_process):

        # print_advancement(plot_to_process_num, len(plot_to_process), steps=[25, 50, 75])
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))

        #### load subject params
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_tmp, monopol)

        if sujet_tmp[:3] != 'pat' and monopol:
            chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

        #### identify plot name
        plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### add length recorded
        band_prep = 'lf'
        len_recorded.append(load_data_sujet(sujet_tmp, band_prep, cond, 0, monopol)[plot_tmp_i,:].shape[0]/srate/60)

        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))

        #### identify trial number
        band, freq = list(dict_freq_band.items())[0]

        if monopol:
            n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and i.find('STATS') != -1 and i.find('bi') == -1])
        else:
            n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and i.find('STATS') != -1 and i.find('bi') != -1])


        #### load TF and mean trial
        #band, freq = 'l_gamma', [50, 80]
        for band, freq in dict_freq_band.items():

            _, nfrex = get_wavelets(sujet_tmp, band_prep, freq, monopol)
    
            #trial_i = 0
            for trial_i in range(n_trials):
                
                if trial_i == 0:

                    if monopol:

                        tf = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')[plot_tmp_i, :, :]
                    
                    else:

                        tf = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}_bi.npy')[plot_tmp_i, :, :]

                else:

                    if monopol:
                        
                        tf += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')[plot_tmp_i, :, :]

                    else:
                        
                        tf += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}_bi.npy')[plot_tmp_i, :, :]

            tf /= n_trials

            #### thresh significant
            if monopol:
                pixel_based_distrib = np.load(f'{sujet_tmp}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy')
            else:
                pixel_based_distrib = np.load(f'{sujet_tmp}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy')
                
            tf_stats = get_tf_stats(rscore_mat(tf), plot_tmp_i, pixel_based_distrib, nfrex)

            dict_TF_for_ROI_to_process[band] = (dict_TF_for_ROI_to_process[band] + tf_stats)

    #### verif
    if debug:
        for band, freq in dict_freq_band.items():
            plt.pcolormesh(dict_TF_for_ROI_to_process[band])
            plt.show()

    # #### mean thresh significant
    # for band, freq in dict_freq_band.items():
    #     dict_TF_for_ROI_to_process[band] = (dict_TF_for_ROI_to_process[band] == len(plot_to_process)).astype(int)

    #### verif
    if debug:
        band = 'theta'
        plt.pcolormesh(dict_TF_for_ROI_to_process[band])
        plt.show()

    return dict_TF_for_ROI_to_process





#Lobe_to_process = lobe_to_include[4]
def get_STATS_for_Lobes(Lobe_to_process, cond, monopol):

    #### load srate
    srate = get_params(sujet_list[0], monopol)['srate']

    #### load anat
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(monopol)

    #### identify stretch point
    stretch_point = stretch_point_TF

    #### identify if need to be proccessed
    if (Lobe_to_process in lobe_to_include) == False:
        return

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
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict_i in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict_i.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict_i.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### initiate len recorded
    len_recorded = []
    
    #### compute TF
    #plot_to_process_i = plot_to_process[0]    
    for plot_to_process_num, plot_to_process_i in enumerate(plot_to_process):

        # print_advancement(plot_to_process_num, len(plot_to_process), steps=[25, 50, 75])
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))

        #### load subject params
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_tmp, monopol)

        if sujet_tmp[:3] != 'pat' and monopol:
            chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

        #### identify plot name
        plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### add length recorded
        band_prep = 'lf'
        len_recorded.append(load_data_sujet(sujet_tmp, band_prep, cond, 0, monopol)[plot_tmp_i,:].shape[0]/srate/60)

        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))

        #### identify trial number
        band, freq = list(dict_freq_band.items())[0]

        if monopol:
            n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and i.find('STATS') == -1 and i.find('bi') == -1])
        else:
            n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and i.find('STATS') == -1 and i.find('bi') != -1])


        #### load TF and mean trial
        #band, freq = 'l_gamma', [50, 80]
        for band, freq in dict_freq_band.items():

            _, nfrex = get_wavelets(sujet_tmp, band_prep, freq, monopol)
    
            #trial_i = 0
            for trial_i in range(n_trials):
                
                if trial_i == 0:

                    if monopol:

                        tf = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')[plot_tmp_i, :, :]
                    
                    else:

                        tf = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}_bi.npy')[plot_tmp_i, :, :]

                else:

                    if monopol:
                        
                        tf += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')[plot_tmp_i, :, :]

                    else:
                        
                        tf += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}_bi.npy')[plot_tmp_i, :, :]

            tf /= n_trials

            #### thresh significant
            if monopol:
                pixel_based_distrib = np.load(f'{sujet_tmp}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy')
            else:
                pixel_based_distrib = np.load(f'{sujet_tmp}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy')
                
            tf_stats = get_tf_stats(rscore_mat(tf), plot_tmp_i, pixel_based_distrib, nfrex)

            dict_TF_for_Lobe_to_process[band] = (dict_TF_for_Lobe_to_process[band] + tf_stats)

    #### verif
    if debug:
        for band, freq in dict_freq_band.items():
            plt.pcolormesh(dict_TF_for_Lobe_to_process[band])
            plt.show()

    #### mean thresh significant
    # for band, freq in dict_freq_band.items():
        # dict_TF_for_Lobe_to_process[band] = (dict_TF_for_Lobe_to_process[band] == len(plot_to_process)).astype(int)

    #### verif
    if debug:
        band = 'theta'
        plt.pcolormesh(dict_TF_for_Lobe_to_process[band])
        plt.show()

    return dict_TF_for_Lobe_to_process














########################################
######## COMPUTE TF FOR COND ######## 
########################################


def robust_zscore(data):
    
    _median = np.median(data) 
    MAD = np.median(np.abs(data-np.median(data)))
    data_zscore = (0.6745*(data-_median))/ MAD
        
    return data_zscore


#struct_name, cond, mat_type, anat_type = ROI_name, 'FR_CV', 'TF', 'ROI'
def open_TForITPC_data(struct_name, cond, mat_type, anat_type, monopol):
    
    #### open file
    os.chdir(os.path.join(path_precompute, 'allplot'))
    
    listdir = os.listdir()
    file_to_open = []

    if monopol:
        [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1 and file_i.find('bi') == -1]
    else:
        [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1 and file_i.find('bi') != -1]

    #### extract band names
    band_names = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]

    #### load matrix
    xr_TF = xr.open_dataarray(file_to_open[0])
    try:
        struct_xr = xr_TF.loc[struct_name, :, mat_type, :, :]
    except:
        print(f'{struct_name} {cond} not found')
        return 0, 0

    #### identify plot number
    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i, monopol)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### search for ROI & lobe that have been counted
    n_count = 0
    
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))

        if monopol:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        else:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')

        if anat_type == 'ROI':
            n_count += np.sum(plot_loca_df['localisation_corrected'] == struct_name)
        if anat_type == 'Lobes':
            n_count += np.sum(plot_loca_df['lobes_corrected'] == struct_name)

    return struct_xr, n_count







#ROI_name, mat_type = 'amygdala', 'TF'
def compute_for_one_ROI_allcond(ROI_name, mat_type, cond_to_compute, srate, monopol):

    print(ROI_name)

    #### params
    anat_type = 'ROI'

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    #### get stats
    STATS_for_ROI_to_process = {}

    for cond in cond_to_compute:

        if cond != 'FR_CV':

            STATS_for_ROI_to_process[cond] = get_STATS_for_ROI(ROI_name, cond, monopol)

    #### get data
    allcond_TF = {}
    allcond_count = {}
    #cond = 'FR_CV'
    for cond in cond_to_compute:

        allcond_TF[cond], allcond_count[cond] = open_TForITPC_data(ROI_name, cond, mat_type, anat_type, monopol)

    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'ROI'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'ROI'))

    #### plot
    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### extract band to plot
        freq_band = freq_band_dict[band_prep]

        #### initiate fig
        fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(cond_to_compute))

        fig.set_figheight(10)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        if band_prep == 'hf':
            keys_list_reversed = list(freq_band.keys())
            keys_list_reversed.reverse()
            freq_band_reversed = {}
            for key_i in keys_list_reversed:
                freq_band_reversed[key_i] = freq_band[key_i]
            freq_band = freq_band_reversed

        if monopol:
            plt.suptitle(ROI_name)
        else:
            plt.suptitle(f'{ROI_name} bi')

        #cond_i, cond = 0, 'FR_CV'
        for c, cond in enumerate(cond_to_compute):

            #### generate time vec
            time_vec = np.arange(stretch_point_TF)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for r, (band, freq) in enumerate(list(freq_band.items())) :

                TF_i = allcond_TF[cond].loc[band, :, :].data
                TF_count_i = allcond_count[cond]
                frex = np.linspace(freq[0], freq[1], TF_i.shape[0])
                
                ax = axs[r, c]
                if r == 0 :
                    ax.set_title(f' {cond} : {TF_count_i}')
                if c == 0:
                    ax.set_ylabel(band)
                    
                ax.pcolormesh(time_vec, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                if cond != 'FR_CV':
                    # ax.contour(time_vec, frex, STATS_for_ROI_to_process[band], levels=0, colors='g')
                    ax.contour(time_vec, frex, STATS_for_ROI_to_process[cond][band])

                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')

        # plt.show()
                    
        #### save
        if monopol:
            fig.savefig(f'{ROI_name}_allcond_{band_prep}.jpeg', dpi=150)
        else:
            fig.savefig(f'{ROI_name}_allcond_{band_prep}_bi.jpeg', dpi=150)
        
        plt.close('all')









#Lobe_name, mat_type = 'Temporal', 'TF'
def compute_for_one_Lobe_allcond(Lobe_name, mat_type, cond_to_compute, srate, monopol):

    print(Lobe_name)

    #### params
    anat_type = 'Lobes'

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    #### get stats
    STATS_for_Lobe_to_process = {}

    for cond in cond_to_compute:

        if cond != 'FR_CV':

            STATS_for_Lobe_to_process[cond] = get_STATS_for_Lobes(Lobe_name, cond, monopol)

    allcond_TF = {}
    allcond_count = {}
    #cond = 'FR_CV'
    for cond in cond_to_compute:

        allcond_TF[cond], allcond_count[cond] = open_TForITPC_data(Lobe_name, cond, mat_type, anat_type, monopol)

    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'Lobes'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'Lobes'))

    #### plot
    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### extract band to plot
        freq_band = freq_band_dict[band_prep]

        #### initiate fig
        fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(cond_to_compute))

        fig.set_figheight(10)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        if band_prep == 'hf':
            keys_list_reversed = list(freq_band.keys())
            keys_list_reversed.reverse()
            freq_band_reversed = {}
            for key_i in keys_list_reversed:
                freq_band_reversed[key_i] = freq_band[key_i]
            freq_band = freq_band_reversed

        if monopol:
            plt.suptitle(Lobe_name)
        else:
            plt.suptitle(f'{Lobe_name} bi')

        #cond_i, cond = 0, 'FR_CV'
        for c, cond in enumerate(cond_to_compute):

            #### generate time vec
            time_vec = np.arange(stretch_point_TF)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for r, (band, freq) in enumerate(list(freq_band.items())) :

                TF_i = allcond_TF[cond].loc[band, :, :].data
                TF_count_i = allcond_count[cond]
                frex = np.linspace(freq[0], freq[1], TF_i.shape[0])
                
                ax = axs[r, c]
                if r == 0 :
                    ax.set_title(f' {cond} : {TF_count_i}')
                if c == 0:
                    ax.set_ylabel(band)
                    
                ax.pcolormesh(time_vec, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                if cond != 'FR_CV':
                    # ax.contour(time_vec, frex, STATS_for_Lobe_to_process[band], levels=0, colors='g')
                    ax.contour(time_vec, frex, STATS_for_Lobe_to_process[cond][band])

                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')

        # plt.show()
                    
        #### save
        if monopol:
            fig.savefig(f'{Lobe_name}_allcond_{band_prep}.jpeg', dpi=150)
        else:
            fig.savefig(f'{Lobe_name}_allcond_{band_prep}_bi.jpeg', dpi=150)

        plt.close('all')






################################
######## COMPILATION ########
################################

def compilation_slurm(anat_type, mat_type, monopol):

    print(f'#### {anat_type} {mat_type} ####')

    cond_to_compute = ['FR_CV', 'RD_CV', 'RD_SV', 'RD_FV']

    #### verify srate for all sujet
    if np.unique(np.array([get_params(sujet, monopol)['srate'] for sujet in sujet_list])).shape[0] == 1:
        srate = get_params(sujet_list[0], monopol)['srate']

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(monopol)

    if anat_type == 'ROI':

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_ROI_allcond)(ROI_i, mat_type, cond_to_compute, srate, monopol) for ROI_i in ROI_to_include)
        # for ROI_i in ROI_to_include:
        #     compute_for_one_ROI_allcond(ROI_i, mat_type, cond_to_compute, srate)

    if anat_type == 'Lobes':

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_Lobe_allcond)(Lobe_i, mat_type, cond_to_compute, srate, monopol) for Lobe_i in lobe_to_include)
        # for Lobe_i in lobe_to_include:
        #     compute_for_one_Lobe_allcond(Lobe_i, mat_type, cond_to_compute, srate)




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #monopol = False
    for monopol in [True, False]:

        #anat_type = 'ROI'
        for anat_type in ['ROI', 'Lobes']:
        
            #mat_type = 'TF'
            for mat_type in ['TF', 'ITPC']:
                
                # compilation_slurm(anat_type, mat_type, monopol)
                execute_function_in_slurm_bash('n16_res_allplot_TF', 'compilation_slurm', [anat_type, mat_type, monopol])




