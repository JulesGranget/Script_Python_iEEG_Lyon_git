

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.plts import plot_cyclepoints_array

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False







########################################
######## COMPUTE RESPI FEATURES ########
########################################


#respi, respi_clean = respi_allcond_detection[cond][session_i], respi_allcond[cond][session_i]
def exclude_bad_cycles(respi, respi_clean, cycles, srate, exclusion_metrics='med', outlier_coeff_removing=6, metric_coeff_exclusion=3, respi_scale=[0.1, 0.35]):

    cycles_init = cycles.copy()

    #### check respi
    if debug:

        plt.plot(zscore(respi), label='detection')
        plt.plot(zscore(respi_clean), label='clean')
        plt.legend()
        plt.show()

        plt.plot(rscore(respi), label='detection')
        plt.plot(rscore(respi_clean), label='clean')
        plt.legend()
        plt.show()

    ######## REMOVE OUTLIERS ########

    #### remove outlier regarding inspi/expi diff
    ratio_inspi_expi = np.log(cycles_init[:,0] / cycles_init[:,1])

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(ratio_inspi_expi)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = ratio_inspi_expi.mean(), ratio_inspi_expi.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(ratio_inspi_expi)
        mod = physio.get_empirical_mode(ratio_inspi_expi)
        metric_center, metric_dispersion = mod, med

    ratio_val_to_exclude = ratio_inspi_expi[(ratio_inspi_expi < (metric_center - metric_dispersion*outlier_coeff_removing)) | (ratio_inspi_expi > (metric_center + metric_dispersion*outlier_coeff_removing))]
    ratio_i_excluded = [i for i, val in enumerate(ratio_inspi_expi) if val in ratio_val_to_exclude]

    if debug:

        inspi_starts_init = cycles_init[:,0]
        fig, ax = plt.subplots()
        plt.title('ratio')
        ax.plot(respi)
        ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
        ax.scatter(inspi_starts_init[ratio_i_excluded], respi[inspi_starts_init[ratio_i_excluded]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts_init, ratio_inspi_expi, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
        plt.legend()
        plt.show()

    #### remove regarding integral
    inspi_starts = cycles_init[:,0]
    sums = np.zeros(inspi_starts.shape[0])

    for cycle_i in range(inspi_starts.shape[0]):

        if cycle_i == inspi_starts.shape[0]-1:
            start_i, stop_i = inspi_starts[cycle_i], respi.shape[0]

        else:
            start_i, stop_i = inspi_starts[cycle_i], inspi_starts[cycle_i+1] 

        sums[cycle_i] = np.sum(np.abs(respi[start_i:stop_i] - respi[start_i:stop_i].mean()))

    cycle_metrics = np.log(sums)

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(cycle_metrics)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = cycle_metrics.mean(), cycle_metrics.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(cycle_metrics)
        mod = physio.get_empirical_mode(cycle_metrics)
        metric_center, metric_dispersion = mod, med

    sum_excluded_val = cycle_metrics[(cycle_metrics < (metric_center - metric_dispersion*outlier_coeff_removing)) | (cycle_metrics > (metric_center + metric_dispersion*outlier_coeff_removing))]
    sum_excluded_i = [i for i, val in enumerate(cycle_metrics) if val in sum_excluded_val]

    if sujet in sujet_info_process['cut_population']:
        metric_center, metric_dispersion = cycle_metrics.mean(), cycle_metrics.std()
        sum_excluded_val = cycle_metrics[cycle_metrics < metric_center]
        sum_excluded_i = [i for i, val in enumerate(cycle_metrics) if val in sum_excluded_val]

    if debug:

        inspi_starts_init = cycles_init[:,0]
        fig, ax = plt.subplots()
        plt.title('sum')
        ax.plot(respi)
        ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
        ax.scatter(inspi_starts_init[sum_excluded_i], respi[inspi_starts_init[sum_excluded_i]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts_init, cycle_metrics, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
        plt.legend()
        plt.show()

    #### remove regarding freq
    freq = np.log(np.diff(cycles_init[:,0]))
    freq = np.concatenate(( freq, np.array([np.median(freq)]) ))

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(freq)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = freq.mean(), freq.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(freq)
        mod = physio.get_empirical_mode(freq)
        metric_center, metric_dispersion = mod, med

    freq_excluded_val = freq[(freq < (metric_center - metric_dispersion*outlier_coeff_removing)) | (freq > (metric_center + metric_dispersion*outlier_coeff_removing))]
    freq_excluded_i = [i for i, val in enumerate(freq) if val in freq_excluded_val]

    if debug:

        inspi_starts_init = cycles_init[:,0]
        fig, ax = plt.subplots()
        plt.title('freq_log')
        ax.plot(respi)
        ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
        ax.scatter(inspi_starts_init[freq_excluded_i], respi[inspi_starts_init[freq_excluded_i]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts_init, freq, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
        plt.legend()
        plt.show()
        
    #### remove all bad cycles
    remove_i = np.unique( np.concatenate((np.array(ratio_i_excluded), np.array(sum_excluded_i), np.array(freq_excluded_i))) ).astype(int)
    include_i = [i for i in range(cycles_init.shape[0]) if i not in remove_i]
    cycles_clean = cycles_init[include_i]

    if debug:

        inspi_starts = cycles_clean[:,0]
        inspi_starts_init = cycles_init[:,0]
        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        ax.scatter(inspi_starts_init[remove_i], respi[inspi_starts_init[remove_i]], color='r')
        plt.show()




    ######## ESTABLISH CONTINUITY ########

    freq = np.diff(cycles_clean[:,0])

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(freq)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = freq.mean(), freq.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(freq)
        mod = physio.get_empirical_mode(freq)
        metric_center, metric_dispersion = mod, med

    n_cycles_add = []

    #cycle_i = 0
    for cycle_i in range(cycles_clean.shape[0]):

        if cycle_i == (cycles_clean.shape[0]-1):
            break

        freq_with_next_inspi = cycles_clean[cycle_i,-1] - cycles_clean[cycle_i,0]
        freq_with_next_cycle = cycles_clean[cycle_i+1,0] - cycles_clean[cycle_i,0] 

        if freq_with_next_inspi == freq_with_next_cycle:
            continue

        else:

            if freq_with_next_cycle < metric_center + metric_coeff_exclusion*metric_dispersion and freq_with_next_cycle > metric_center - metric_coeff_exclusion*metric_dispersion:

                cycles_clean[cycle_i,-1] = cycles_clean[cycle_i+1,0]

            else:

                cycles_add = np.array([cycles_clean[cycle_i,-1], cycles_clean[cycle_i,-1]+((cycles_clean[cycle_i+1,0] - cycles_clean[cycle_i,-1])/2), cycles_clean[cycle_i+1,0]]).astype(int)
                n_cycles_add.append([cycle_i+1, cycles_add])

    #### add
    cycles_clean_final = cycles_clean.copy()
    mask_cycle = np.array([True]*(cycles_clean.shape[0] + len(n_cycles_add)))

    add_i = 0
    for cycle_add_i, cycle_add_val in n_cycles_add:
        mask_cycle[cycle_add_i] = False
        cycles_clean_final = np.insert(cycles_clean_final, cycle_add_i+add_i, cycle_add_val, axis=0)
        add_i += 1

    if debug:

        inspi_starts = cycles_clean_final[:,0]
        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts[mask_cycle], respi[inspi_starts[mask_cycle]], color='g', label='raw')
        ax.scatter(inspi_starts[~mask_cycle], respi[inspi_starts[~mask_cycle]], color='r', label='added')
        plt.legend()
        plt.show()




    ######## EXCLUDE CYCLES ########

    #### exclude regarding inspi/expi ratio
    ratio_inspi_expi = np.log(cycles_clean_final[:,0] / cycles_clean_final[:,1])

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(ratio_inspi_expi)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = ratio_inspi_expi.mean(), ratio_inspi_expi.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(ratio_inspi_expi)
        mod = physio.get_empirical_mode(ratio_inspi_expi)
        metric_center, metric_dispersion = mod, med

    ratio_val_to_exclude = ratio_inspi_expi[(ratio_inspi_expi < (metric_center - metric_dispersion*metric_coeff_exclusion)) | (ratio_inspi_expi > (metric_center + metric_dispersion*metric_coeff_exclusion))]
    ratio_i_excluded_clean = [i for i, val in enumerate(ratio_inspi_expi) if val in ratio_val_to_exclude]

    if debug:

        inspi_starts = cycles_clean_final[:,0]
        fig, ax = plt.subplots()
        plt.title('ratio')
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        ax.scatter(inspi_starts[ratio_i_excluded_clean], respi[inspi_starts[ratio_i_excluded_clean]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts, ratio_inspi_expi, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        plt.legend()
        plt.show()

    #### exclude regarding integral
    inspi_starts = cycles_clean_final[:,0]
    sums = np.zeros(inspi_starts.shape[0])

    for cycle_i in range(inspi_starts.shape[0]):

        if cycle_i == inspi_starts.shape[0]-1:
            start_i, stop_i = inspi_starts[cycle_i], respi.shape[0]

        else:
            start_i, stop_i = inspi_starts[cycle_i], inspi_starts[cycle_i+1] 

        sums[cycle_i] = np.sum(np.abs(respi[start_i:stop_i] - respi[start_i:stop_i].mean()))

    cycle_metrics = np.log(sums)

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(cycle_metrics)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = cycle_metrics.mean(), cycle_metrics.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(cycle_metrics)
        mod = physio.get_empirical_mode(cycle_metrics)
        metric_center, metric_dispersion = mod, med

    sum_excluded_val = cycle_metrics[(cycle_metrics < (metric_center - metric_dispersion*metric_coeff_exclusion)) | (cycle_metrics > (metric_center + metric_dispersion*metric_coeff_exclusion))]
    sum_excluded_i_clean = [i for i, val in enumerate(cycle_metrics) if val in sum_excluded_val]

    if debug:

        inspi_starts = cycles_clean_final[:,0]
        fig, ax = plt.subplots()
        plt.title('sum')
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        ax.scatter(inspi_starts[sum_excluded_i_clean], respi[inspi_starts[sum_excluded_i_clean]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts, cycle_metrics, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        plt.legend()
        plt.show()

    #### exclude regarding freq
    freq = np.log(np.diff(cycles_clean_final[:,0]))
    freq = np.concatenate(( freq, np.array([np.median(freq)]) ))

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(freq)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = freq.mean(), freq.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(freq)
        mod = physio.get_empirical_mode(freq)
        metric_center, metric_dispersion = mod, med

    freq_excluded_val = freq[(freq < (metric_center - metric_dispersion*metric_coeff_exclusion)) | (freq > (metric_center + metric_dispersion*metric_coeff_exclusion))]
    freq_excluded_i_clean = [i for i, val in enumerate(freq) if val in freq_excluded_val]

    if debug:

        inspi_starts = cycles_clean_final[:,0]
        fig, ax = plt.subplots()
        plt.title('freq_log')
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        ax.scatter(inspi_starts[freq_excluded_i_clean], respi[inspi_starts[freq_excluded_i_clean]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts, freq, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        plt.legend()
        plt.show()

    #### exclude regarding scale
    freq_Hz = 1/(np.diff(cycles_clean_final[:,0])/srate)
    freq_Hz_plot = np.append(freq_Hz, np.median(freq_Hz))

    if sujet in sujet_info_process['wide_scale_respi_freq']:
        freq_scale_excluded_i_clean = [i for i, val in enumerate(freq_Hz) if val > 0.55 or val < 0.1]
    else:
        freq_scale_excluded_i_clean = [i for i, val in enumerate(freq_Hz) if val > respi_scale[-1] or val < respi_scale[0]]

    if debug:

        inspi_starts = cycles_clean_final[:,0]
        fig, ax = plt.subplots()
        plt.title('scale')
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        ax.scatter(inspi_starts[freq_scale_excluded_i_clean], respi[inspi_starts[freq_scale_excluded_i_clean]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts, freq_Hz_plot, color='r', label=exclusion_metrics)

        if sujet in sujet_info_process['bad_respi']:
            ax2.axhline(0.1, color='r', linestyle='--')
            ax2.axhline(0.55, color='r', linestyle='--')
        else:
            ax2.axhline(respi_scale[-1], color='r', linestyle='--')
            ax2.axhline(respi_scale[0], color='r', linestyle='--')
        
        plt.legend()
        plt.show()

    #### exclude all bad cycles
    exclude_i_final = []
    for exclude_i in ratio_i_excluded_clean:
        if exclude_i in sum_excluded_i_clean and exclude_i in freq_excluded_i_clean:
            exclude_i_final.append(exclude_i)

    mask_cycle[exclude_i_final] = False
    mask_cycle[freq_scale_excluded_i_clean] = False

    if debug:

        inspi_starts = cycles_clean_final[:,0]
        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        plt.show()




    ######## GENERATE FIGURE ########

    respi = respi_clean

    time_vec = np.arange(respi.shape[0])/srate
    
    inspi_starts_init = cycles_init[:,0]
    fig_respi_exclusion, ax = plt.subplots(figsize=(18, 10))
    ax.plot(time_vec, respi)
    ax.scatter(inspi_starts_init/srate, respi[inspi_starts_init], color='g', label='inspi_selected')
    ax.scatter(cycles_init[:-1,1]/srate, respi[cycles_init[:-1,1]], color='c', label='expi_selected', marker='s')
    ax.scatter(inspi_starts_init[remove_i]/srate, respi[inspi_starts_init[remove_i]], color='k', label='removed_inspi', marker='+', s=200)
    plt.legend()
    # plt.show()
    plt.close()

    #### fig final
    fig_final, ax = plt.subplots(figsize=(18, 10))
    ax.plot(time_vec, respi)
    ax.scatter(cycles_clean_final[:,0]/srate, respi[cycles_clean_final[:,0]], color='g', label='inspi_selected')
    ax.scatter(cycles_clean_final[:,1]/srate, respi[cycles_clean_final[:,1]], color='c', label='expi_selected', marker='s')
    ax.scatter(cycles_clean_final[:,0][~mask_cycle]/srate, respi[cycles_clean_final[:,0][~mask_cycle]], color='r', label='inspi_excluded')
    ax.scatter(cycles_clean_final[:,1][~mask_cycle]/srate, respi[cycles_clean_final[:,1][~mask_cycle]], color='r', label='expi_excluded', marker='s')
    plt.legend()
    # plt.show()
    plt.close()

    return cycles_clean_final, mask_cycle, fig_respi_exclusion, fig_final










############################
######## LOAD DATA ########
############################



def load_respi_allcond_data(sujet, cycle_detection_params, sujet_info_process):

    #### load data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw_allcond = {}

    for cond in conditions:

        raw_allcond[cond] = {}

        for session_i in range(session_count[cond]):

            load_name = f'{sujet}_{cond}_{session_i+1}_wb.fif'

            load_data = mne.io.read_raw_fif(load_name, preload=True)
            load_data = load_data.pick_channels(['nasal']).get_data().reshape(-1)

            raw_allcond[cond][session_i] = load_data

    #### preproc respi
    respi_allcond = {}

    for cond in conditions:

        respi_allcond[cond] = {}

        for session_i in range(session_count[cond]):

            resp_clean = physio.preprocess(raw_allcond[cond][session_i], srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
            resp_clean_smooth = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=40.0)

            respi_allcond[cond][session_i] = resp_clean_smooth

    respi_allcond_detection = {}

    for cond in conditions:

        respi_allcond_detection[cond] = {}

        for session_i in range(session_count[cond]):

            resp_clean = physio.preprocess(raw_allcond[cond][session_i], srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
            
            if sujet[:3] == 'pat':
                resp_clean_smooth = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=120.0)
            else:
                resp_clean_smooth = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=40.0)

            respi_allcond_detection[cond][session_i] = resp_clean_smooth

            if debug:
                plt.plot(zscore(resp_clean_smooth), label='raw')
                plt.plot(zscore(resp_clean_smooth**3), label='cube')
                plt.legend()
                plt.show()

    #### detect
    respfeatures_allcond = {}

    #cond = 'RD_SV'
    for cond in conditions:

        respfeatures_allcond[cond] = {}

        #session_i = 0
        for session_i in range(session_count[cond]):

            # cycles = physio.detect_respiration_cycles(respi_allcond[cond][session_i], srate, baseline_mode='median',inspration_ajust_on_derivative=True)
            cycles = physio.detect_respiration_cycles(respi_allcond_detection[cond][session_i], srate, baseline_mode='median',inspration_ajust_on_derivative=True)

            # cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond[cond][session_i], cycles, srate, 
            #         exclusion_metrics=cycle_detection_params['exclusion_metrics'], outlier_coeff_removing=cycle_detection_params['outlier_coeff_removing'][cond], 
            #         metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], respi_scale=cycle_detection_params['respi_scale'][cond])
            cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond_detection[cond][session_i], respi_allcond[cond][session_i], cycles, srate, 
                    exclusion_metrics=cycle_detection_params['exclusion_metrics'], outlier_coeff_removing=cycle_detection_params['outlier_coeff_removing'][cond], 
                    metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], respi_scale=cycle_detection_params['respi_scale'][cond])
            
            #### get resp_features
            resp_features_i = physio.compute_respiration_cycle_features(respi_allcond[cond][session_i], srate, cycles, baseline=None)
    
            select_vec = np.ones((resp_features_i.index.shape[0]), dtype='int')
            select_vec[~cycles_mask_keep] = 0
            resp_features_i.insert(resp_features_i.columns.shape[0], 'select', select_vec)
            
            respfeatures_allcond[cond][session_i] = [resp_features_i, fig_respi_exclusion, fig_final]


    return raw_allcond, respi_allcond, respfeatures_allcond










########################################
######## EDIT CYCLES SELECTED ########
########################################


#respi_allcond = respi_allcond_bybycle
def edit_df_for_sretch_cycles_deleted(respi_allcond, respfeatures_allcond):

    for cond in conditions:
        
        for session_i in range(session_count[cond]):

            #### stretch
            cycles = respfeatures_allcond[cond][session_i][0][['inspi_index', 'expi_index']].values/srate
            times = np.arange(respi_allcond[cond][session_i].shape[0])/srate
            clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
                    respi_allcond[cond][session_i], times, cycles, nb_point_by_cycle=stretch_point_TF, inspi_ratio=ratio_stretch_TF)

            if debug:
                plt.plot(data_stretch_linear)
                plt.show()

            i_to_update = respfeatures_allcond[cond][session_i][0].index.values[~np.isin(respfeatures_allcond[cond][session_i][0].index.values, cycles)]
            for i_to_update_i in i_to_update:
                
                if i_to_update_i == respfeatures_allcond[cond][session_i][0].shape[0] - 1:
                    continue

                else:
                    respfeatures_allcond[cond][session_i][0]['select'][i_to_update_i] = 0

    return respfeatures_allcond



def export_cycle_count(sujet, respfeatures_allcond):

    #### generate df
    df_count_cycle = pd.DataFrame(columns={'sujet' : [], 'cond' : [], 'odor' : [], 'count' : []})

    for cond in conditions:
        
        for session_i in range(session_count[cond]):

            data_i = {'sujet' : [sujet], 'cond' : [cond], 'session' : [session_i+1], 'count' : [int(np.sum(respfeatures_allcond[cond][session_i][0]['select'].values))]}
            df_i = pd.DataFrame(data_i, columns=data_i.keys())
            df_count_cycle = pd.concat([df_count_cycle, df_i])

    #### export
    os.chdir(os.path.join(path_results, sujet, 'RESPI'))
    df_count_cycle.to_excel(f'{sujet}_count_cycles.xlsx')










############################
######## EXECUTE ########
############################



if __name__ == '__main__':

    ############################
    ######## LOAD DATA ########
    ############################

    
    #### whole protocole
    # sujet = 'CHEe'
    # sujet = 'GOBc' 
    # sujet = 'MAZm' 
    # sujet = 'TREt' 
    # sujet = 'POTm'

    #### FR_CV only
    # sujet = 'KOFs'
    # sujet = 'MUGa'
    # sujet = 'BANc'
    # sujet = 'LEMl'

    # sujet = 'pat_02459_0912'
    # sujet = 'pat_02476_0929'
    # sujet = 'pat_02495_0949'

    # sujet = 'pat_03083_1527'
    # sujet = 'pat_03105_1551'
    # sujet = 'pat_03128_1591'
    # sujet = 'pat_03138_1601'
    # sujet = 'pat_03146_1608'
    # sujet = 'pat_03174_1634'


    for sujet in sujet_list_FR_CV:

        sujet_info_process = {'cut_population' : ['pat_02459_0912', 'pat_02476_0929', 'pat_03083_1527', 'pat_03128_1591', 'pat_03138_1601', 'pat_03174_1634'],
                              'wide_scale_respi_freq' : ['pat_03146_1608']}

        print(f"#### #### ####")
        print(f"#### {sujet} ####")
        print(f"#### #### ####")

        if sujet in sujet_list:

            conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

        else:

            conditions = ['FR_CV']

        #### load data        
        raw_allcond, respi_allcond, respfeatures_allcond = load_respi_allcond_data(sujet, cycle_detection_params, sujet_info_process)



        ########################################
        ######## VERIF RESPIFEATURES ########
        ########################################
        
        if debug == True :

            for cond in conditions:

                for session_i in range(session_count[cond]):

                    respfeatures_allcond[cond][session_i][2].suptitle(f'{cond} {session_i}')
                    respfeatures_allcond[cond][session_i][2].show()

                    respfeatures_allcond[cond][session_i][1].suptitle(f'{cond} {session_i}')
                    respfeatures_allcond[cond][session_i][1].show()

            cond = 'FR_CV'
            cond = 'RD_CV' 
            cond = 'RD_FV' 
            cond = 'RD_SV'
            
            session_i = 0
            session_i = 1
            session_i = 2

            respfeatures_allcond[cond][session_i][1].show()
            respfeatures_allcond[cond][session_i][2].show()

            respfeatures_allcond[cond][session_i][0]['select']




        ########################################
        ######## EDIT CYCLES SELECTED ########
        ########################################

        respfeatures_allcond = edit_df_for_sretch_cycles_deleted(respi_allcond, respfeatures_allcond)

        export_cycle_count(sujet, respfeatures_allcond)





        ################################
        ######## SAVE FIG ########
        ################################

        os.chdir(os.path.join(path_results, sujet, 'RESPI'))

        for cond in conditions:

            for session_i in range(session_count[cond]):

                respfeatures_allcond[cond][session_i][0].to_excel(f"{sujet}_{cond}_{session_i+1}_respfeatures.xlsx")
                respfeatures_allcond[cond][session_i][1].savefig(f"{sujet}_{cond}_{session_i+1}_fig0.jpeg")
                respfeatures_allcond[cond][session_i][2].savefig(f"{sujet}_{cond}_{session_i+1}_fig1.jpeg")


        
