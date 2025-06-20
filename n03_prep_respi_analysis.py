

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import physio
import seaborn as sns

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False







########################################
######## COMPUTE RESPI FEATURES ########
########################################


#respi, respi_clean = respi_allcond_detection[cond][session_i], respi_allcond[cond][session_i]
# def exclude_bad_cycles(respi, respi_clean, cycles, srate, exclusion_metrics='med', outlier_coeff_removing=6, metric_coeff_exclusion=3, respi_scale=[0.1, 0.35]):

#     cycles_init = cycles.copy()

#     #### check respi
#     if debug:

#         plt.plot(zscore(respi), label='detection')
#         plt.plot(zscore(respi_clean), label='clean')
#         plt.legend()
#         plt.show()

#         plt.plot(rscore(respi), label='detection')
#         plt.plot(rscore(respi_clean), label='clean')
#         plt.legend()
#         plt.show()

#     ######## REMOVE OUTLIERS ########

#     #### remove outlier regarding inspi/expi diff
#     ratio_inspi_expi = np.log(cycles_init[:,0] / cycles_init[:,1])

#     if exclusion_metrics == 'med':
#         med, mad = physio.compute_median_mad(ratio_inspi_expi)
#         metric_center, metric_dispersion = med, mad

#     if exclusion_metrics == 'mean':
#         metric_center, metric_dispersion = ratio_inspi_expi.mean(), ratio_inspi_expi.std()

#     if exclusion_metrics == 'mod':
#         med, mad = physio.compute_median_mad(ratio_inspi_expi)
#         mod = physio.get_empirical_mode(ratio_inspi_expi)
#         metric_center, metric_dispersion = mod, med

#     ratio_val_to_exclude = ratio_inspi_expi[(ratio_inspi_expi < (metric_center - metric_dispersion*outlier_coeff_removing)) | (ratio_inspi_expi > (metric_center + metric_dispersion*outlier_coeff_removing))]
#     ratio_i_excluded = [i for i, val in enumerate(ratio_inspi_expi) if val in ratio_val_to_exclude]

#     if debug:

#         inspi_starts_init = cycles_init[:,0]
#         fig, ax = plt.subplots()
#         plt.title('ratio')
#         ax.plot(respi)
#         ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
#         ax.scatter(inspi_starts_init[ratio_i_excluded], respi[inspi_starts_init[ratio_i_excluded]], color='k', marker='x', s=100)

#         ax2 = ax.twinx()
#         ax2.scatter(inspi_starts_init, ratio_inspi_expi, color='r', label=exclusion_metrics)
#         ax2.axhline(metric_center, color='r')
#         ax2.axhline(metric_center - metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
#         ax2.axhline(metric_center + metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
#         plt.legend()
#         plt.show()

#     #### remove regarding integral
#     inspi_starts = cycles_init[:,0]
#     sums = np.zeros(inspi_starts.shape[0])

#     for cycle_i in range(inspi_starts.shape[0]):

#         if cycle_i == inspi_starts.shape[0]-1:
#             start_i, stop_i = inspi_starts[cycle_i], respi.shape[0]

#         else:
#             start_i, stop_i = inspi_starts[cycle_i], inspi_starts[cycle_i+1] 

#         sums[cycle_i] = np.sum(np.abs(respi[start_i:stop_i] - respi[start_i:stop_i].mean()))

#     cycle_metrics = np.log(sums)

#     if exclusion_metrics == 'med':
#         med, mad = physio.compute_median_mad(cycle_metrics)
#         metric_center, metric_dispersion = med, mad

#     if exclusion_metrics == 'mean':
#         metric_center, metric_dispersion = cycle_metrics.mean(), cycle_metrics.std()

#     if exclusion_metrics == 'mod':
#         med, mad = physio.compute_median_mad(cycle_metrics)
#         mod = physio.get_empirical_mode(cycle_metrics)
#         metric_center, metric_dispersion = mod, med

#     sum_excluded_val = cycle_metrics[(cycle_metrics < (metric_center - metric_dispersion*outlier_coeff_removing)) | (cycle_metrics > (metric_center + metric_dispersion*outlier_coeff_removing))]
#     sum_excluded_i = [i for i, val in enumerate(cycle_metrics) if val in sum_excluded_val]

#     if sujet in sujet_info_process['cut_population']:
#         metric_center, metric_dispersion = cycle_metrics.mean(), cycle_metrics.std()
#         sum_excluded_val = cycle_metrics[cycle_metrics < metric_center]
#         sum_excluded_i = [i for i, val in enumerate(cycle_metrics) if val in sum_excluded_val]

#     if debug:

#         inspi_starts_init = cycles_init[:,0]
#         fig, ax = plt.subplots()
#         plt.title('sum')
#         ax.plot(respi)
#         ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
#         ax.scatter(inspi_starts_init[sum_excluded_i], respi[inspi_starts_init[sum_excluded_i]], color='k', marker='x', s=100)

#         ax2 = ax.twinx()
#         ax2.scatter(inspi_starts_init, cycle_metrics, color='r', label=exclusion_metrics)
#         ax2.axhline(metric_center, color='r')
#         ax2.axhline(metric_center - metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
#         ax2.axhline(metric_center + metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
#         plt.legend()
#         plt.show()

#     #### remove regarding freq
#     freq = np.log(np.diff(cycles_init[:,0]))
#     freq = np.concatenate(( freq, np.array([np.median(freq)]) ))

#     if exclusion_metrics == 'med':
#         med, mad = physio.compute_median_mad(freq)
#         metric_center, metric_dispersion = med, mad

#     if exclusion_metrics == 'mean':
#         metric_center, metric_dispersion = freq.mean(), freq.std()

#     if exclusion_metrics == 'mod':
#         med, mad = physio.compute_median_mad(freq)
#         mod = physio.get_empirical_mode(freq)
#         metric_center, metric_dispersion = mod, med

#     freq_excluded_val = freq[(freq < (metric_center - metric_dispersion*outlier_coeff_removing)) | (freq > (metric_center + metric_dispersion*outlier_coeff_removing))]
#     freq_excluded_i = [i for i, val in enumerate(freq) if val in freq_excluded_val]

#     if debug:

#         inspi_starts_init = cycles_init[:,0]
#         fig, ax = plt.subplots()
#         plt.title('freq_log')
#         ax.plot(respi)
#         ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
#         ax.scatter(inspi_starts_init[freq_excluded_i], respi[inspi_starts_init[freq_excluded_i]], color='k', marker='x', s=100)

#         ax2 = ax.twinx()
#         ax2.scatter(inspi_starts_init, freq, color='r', label=exclusion_metrics)
#         ax2.axhline(metric_center, color='r')
#         ax2.axhline(metric_center - metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
#         ax2.axhline(metric_center + metric_dispersion*outlier_coeff_removing, color='r', linestyle='--')
#         plt.legend()
#         plt.show()
        
#     #### remove all bad cycles
#     remove_i = np.unique( np.concatenate((np.array(ratio_i_excluded), np.array(sum_excluded_i), np.array(freq_excluded_i))) ).astype(int)
#     include_i = [i for i in range(cycles_init.shape[0]) if i not in remove_i]
#     cycles_clean = cycles_init[include_i]

#     if debug:

#         inspi_starts = cycles_clean[:,0]
#         inspi_starts_init = cycles_init[:,0]
#         fig, ax = plt.subplots()
#         ax.plot(respi)
#         ax.scatter(inspi_starts, respi[inspi_starts], color='g')
#         ax.scatter(inspi_starts_init[remove_i], respi[inspi_starts_init[remove_i]], color='r')
#         plt.show()




#     ######## ESTABLISH CONTINUITY ########

#     freq = np.diff(cycles_clean[:,0])

#     if exclusion_metrics == 'med':
#         med, mad = physio.compute_median_mad(freq)
#         metric_center, metric_dispersion = med, mad

#     if exclusion_metrics == 'mean':
#         metric_center, metric_dispersion = freq.mean(), freq.std()

#     if exclusion_metrics == 'mod':
#         med, mad = physio.compute_median_mad(freq)
#         mod = physio.get_empirical_mode(freq)
#         metric_center, metric_dispersion = mod, med

#     n_cycles_add = []

#     #cycle_i = 0
#     for cycle_i in range(cycles_clean.shape[0]):

#         if cycle_i == (cycles_clean.shape[0]-1):
#             break

#         freq_with_next_inspi = cycles_clean[cycle_i,-1] - cycles_clean[cycle_i,0]
#         freq_with_next_cycle = cycles_clean[cycle_i+1,0] - cycles_clean[cycle_i,0] 

#         if freq_with_next_inspi == freq_with_next_cycle:
#             continue

#         else:

#             if freq_with_next_cycle < metric_center + metric_coeff_exclusion*metric_dispersion and freq_with_next_cycle > metric_center - metric_coeff_exclusion*metric_dispersion:

#                 cycles_clean[cycle_i,-1] = cycles_clean[cycle_i+1,0]

#             else:

#                 cycles_add = np.array([cycles_clean[cycle_i,-1], cycles_clean[cycle_i,-1]+((cycles_clean[cycle_i+1,0] - cycles_clean[cycle_i,-1])/2), cycles_clean[cycle_i+1,0]]).astype(int)
#                 n_cycles_add.append([cycle_i+1, cycles_add])

#     #### add
#     cycles_clean_final = cycles_clean.copy()
#     mask_cycle = np.array([True]*(cycles_clean.shape[0] + len(n_cycles_add)))

#     add_i = 0
#     for cycle_add_i, cycle_add_val in n_cycles_add:
#         mask_cycle[cycle_add_i] = False
#         cycles_clean_final = np.insert(cycles_clean_final, cycle_add_i+add_i, cycle_add_val, axis=0)
#         add_i += 1

#     if debug:

#         inspi_starts = cycles_clean_final[:,0]
#         fig, ax = plt.subplots()
#         ax.plot(respi)
#         ax.scatter(inspi_starts[mask_cycle], respi[inspi_starts[mask_cycle]], color='g', label='raw')
#         ax.scatter(inspi_starts[~mask_cycle], respi[inspi_starts[~mask_cycle]], color='r', label='added')
#         plt.legend()
#         plt.show()




#     ######## EXCLUDE CYCLES ########

#     #### exclude regarding inspi/expi ratio
#     ratio_inspi_expi = np.log(cycles_clean_final[:,0] / cycles_clean_final[:,1])

#     if exclusion_metrics == 'med':
#         med, mad = physio.compute_median_mad(ratio_inspi_expi)
#         metric_center, metric_dispersion = med, mad

#     if exclusion_metrics == 'mean':
#         metric_center, metric_dispersion = ratio_inspi_expi.mean(), ratio_inspi_expi.std()

#     if exclusion_metrics == 'mod':
#         med, mad = physio.compute_median_mad(ratio_inspi_expi)
#         mod = physio.get_empirical_mode(ratio_inspi_expi)
#         metric_center, metric_dispersion = mod, med

#     ratio_val_to_exclude = ratio_inspi_expi[(ratio_inspi_expi < (metric_center - metric_dispersion*metric_coeff_exclusion)) | (ratio_inspi_expi > (metric_center + metric_dispersion*metric_coeff_exclusion))]
#     ratio_i_excluded_clean = [i for i, val in enumerate(ratio_inspi_expi) if val in ratio_val_to_exclude]

#     if debug:

#         inspi_starts = cycles_clean_final[:,0]
#         fig, ax = plt.subplots()
#         plt.title('ratio')
#         ax.plot(respi)
#         ax.scatter(inspi_starts, respi[inspi_starts], color='g')
#         ax.scatter(inspi_starts[ratio_i_excluded_clean], respi[inspi_starts[ratio_i_excluded_clean]], color='k', marker='x', s=100)

#         ax2 = ax.twinx()
#         ax2.scatter(inspi_starts, ratio_inspi_expi, color='r', label=exclusion_metrics)
#         ax2.axhline(metric_center, color='r')
#         ax2.axhline(metric_center - metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
#         ax2.axhline(metric_center + metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
#         plt.legend()
#         plt.show()

#     #### exclude regarding integral
#     inspi_starts = cycles_clean_final[:,0]
#     sums = np.zeros(inspi_starts.shape[0])

#     for cycle_i in range(inspi_starts.shape[0]):

#         if cycle_i == inspi_starts.shape[0]-1:
#             start_i, stop_i = inspi_starts[cycle_i], respi.shape[0]

#         else:
#             start_i, stop_i = inspi_starts[cycle_i], inspi_starts[cycle_i+1] 

#         sums[cycle_i] = np.sum(np.abs(respi[start_i:stop_i] - respi[start_i:stop_i].mean()))

#     cycle_metrics = np.log(sums)

#     if exclusion_metrics == 'med':
#         med, mad = physio.compute_median_mad(cycle_metrics)
#         metric_center, metric_dispersion = med, mad

#     if exclusion_metrics == 'mean':
#         metric_center, metric_dispersion = cycle_metrics.mean(), cycle_metrics.std()

#     if exclusion_metrics == 'mod':
#         med, mad = physio.compute_median_mad(cycle_metrics)
#         mod = physio.get_empirical_mode(cycle_metrics)
#         metric_center, metric_dispersion = mod, med

#     sum_excluded_val = cycle_metrics[(cycle_metrics < (metric_center - metric_dispersion*metric_coeff_exclusion)) | (cycle_metrics > (metric_center + metric_dispersion*metric_coeff_exclusion))]
#     sum_excluded_i_clean = [i for i, val in enumerate(cycle_metrics) if val in sum_excluded_val]

#     if debug:

#         inspi_starts = cycles_clean_final[:,0]
#         fig, ax = plt.subplots()
#         plt.title('sum')
#         ax.plot(respi)
#         ax.scatter(inspi_starts, respi[inspi_starts], color='g')
#         ax.scatter(inspi_starts[sum_excluded_i_clean], respi[inspi_starts[sum_excluded_i_clean]], color='k', marker='x', s=100)

#         ax2 = ax.twinx()
#         ax2.scatter(inspi_starts, cycle_metrics, color='r', label=exclusion_metrics)
#         ax2.axhline(metric_center, color='r')
#         ax2.axhline(metric_center - metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
#         ax2.axhline(metric_center + metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
#         plt.legend()
#         plt.show()

#     #### exclude regarding freq
#     freq = np.log(np.diff(cycles_clean_final[:,0]))
#     freq = np.concatenate(( freq, np.array([np.median(freq)]) ))

#     if exclusion_metrics == 'med':
#         med, mad = physio.compute_median_mad(freq)
#         metric_center, metric_dispersion = med, mad

#     if exclusion_metrics == 'mean':
#         metric_center, metric_dispersion = freq.mean(), freq.std()

#     if exclusion_metrics == 'mod':
#         med, mad = physio.compute_median_mad(freq)
#         mod = physio.get_empirical_mode(freq)
#         metric_center, metric_dispersion = mod, med

#     freq_excluded_val = freq[(freq < (metric_center - metric_dispersion*metric_coeff_exclusion)) | (freq > (metric_center + metric_dispersion*metric_coeff_exclusion))]
#     freq_excluded_i_clean = [i for i, val in enumerate(freq) if val in freq_excluded_val]

#     if debug:

#         inspi_starts = cycles_clean_final[:,0]
#         fig, ax = plt.subplots()
#         plt.title('freq_log')
#         ax.plot(respi)
#         ax.scatter(inspi_starts, respi[inspi_starts], color='g')
#         ax.scatter(inspi_starts[freq_excluded_i_clean], respi[inspi_starts[freq_excluded_i_clean]], color='k', marker='x', s=100)

#         ax2 = ax.twinx()
#         ax2.scatter(inspi_starts, freq, color='r', label=exclusion_metrics)
#         ax2.axhline(metric_center, color='r')
#         ax2.axhline(metric_center - metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
#         ax2.axhline(metric_center + metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
#         plt.legend()
#         plt.show()

#     #### exclude regarding scale
#     freq_Hz = 1/(np.diff(cycles_clean_final[:,0])/srate)
#     freq_Hz_plot = np.append(freq_Hz, np.median(freq_Hz))

#     if sujet in sujet_info_process['wide_scale_respi_freq']:
#         freq_scale_excluded_i_clean = [i for i, val in enumerate(freq_Hz) if val > 0.55 or val < 0.1]
#     else:
#         freq_scale_excluded_i_clean = [i for i, val in enumerate(freq_Hz) if val > respi_scale[-1] or val < respi_scale[0]]

#     if debug:

#         inspi_starts = cycles_clean_final[:,0]
#         fig, ax = plt.subplots()
#         plt.title('scale')
#         ax.plot(respi)
#         ax.scatter(inspi_starts, respi[inspi_starts], color='g')
#         ax.scatter(inspi_starts[freq_scale_excluded_i_clean], respi[inspi_starts[freq_scale_excluded_i_clean]], color='k', marker='x', s=100)

#         ax2 = ax.twinx()
#         ax2.scatter(inspi_starts, freq_Hz_plot, color='r', label=exclusion_metrics)

#         if sujet in sujet_info_process['bad_respi']:
#             ax2.axhline(0.1, color='r', linestyle='--')
#             ax2.axhline(0.55, color='r', linestyle='--')
#         else:
#             ax2.axhline(respi_scale[-1], color='r', linestyle='--')
#             ax2.axhline(respi_scale[0], color='r', linestyle='--')
        
#         plt.legend()
#         plt.show()

#     #### exclude all bad cycles
#     exclude_i_final = []
#     for exclude_i in ratio_i_excluded_clean:
#         if exclude_i in sum_excluded_i_clean and exclude_i in freq_excluded_i_clean:
#             exclude_i_final.append(exclude_i)

#     mask_cycle[exclude_i_final] = False
#     mask_cycle[freq_scale_excluded_i_clean] = False

#     if debug:

#         inspi_starts = cycles_clean_final[:,0]
#         fig, ax = plt.subplots()
#         ax.plot(respi)
#         ax.scatter(inspi_starts, respi[inspi_starts], color='g')
#         plt.show()




#     ######## GENERATE FIGURE ########

#     respi = respi_clean

#     time_vec = np.arange(respi.shape[0])/srate
    
#     inspi_starts_init = cycles_init[:,0]
#     fig_respi_exclusion, ax = plt.subplots(figsize=(18, 10))
#     ax.plot(time_vec, respi)
#     ax.scatter(inspi_starts_init/srate, respi[inspi_starts_init], color='g', label='inspi_selected')
#     ax.scatter(cycles_init[:-1,1]/srate, respi[cycles_init[:-1,1]], color='c', label='expi_selected', marker='s')
#     ax.scatter(inspi_starts_init[remove_i]/srate, respi[inspi_starts_init[remove_i]], color='k', label='removed_inspi', marker='+', s=200)
#     plt.legend()
#     # plt.show()
#     plt.close()

#     #### fig final
#     fig_final, ax = plt.subplots(figsize=(18, 10))
#     ax.plot(time_vec, respi)
#     ax.scatter(cycles_clean_final[:,0]/srate, respi[cycles_clean_final[:,0]], color='g', label='inspi_selected')
#     ax.scatter(cycles_clean_final[:,1]/srate, respi[cycles_clean_final[:,1]], color='c', label='expi_selected', marker='s')
#     ax.scatter(cycles_clean_final[:,0][~mask_cycle]/srate, respi[cycles_clean_final[:,0][~mask_cycle]], color='r', label='inspi_excluded')
#     ax.scatter(cycles_clean_final[:,1][~mask_cycle]/srate, respi[cycles_clean_final[:,1][~mask_cycle]], color='r', label='expi_excluded', marker='s')
#     plt.legend()
#     # plt.show()
#     plt.close()

#     return cycles_clean_final, mask_cycle, fig_respi_exclusion, fig_final










############################
######## LOAD DATA ########
############################



def load_respi_allcond_data(sujet):

    #### load data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw_allcond = {}

    for cond in conditions:

        raw_allcond[cond] = {}

        for session_i in range(session_count[cond]):

            load_name = f'{sujet}_{cond}_{session_i+1}_wb.fif'

            load_data = mne.io.read_raw_fif(load_name, preload=True)
            if sujet in ['LAVs']:
                load_data = load_data.pick_channels(['ventral']).get_data().reshape(-1)
            else:
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
            cycles = physio.detect_respiration_cycles(respi_allcond_detection[cond][session_i], srate, baseline_mode='median')

            # cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond_detection[cond][session_i], respi_allcond[cond][session_i], cycles, srate, 
            #         exclusion_metrics=cycle_detection_params['exclusion_metrics'], outlier_coeff_removing=cycle_detection_params['outlier_coeff_removing'][cond], 
            #         metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], respi_scale=cycle_detection_params['respi_scale'][cond])
            
            #### get resp_features
            resp_features_i = physio.compute_respiration_cycle_features(respi_allcond[cond][session_i], srate, cycles, baseline=None)
    
            select_vec = np.ones((resp_features_i.index.shape[0]), dtype='int')
            # select_vec[~cycles_mask_keep] = 0
            resp_features_i.insert(resp_features_i.columns.shape[0], 'select', select_vec)

            respi = respi_allcond[cond][session_i]

            time_vec = np.arange(respi.shape[0])/srate

            #### fig final
            fig_final, ax = plt.subplots(figsize=(18, 10))
            ax.plot(time_vec, respi)
            ax.scatter(cycles[:,0]/srate, respi[cycles[:,0]], color='g', label='inspi_selected')
            ax.scatter(cycles[:,1]/srate, respi[cycles[:,1]], color='c', label='expi_selected', marker='s')
            plt.legend()
            # plt.show()
            plt.close()
            
            # respfeatures_allcond[cond][session_i] = [resp_features_i, fig_respi_exclusion, fig_final]
            respfeatures_allcond[cond][session_i] = [resp_features_i, fig_final]


    return raw_allcond, respi_allcond, respfeatures_allcond






def export_cycle_count(sujet, respfeatures_allcond):

    #### generate df
    df_count_cycle = pd.DataFrame(columns={'sujet' : [], 'cond' : [], 'session' : [], 'count' : []})

    for cond in conditions:
        
        for session_i in range(session_count[cond]):

            data_i = {'sujet' : [sujet], 'cond' : [cond], 'session' : [session_i+1], 'count' : [int(np.sum(respfeatures_allcond[cond][session_i][0]['select'].values))]}
            df_i = pd.DataFrame(data_i, columns=data_i.keys())
            df_count_cycle = pd.concat([df_count_cycle, df_i])

    #### export
    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'count_cycle'))
    df_count_cycle.to_excel(f'{sujet}_count_cycles.xlsx')






####################################
######## PLOT MEAN RESPI ########
####################################

def plot_mean_respi(sujet, conditions):

    time_point_respi = srate*15
    time_vec = np.arange(time_point_respi)/srate
    colors_respi = {'FR_CV' : 'tab:blue', 'RD_CV' : 'tab:orange', 'RD_FV' : 'tab:red', 'RD_SV' : 'tab:green'}
    colors_respi_sem = {'FR_CV' : 'tab:blue', 'RD_CV' : 'tab:orange', 'RD_FV' : 'tab:red', 'RD_SV' : 'tab:green'}

    respi_allcond = load_respi_allcond(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
    sem_allcond = {}
    lim = {'min' : np.array([]), 'max' : np.array([])} 

    #### load
    #cond = 'VS'
    for cond in conditions:

        for session_i in range(session_count[cond]):

            _respi_cycles = np.zeros((time_point_respi))

            _respi = respi_allcond[cond][session_i]
            _respfeature = respfeatures_allcond[cond][session_i]

            for cycle_i, cycle_idx in enumerate(_respfeature['expi_index']):

                pre_chunk, post_chunk = cycle_idx - time_point_respi/2, cycle_idx + time_point_respi/2
                if pre_chunk < 0 or post_chunk > _respi.size:
                    continue
                _cycle_chunk = _respi[int(pre_chunk):int(post_chunk)] 
                
                if cycle_i == 0 and session_i == 0:
                    _respi_cycles = _cycle_chunk
                else:
                    _respi_cycles = np.vstack((_respi_cycles, _cycle_chunk))

        respi_allcond[cond] = _respi_cycles.mean(axis=0)
        sem_allcond[cond] = _respi_cycles.std(axis=0)/np.sqrt(_respi_cycles.shape[0])
        lim['min'], lim['max'] = np.append(lim['min'], respi_allcond[cond].min()-sem_allcond[cond]), np.append(lim['max'], respi_allcond[cond].max()+sem_allcond[cond])

    #### plot
    fig, ax = plt.subplots()

    #cond = 'VS'
    for cond in conditions:

        ax.plot(time_vec, respi_allcond[cond], color=colors_respi[cond], label=cond)
        ax.fill_between(time_vec, respi_allcond[cond]+sem_allcond[cond], respi_allcond[cond]-sem_allcond[cond], alpha=0.25, color=colors_respi_sem[cond])

    ax.vlines(time_point_respi/2/srate, ymin=lim['min'].min(), ymax=lim['max'].max(), color='r')
    plt.ylim(lim['min'].min(), lim['max'].max())
    plt.title(sujet)
    plt.legend()
    # plt.show()

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'plot'))
    plt.savefig(f"{sujet}_respi_mean.png")

    plt.close('all')






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

    # sujet = 'VERj'
    # sujet = 'DUCa'
    # sujet = 'CARv'
    # sujet = 'BOUt'
    # sujet = 'FLAb'

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

        print(f"#### #### ####")
        print(f"#### {sujet} ####")
        print(f"#### #### ####")

        if sujet in sujet_list:

            conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

        else:

            conditions = ['FR_CV']

        #### load data        
        raw_allcond, respi_allcond, respfeatures_allcond = load_respi_allcond_data(sujet)



        ########################################
        ######## VERIF RESPIFEATURES ########
        ########################################
        
        if debug == True :

            for cond in conditions:

                for session_i in range(session_count[cond]):

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

            respfeatures_corrected = respfeatures_allcond[cond][session_i][0]
            respi = respi_allcond[cond][session_i]
            cycle_corrected = respfeatures_corrected['select'].values.astype('bool')

            cycle_corrected[:27] = False
            chunk_pre, chunk_post = 27+20+14+14+1, 27+20+14+14+14+3
            cycle_corrected[chunk_pre:chunk_post] = False

            fig_final, ax = plt.subplots(figsize=(18, 10))
            ax.plot(np.arange(respi_allcond[cond][session_i].size), respi)
            ax.scatter(respfeatures_corrected['inspi_index'].values[cycle_corrected], respi[respfeatures_corrected['inspi_index'].values[cycle_corrected]], color='g', label='inspi_selected')
            ax.scatter(respfeatures_corrected['expi_index'].values[cycle_corrected], respi[respfeatures_corrected['expi_index'].values[cycle_corrected]], color='c', label='expi_selected', marker='s')
            plt.legend()
            plt.show()

            respfeatures_corrected['select'] = cycle_corrected * 1
            
            # respfeatures_allcond[cond][session_i] = [resp_features_i, fig_respi_exclusion, fig_final]
            respfeatures_allcond[cond][session_i] = [respfeatures_corrected, fig_final]




        ########################################
        ######## EDIT CYCLES SELECTED ########
        ########################################

        export_cycle_count(sujet, respfeatures_allcond)





        ################################
        ######## SAVE FIG ########
        ################################

        os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'respfeatures'))

        for cond in conditions:

            for session_i in range(session_count[cond]):

                respfeatures_allcond[cond][session_i][0].to_excel(f"{sujet}_{cond}_{session_i+1}_respfeatures.xlsx")
                respfeatures_allcond[cond][session_i][1].savefig(f"{sujet}_{cond}_{session_i+1}_fig0.jpeg")



        ################################
        ######## PLOT MEAN RESP ########
        ################################

        plot_mean_respi(sujet, conditions)



    ################################
    ######## ALL SUJETS ########
    ################################

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'count_cycle'))

    df_count_allsujet = pd.DataFrame()

    for sujet in sujet_list_FR_CV:

        df_count_allsujet = pd.concat([df_count_allsujet, pd.read_excel(f"{sujet}_count_cycles.xlsx")]).drop(columns=['Unnamed: 0', 'session'])

    df_count_allsujet = df_count_allsujet.groupby(['sujet', 'cond']).sum()

    #### FR_CV
    df_plot = df_count_allsujet.query(f"cond == 'FR_CV'")
    sns.barplot(df_plot, x='sujet', y='count')

    plt.title('FR_CV count cycles')
    plt.xticks(rotation=45)

    plt.tight_layout()

    #plt.show()
    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'allsujet'))
    plt.savefig('FR_CV_count_cycle.png')
    plt.close('all')

    #### ALLCOND
    df_plot = df_count_allsujet.query(f"sujet in {sujet_list}")
    sns.barplot(df_plot, x='sujet', y='count', hue='cond')

    plt.title('ALLCOND count cycles')
    plt.xticks(rotation=45)

    plt.tight_layout()

    #plt.show()
    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'allsujet'))
    plt.savefig('ALLPROTOCOL_count_cycle.png')
    plt.close('all')

    #### DFC
    df_plot = df_count_allsujet.query(f"sujet in {sujet_list_dfc_FR_CV} and cond == 'FR_CV'")
    sns.barplot(df_plot, x='sujet', y='count')

    plt.title('DFC count cycles')
    plt.xticks(rotation=45)

    plt.tight_layout()

    #plt.show()
    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'allsujet'))
    plt.savefig('DFC_count_cycle_FR_CV.png')
    plt.close('all')

    df_plot = df_count_allsujet.query(f"sujet in {sujet_list_dfc_allcond}")
    sns.barplot(df_plot, x='sujet', y='count', hue='cond')

    plt.title('DFC count cycles')
    plt.xticks(rotation=45)

    plt.tight_layout()

    #plt.show()
    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'allsujet'))
    plt.savefig('DFC_count_cycle_allcond.png')
    plt.close('all')




    
