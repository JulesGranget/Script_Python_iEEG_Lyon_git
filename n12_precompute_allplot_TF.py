


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *


debug = False










########################################
######## PREP ALLPLOT ANALYSIS ########
########################################



def get_ROI_Lobes_list_and_Plots(cond, monopol):

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

    #### filter only sujet with correct cond
    if cond == 'FR_CV':
        sujet_list_selected = sujet_list_FR_CV
    if cond != 'FR_CV':
        sujet_list_selected = sujet_list

    #### search for ROI & lobe that have been counted
    #sujet_i = sujet_list_selected[0]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))

        if monopol:
            file_path = f"{sujet_i}_chanlist_ieeg.txt"
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        else:
            file_path = f"{sujet_i}_chanlist_ieeg_bi.txt"
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        chan_list_ieeg = [line.strip() for line in lines]

        if monopol:
            chan_list_ieeg, _ = modify_name(chan_list_ieeg)

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        #nchan = chan_list_ieeg_csv[0]
        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df.query(f"plot == '{nchan}' and select == 1")['localisation_corrected'].values[0]
            lobe_tmp = plot_loca_df.query(f"plot == '{nchan}' and select == 1")['lobes_corrected'].values[0]
            
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
######## COMPILATION ########
################################



def compilation_allplot_analysis(cond, monopol):

    #### verify computation
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

    if monopol:
        if os.path.exists(f'allsujet_{cond}_ROI.nc'):
            print(f'ALREADY COMPUTED {cond}', flush=True)
            return
    else:
        if os.path.exists(f'allsujet_{cond}_ROI_bi.nc'):
            print(f'ALREADY COMPUTED {cond}', flush=True)
            return

    print(f'COMPUTE {cond} {monopol}', flush=True)

    #### load anat
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond, monopol)

    #### identify stretch point
    stretch_point = stretch_point_TF

    #### function to compute one ROI
    # for ROI_index in range(len(ROI_to_include)):

    def compute_roi(ROI_index):

        ROI_to_process = ROI_to_include[ROI_index]
        print(ROI_to_process, flush=True)

        tf_allplot = np.zeros((len(ROI_dict_plots[ROI_to_process]), nfrex, stretch_point), dtype=np.float32)

        for site_i, (sujet, site) in enumerate(ROI_dict_plots[ROI_to_process]):

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

            if sujet[:3] != 'pat':
                if monopol:
                    chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

            if monopol:
                tf = np.load(f'{sujet}_tf_conv_{cond}.npy')[chan_list_ieeg.index(site), :, :, :]
            else:
                tf = np.load(f'{sujet}_tf_conv_{cond}_bi.npy')[chan_list_ieeg.index(site), :, :, :]

            tf_allplot[site_i, :, :] = np.median(tf, axis=0)

        tf_median = np.median(tf_allplot, axis=0)

        if debug:
            vmin, vmax = np.percentile(tf_allplot[1, :, :].reshape(-1), tf_plot_percentile_scale), \
                        np.percentile(tf_allplot[1, :, :].reshape(-1), 100 - tf_plot_percentile_scale)
            plt.pcolormesh(tf_allplot[1, :, :], vmin=vmin, vmax=vmax)
            plt.show()

        return tf_median

    #### run in parallel
    res = joblib.Parallel(n_jobs=n_core)(joblib.delayed(compute_roi)(roi_idx) for roi_idx in range(len(ROI_to_include)))

    #### stack into final array
    ROI_data = np.stack(res, axis=0)  # shape: (n_ROIs, nfrex, stretch_point)

    print('SAVE', flush=True)

    #### extract & save
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))
    dict_xr = {'roi': ROI_to_include, 'nfrex': np.arange(0, nfrex), 'times': np.arange(0, stretch_point)}
    xr_export = xr.DataArray(ROI_data, coords=dict_xr.values(), dims=dict_xr.keys())

    if monopol:
        xr_export.to_netcdf(f'allsujet_{cond}_ROI.nc')
    else:
        xr_export.to_netcdf(f'allsujet_{cond}_ROI_bi.nc')




        






################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    list_params = []
    for monopol in [True, False]:
        for cond in ['FR_CV', 'RD_CV', 'RD_SV', 'RD_FV']:
            list_params.append([monopol, cond])

    execute_function_in_slurm_bash('n12_precompute_allplot_TF', 'compilation_allplot_analysis', list_params)
    #sync_folders__push_to_crnldata()


    for monopol in [True, False]:

        for cond in ['FR_CV', 'RD_CV', 'RD_SV', 'RD_FV']:

            compilation_allplot_analysis(cond, monopol)



 