
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
import xarray as xr
import joblib
import mne_connectivity
import copy

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False




################################
######## CLEAN DATA ########
################################




def clean_data(allband_data, allpairs):

    #### identify pairs to clean
    mask_keep = []

    for pair_i, pair in enumerate(allpairs):

        if pair.split('-')[0] in ROI_for_DFC_plot and pair.split('-')[-1] in ROI_for_DFC_plot:

            mask_keep.append(True)

        else:

            mask_keep.append(False)

    mask_keep = np.array(mask_keep)

    #### clean pairs
    allpairs = allpairs[mask_keep]

    if debug:

        allpairs[~mask_keep]

    #### clean data
    #band_i = 'beta'
    for band_i in allband_data:

        for cond in conditions:

            for phase in allband_data[band_i][cond].keys():

                allband_data[band_i][cond][phase] = allband_data[band_i][cond][phase][:, mask_keep, :]

    return allband_data, allpairs



    





########################################
######## ANALYSIS FUNCTIONS ########
########################################




def get_pair_unique_and_roi_unique(pairs):

    #### pairs unique
    pair_unique = []

    for pair_i in np.unique(pairs):
        if pair_i.split('-')[0] == pair_i.split('-')[-1]:
            continue
        if f"{pair_i.split('-')[-1]}-{pair_i.split('-')[0]}" in pair_unique:
            continue
        if pair_i not in pair_unique:
            pair_unique.append(pair_i)

    pair_unique = np.array(pair_unique)

    #### get roi in data
    roi_in_data = []

    for pair_i in np.unique(pairs):
        if pair_i.split('-')[0] not in roi_in_data:
            roi_in_data.append(pair_i.split('-')[0])

        if pair_i.split('-')[-1] not in roi_in_data:
            roi_in_data.append(pair_i.split('-')[-1])

    roi_in_data = np.array(roi_in_data)

    return pair_unique, roi_in_data











#dfc_data, pairs = allband_data[band][cond][phase][cf_metric_i,:,:], allpairs
def dfc_pairs_to_mat(dfc_data, pairs):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)

    mat_dfc = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_data[pairs == pair_to_find, :]
            x_rev = dfc_data[pairs == pair_to_find_rev, :]

            x_mean_pair = np.vstack([x, x_rev]).mean(axis=0)

            x_mean_pair_band = x_mean_pair.mean(axis=0)

            mat_dfc[x_i, y_i] = x_mean_pair_band

    return mat_dfc



#dfc_data, pairs = allband_data, allpairs
def plot_all_verif(dfc_data, allpairs, phase_list, prms):

    os.chdir(os.path.join(path_results, sujet, 'FC', 'verif'))

    #cf_metric_i, cf_metric = 0, 'ispc'
    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
        #pair_i = 0
        for pair_i in range(allpairs.shape[0]):

            if pair_i % 1500 == 0:

                #band_prep = 'lf'
                for band_prep in band_prep_list:

                    fig, axs = plt.subplots(ncols=len(prms['conditions']), nrows=3, figsize=(15,15))

                    if monopol:
                        plt.suptitle(f'{cf_metric}_pair{pair_i}', color='k')
                    else:
                        plt.suptitle(f'{cf_metric}_pair{pair_i}_bi', color='k')

                    #band = 'theta'
                    for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                        #cond = 'RD_SV'
                        for c, cond in enumerate(prms['conditions']):
                            #phase = 'whole'
                            for phase_i, phase in enumerate(phase_list):
                            
                                ax = axs[r,c]
                                
                                ax.plot(dfc_data[band][cond][phase][cf_metric_i,pair_i,:], label=phase)

                                if r == 0:
                                    ax.set_title(f'{cond}')
                                if c == 0:
                                    ax.set_ylabel(f'{band}')
                                # plt.show()

                    ax.legend()

                    plt.savefig(f'cf_spectre_pair{pair_i}_{cf_metric}_{band_prep}.png')
                    plt.close('all')

    #### select pairs to plot
    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(allpairs)

    pair_to_plot = []

    for pair_i, pair in enumerate(pair_unique):

        if pair_i % 30 == 0:
            pair_to_plot.append(pair)

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'

            if pair_to_find in pair_to_plot or pair_to_find_rev in pair_to_plot:

                try:
                    pair_i = np.where(pair_unique == pair_to_find)[0][0]
                except:
                    pair_i = np.where(pair_unique == pair_to_find_rev)[0][0]

                #cf_metric_i, cf_metric = 0, 'ispc'
                for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            
                    #band_prep = 'lf'
                    for band_prep in band_prep_list:

                        for phase in phase_list:

                            fig, axs = plt.subplots(ncols=len(prms['conditions']), nrows=len(freq_band_dict_FC_function[band_prep]), figsize=(15,15))

                            #band = 'theta'
                            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                                #cond = 'RD_SV'
                                for c, cond in enumerate(prms['conditions']):

                                    dfc_data_i = dfc_data[band][cond][phase][cf_metric_i,:,:]
                                    
                                    x = dfc_data_i[allpairs == pair_to_find, :]
                                    x_rev = dfc_data_i[allpairs == pair_to_find_rev, :]

                                    fc_to_plot = np.vstack([x, x_rev])
                                    
                                    ax = axs[r,c]
                                    
                                    ax.plot(fc_to_plot.mean(axis=0), label='mean')
                                    ax.plot(fc_to_plot.mean(axis=0) + fc_to_plot.std(axis=0), color='r', label='1SD')
                                    ax.plot(fc_to_plot.mean(axis=0) - fc_to_plot.std(axis=0), color='r', label='1SD')
                                    ax.plot([np.percentile(fc_to_plot, 10)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='10p')
                                    ax.plot([np.percentile(fc_to_plot, 25)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='25p')
                                    ax.plot([np.percentile(fc_to_plot, 40)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='40p')
                                    ax.plot([np.percentile(fc_to_plot, 60)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='60p')
                                    ax.plot([np.percentile(fc_to_plot, 75)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='75p')
                                    ax.plot([np.percentile(fc_to_plot, 90)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='90p')

                                    if r == 0:
                                        ax.set_title(f'{cond}')
                                    if c == 0:
                                        ax.set_ylabel(f'{band}')

                            if monopol:
                                plt.suptitle(f'{cf_metric}_{pair_to_find}_count : {fc_to_plot.shape[0]}', color='k')
                            else:
                                plt.suptitle(f'{cf_metric}_{pair_to_find}_count : {fc_to_plot.shape[0]}_bi', color='k')

                            ax.legend()

                            # plt.show()

                            plt.savefig(f'cf_mean_allpair{pair_i}_{phase}_{cf_metric}_{band_prep}.png')
                            plt.close('all')
                    
    #### export mat count pairs
    mat_count_pairs = generate_count_pairs_mat(allpairs)

    fig, ax = plt.subplots(figsize=(15,15))

    cax = ax.matshow(mat_count_pairs)

    fig.colorbar(cax, ax=ax)

    ax.set_yticks(np.arange(roi_in_data.shape[0]))
    ax.set_yticklabels(roi_in_data)

    # plt.show()
    fig.savefig(f'{sujet}_MAT_COUNT.png')
    plt.close('all')

    return 







#dfc_data, pairs = allband_data[band][cond][phase][cf_metric_i,:,:], allpairs
def generate_count_pairs_mat(pairs):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)

    mat_count_pairs = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = pairs[pairs == pair_to_find]
            x_rev = pairs[pairs == pair_to_find_rev]

            x_tot_pair = x.shape[0] + x_rev.shape[0]

            mat_count_pairs[x_i, y_i] = x_tot_pair

    return mat_count_pairs
    
















################################
######## SAVE FIG ########
################################



def process_fc_res(sujet, monopol, plot_circle_dfc=False, plot_verif=False, FR_CV_normalized=True):

    print(f'######## FC ########')

    band_prep = 'wb'

    if sujet in sujet_list:

        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

    else:

        conditions = ['FR_CV']

    #### LOAD DATA ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    if monopol:
        xr_fc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_FR_CV_allpairs.nc')
    else:
        xr_fc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_FR_CV_allpairs_bi.nc')

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(xr_fc['pairs'].data)
    cf_metrics_list = xr_fc['mat_type'].data
    n_band = len(freq_band_dict_FC_function[band_prep])

    phase_list = ['whole', 'inspi', 'expi']

    #### load data 
    allband_data = {}
    #band = 'theta'
    for band in freq_band_dict_FC_function[band_prep]:

        allband_data[band] = {}
        
        #cond = 'RD_SV'
        for cond in conditions:

            allband_data[band][cond] = {}

            for phase_i, phase in enumerate(phase_list):

                if monopol:
                    xr_fc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{cond}_allpairs.nc')
                else:
                    xr_fc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{cond}_allpairs_bi.nc')
                
                allband_data[band][cond][phase] = xr_fc.data[:, :, phase_i, :]
                allpairs = xr_fc['pairs'].data

    #### clean data
    allband_data, allpairs = clean_data(allband_data, allpairs)
    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(allpairs)

    #### plot verif
    if plot_verif:

        plot_all_verif(allband_data, allpairs, phase_list)
            
    #### mean
    #band = 'theta'
    for band in freq_band_dict_FC_function[band_prep]:
        #cond = 'RD_SV'
        for cond in conditions:
            #phase = 'whole'
            for phase_i, phase in enumerate(phase_list):
                #cf_metric_i, cf_metric = 0, 'ispc'

                mat_fc_i = np.zeros((2, len(roi_in_data), len(roi_in_data)))

                for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                    mat_fc_i[cf_metric_i, :, :] = dfc_pairs_to_mat(allband_data[band][cond][phase][cf_metric_i,:,:], allpairs)

                allband_data[band][cond][phase] = mat_fc_i

    #### normalized
    if FR_CV_normalized:

        #band = 'theta'
        for band in freq_band_dict_FC_function[band_prep]:
            #cond = 'RD_SV'
            for cond in conditions:
                if cond == 'FR_CV':
                    continue
                #phase = 'whole'
                for phase_i, phase in enumerate(phase_list):
                    #cf_metric_i, cf_metric = 0, 'ispc'

                    allband_data[band][cond][phase] = allband_data[band][cond][phase] - allband_data[band]['FR_CV'][phase]

    #### identify scales
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        #band = 'theta'
        for band in freq_band_dict_FC_function[band_prep]:

            max_list = np.array(())

            #cond = 'RD_SV'
            for cond in conditions:
                if cond == 'FR_CV':
                    continue

                for phase_i, phase in enumerate(phase_list):

                    max_list = np.append(max_list, np.abs(allband_data[band][cond][phase][mat_type_i,:,:].min()))
                    max_list = np.append(max_list, allband_data[band][cond][phase][mat_type_i,:,:].max())

            scales_abs[mat_type][band] = max_list.max()

    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = copy.deepcopy(allband_data)

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for band in freq_band_dict_FC_function[band_prep]:

            for cond in conditions:

                if cond == 'FR_CV':
                    continue

                for phase_i, phase in enumerate(phase_list):

                    thresh_up = np.percentile(allband_data[band][cond][phase][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                    thresh_down = np.percentile(allband_data[band][cond][phase][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                    for x in range(mat_dfc_clean[band][cond][phase][mat_type_i,:,:].shape[1]):
                        for y in range(mat_dfc_clean[band][cond][phase][mat_type_i,:,:].shape[1]):
                            if mat_type_i == 0:
                                if mat_dfc_clean[band][cond][phase][mat_type_i,x,y] < thresh_up:
                                    mat_dfc_clean[band][cond][phase][mat_type_i,x,y] = 0
                            else:
                                if (mat_dfc_clean[band][cond][phase][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][cond][phase][mat_type_i,x,y] > thresh_down):
                                    mat_dfc_clean[band][cond][phase][mat_type_i,x,y] = 0

    ######## PLOT ########

    #### go to results
    os.chdir(os.path.join(path_results, sujet, 'FC', 'allcond'))

    #### RAW
    n_cols_raw = len(phase_list)

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        ######## NO THRESH ########

        #cond = 'RD_SV'
        for cond in conditions:

            if cond == 'FR_CV':
                continue

            #### mat plot raw

            fig, axs = plt.subplots(nrows=len(freq_band_dict_FC_function[band_prep]), ncols=n_cols_raw, figsize=(15,15))

            if monopol:
                plt.suptitle(f'{cond} {mat_type}')
            else:
                plt.suptitle(f'{cond} {mat_type} bi')
            
            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                
                for c, phase in enumerate(phase_list):

                    ax = axs[r, c]

                    if c == 0:
                        ax.set_ylabel(band)
                    if r == 0:
                        ax.set_title(f'{phase}')
                    
                    cax = ax.matshow(allband_data[band][cond][phase][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                    fig.colorbar(cax, ax=ax)

                    ax.set_yticks(np.arange(roi_in_data.shape[0]))
                    ax.set_yticklabels(roi_in_data)

            # plt.show()

            if monopol:
                if FR_CV_normalized:
                    fig.savefig(f'MAT_{mat_type}_{cond}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'MAT_{mat_type}_{cond}_{band_prep}.png')
            else:
                if FR_CV_normalized:
                    fig.savefig(f'MAT_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'MAT_bi_{mat_type}_{cond}_{band_prep}.png')
            
            plt.close('all')

            #### circle plot RAW
                
            if plot_circle_dfc:
                
                nrows, ncols = len(freq_band_dict_FC_function[band_prep]), n_cols_raw
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                    for c, phase in enumerate(phase_list):

                        mne_connectivity.viz.plot_connectivity_circle(allband_data[band][cond][phase][mat_type_i,:,:], node_names=roi_in_data, n_lines=None, 
                                                    title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')

                if monopol:
                    plt.suptitle(f'{cond}_{mat_type}', color='k')
                else:
                    plt.suptitle(f'{cond}_{mat_type}_bi', color='k')
                
                fig.set_figheight(10)
                fig.set_figwidth(12)
                # fig.show()

                if monopol:
                    if FR_CV_normalized:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_{band_prep}.png')
                else:
                    if FR_CV_normalized:
                        fig.savefig(f'CIRCLE_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'CIRCLE_bi_{mat_type}_{cond}_{band_prep}.png')

                plt.close('all')

        ######## THRESH ########

        #cond = 'RD_SV'
        for cond in conditions:

            if cond == 'FR_CV':
                continue

            #### mat plot raw 

            fig, axs = plt.subplots(nrows=len(freq_band_dict_FC_function[band_prep]), ncols=n_cols_raw, figsize=(15,15))

            if monopol:
                plt.suptitle(f'{cond} {mat_type}')
            else:
                plt.suptitle(f'{cond} {mat_type} bi')
            
            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                for c, phase in enumerate(phase_list):

                    ax = axs[r, c]

                    if c == 0:
                        ax.set_ylabel(band)
                    if r == 0:
                        ax.set_title(f'{phase}')
                    
                    cax = ax.matshow(mat_dfc_clean[band][cond][phase][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                    fig.colorbar(cax, ax=ax)

                    ax.set_yticks(np.arange(roi_in_data.shape[0]))
                    ax.set_yticklabels(roi_in_data)

            # plt.show()

            if monopol:
                if FR_CV_normalized:
                    fig.savefig(f'THRESH_MAT_{mat_type}_{cond}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'THRESH_MAT_{mat_type}_{cond}_{band_prep}.png')
            else:
                if FR_CV_normalized:
                    fig.savefig(f'THRESH_MAT_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'THRESH_MAT_bi_{mat_type}_{cond}_{band_prep}.png')
            
            plt.close('all')

            #### circle plot RAW
                
            if plot_circle_dfc:
                
                nrows, ncols = len(freq_band_dict_FC_function[band_prep]), n_cols_raw
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                    for c, phase in enumerate(phase_list):

                        mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean[band][cond][phase][mat_type_i,:,:], node_names=roi_in_data, n_lines=None, 
                                                    title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')

                if monopol:
                    plt.suptitle(f'{cond}_{mat_type}', color='k')
                else:
                    plt.suptitle(f'{cond}_{mat_type}_bi', color='k')
                
                fig.set_figheight(10)
                fig.set_figwidth(12)
                
                # fig.show()

                if monopol:
                    if FR_CV_normalized:
                        fig.savefig(f'THRESH_CIRCLE_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'THRESH_CIRCLE_{mat_type}_{cond}_{band_prep}.png')
                else:
                    if FR_CV_normalized:
                        fig.savefig(f'THRESH_CIRCLE_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'THRESH_CIRCLE_bi_{mat_type}_{cond}_{band_prep}.png')

                plt.close('all')




    ######## SUMMARY ########




    #### go to results
    os.chdir(os.path.join(path_results, sujet, 'FC', 'summary'))

    #### RAW

    cond_to_plot = [cond for cond in conditions if cond != 'FR_CV']

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for band_prep in band_prep_list:

            n_cols_raw = len(cond_to_plot)

            ######## NO THRESH ########
            #cond = 'RD_SV'
            for phase in phase_list:

                #### mat plot raw 
                fig, axs = plt.subplots(nrows=len(freq_band_dict_FC_function[band_prep]), ncols=n_cols_raw, figsize=(15,15))

                if monopol:
                    plt.suptitle(f'{phase} {mat_type}')
                else:
                    plt.suptitle(f'{phase} {mat_type} bi')
                
                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                    for c, cond in enumerate(cond_to_plot):

                        ax = axs[r, c]

                        if c == 0:
                            ax.set_ylabel(band)
                        if r == 0:
                            ax.set_title(cond)
                        
                        cax = ax.matshow(allband_data[band][cond][phase][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                        fig.colorbar(cax, ax=ax)

                        ax.set_yticks(np.arange(roi_in_data.shape[0]))
                        ax.set_yticklabels(roi_in_data)
                # plt.show()

                if monopol:
                    if FR_CV_normalized:
                        fig.savefig(f'summary_MAT_{mat_type}_{phase}_norm.png')
                    else:
                        fig.savefig(f'summary_MAT_{mat_type}_{phase}.png')
                else:
                    if FR_CV_normalized:
                        fig.savefig(f'summary_MAT_bi_{mat_type}_{phase}_norm.png')
                    else:
                        fig.savefig(f'summary_MAT_bi_{mat_type}_{phase}.png')
                
                plt.close('all')
                    
                #### circle plot RAW

                if plot_circle_dfc:
                        
                    nrows, ncols = len(freq_band_dict_FC_function[band_prep]), n_cols_raw
                    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                    for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                        for c, cond in enumerate(cond_to_plot):

                            mne_connectivity.viz.plot_connectivity_circle(allband_data[band][cond][phase][mat_type_i,:,:], node_names=roi_in_data, n_lines=None, 
                                                        title=f'{band} {cond}', show=False, padding=7, ax=axs[r, c],
                                                        vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                        textcolor='k')

                    if monopol:
                        plt.suptitle(f'{phase}_{mat_type}', color='k')
                    else:
                        plt.suptitle(f'{phase}_{mat_type}_bi', color='k')
                    
                    fig.set_figheight(10)
                    fig.set_figwidth(12)
                    # fig.show()

                    if monopol:
                        if FR_CV_normalized:
                            fig.savefig(f'summary_CIRCLE_{mat_type}_{phase}_norm.png')
                        else:
                            fig.savefig(f'summary_CIRCLE_{mat_type}_{phase}.png')
                    else:
                        if FR_CV_normalized:
                            fig.savefig(f'summary_CIRCLE_bi_{mat_type}_{phase}_norm.png')
                        else:
                            fig.savefig(f'summary_CIRCLE_bi_{mat_type}_{phase}.png')

                    plt.close('all')


                ######## THRESH ########

                #### mat plot raw 
                fig, axs = plt.subplots(nrows=len(freq_band_dict_FC_function[band_prep]), ncols=n_cols_raw, figsize=(15,15))

                if monopol:
                    plt.suptitle(f'{phase} {mat_type}')
                else:
                    plt.suptitle(f'{phase} {mat_type} bi')
                
                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                    for c, cond in enumerate(cond_to_plot):

                        ax = axs[r, c]

                        if c == 0:
                            ax.set_ylabel(band)
                        if r == 0:
                            ax.set_title(f'{phase}')
                        
                        cax = ax.matshow(mat_dfc_clean[band][cond][phase][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                        fig.colorbar(cax, ax=ax)

                        ax.set_yticks(np.arange(roi_in_data.shape[0]))
                        ax.set_yticklabels(roi_in_data)
                # plt.show()

                if monopol:
                    if FR_CV_normalized:
                        fig.savefig(f'summary_THRESH_MAT_{mat_type}_{phase}_norm.png')
                    else:
                        fig.savefig(f'summary_THRESH_MAT_{mat_type}_{phase}.png')
                else:
                    if FR_CV_normalized:
                        fig.savefig(f'summary_THRESH_MAT_bi_{mat_type}_{phase}_norm.png')
                    else:
                        fig.savefig(f'summary_THRESH_MAT_bi_{mat_type}_{phase}.png')
                
                plt.close('all')
                    
                #### circle plot RAW

                if plot_circle_dfc:
                        
                    nrows, ncols = len(freq_band_dict_FC_function[band_prep]), n_cols_raw
                    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                    for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                        for c, cond in enumerate(cond_to_plot):

                            mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean[band][cond][phase][mat_type_i,:,:], node_names=roi_in_data, n_lines=None, 
                                                        title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                        vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                        textcolor='k')

                    if monopol:
                        plt.suptitle(f'{phase}_{mat_type}', color='k')
                    else:
                        plt.suptitle(f'{phase}_{mat_type}_bi', color='k')
                    
                    fig.set_figheight(10)
                    fig.set_figwidth(12)
                    # fig.show()

                    if monopol:
                        if FR_CV_normalized:
                            fig.savefig(f'summary_THRESH_CIRCLE_{mat_type}_{phase}_norm.png')
                        else:
                            fig.savefig(f'summary_THRESH_CIRCLE_{mat_type}_{phase}.png')
                    else:
                        if FR_CV_normalized:
                            fig.savefig(f'summary_THRESH_CIRCLE_bi_{mat_type}_{phase}_norm.png')
                        else:
                            fig.savefig(f'summary_THRESH_CIRCLE_bi_{mat_type}_{phase}.png')

                    plt.close('all')





################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list_FR_CV[0]
    for sujet in sujet_list_FR_CV:

        #monopol = True
        for monopol in [True, False]:

            # process_fc_res(sujet, monopol)
            execute_function_in_slurm_bash('n9_res_FC', 'process_fc_res', [sujet, monopol])

