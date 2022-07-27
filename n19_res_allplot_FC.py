

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import copy


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





################################
######## PRECOMPUTE MAT ########
################################



def get_ROI_names_for_sujet_list(sujet_list_selected, cond):

    #### identify plot names for all sujet
    ROI_list = []

    for sujet in sujet_list_selected:

        band = list(freq_band_dict_FC_function[band_prep_list[0]].keys())[0]

        os.chdir(os.path.join(path_precompute, sujet, 'FC'))
        xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
        ROI_list_sujet = xr_dfc['x'].data

        [ROI_list.append(i) for i in ROI_list_sujet if i not in ROI_list]

    ROI_list = np.array(ROI_list)

    #### identify count number for each pairs
    ROI_count = np.zeros(( len(ROI_list), len(ROI_list) ))

    for sujet in sujet_list_selected:

        band = list(freq_band_dict_FC_function[band_prep_list[0]].keys())[0]

        os.chdir(os.path.join(path_precompute, sujet, 'FC'))
        xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
        ROI_list_sujet = xr_dfc['x'].data

        for x in ROI_list_sujet:
            for y in ROI_list_sujet:
                if x == y:
                    continue
                x_i_post = np.where(ROI_list == x)[0]
                y_i_post = np.where(ROI_list == y)[0]

                ROI_count[x_i_post, y_i_post] += 1

    if debug:
        plt.matshow(ROI_count)
        plt.show()

    return np.array(ROI_list), ROI_count




#cond = 'RD_CV'
def precompute_fc_mat_allplot(cond):

    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list_FR_CV:
        prms_i = get_params(sujet_i)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### get ROI names
    ROI_list, ROI_count = get_ROI_names_for_sujet_list(sujet_list_selected, cond)

    #### initiate containers
    mat_allplot = {}

    #cf_metric = 'ispc'
    for cf_metric in ['ispc', 'wpli']:
        mat_allplot[cf_metric] = {}
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'theta', [4,8]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                mat_band_allplot = np.zeros(( len(ROI_list), len(ROI_list) ))

                #sujet = sujet_list_selected[0]
                for sujet in sujet_list_selected:

                    #### extract data
                    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
                    xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
                    ROI_sujet_i = xr_dfc['x'].data

                    mat_i = xr_dfc.loc[cf_metric,:,:].data

                    #### fill global mat
                    for x_i_pre, x in enumerate(ROI_sujet_i):
                        for y_i_pre, y in enumerate(ROI_sujet_i):
                            if x == y:
                                continue
                            x_i_post = np.where(ROI_list == x)[0]
                            y_i_post = np.where(ROI_list == y)[0]

                            mat_band_allplot[x_i_post, y_i_post] += mat_i[x_i_pre, y_i_pre]

                for x_i, x in enumerate(ROI_list):
                        for y_i, y in enumerate(ROI_list):
                            if mat_band_allplot[x_i, y_i] != 0:
                                mat_band_allplot[x_i, y_i] /= ROI_count[x_i, y_i]

                if debug:
                    plt.matshow(mat_band_allplot)
                    plt.show()

                #### fill containers
                mat_allplot[cf_metric][band] = mat_band_allplot

    return mat_allplot, ROI_list





################################
######## SAVE FIG ######## 
################################

#mat = mat_allplot_reduced
def process_fc_res_for_cond(cond):

    #### extract band names
    band_names = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys()) if band_name_i != 'whole']

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_names[0]) != -1)]
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data

    #### load data 
    data_cond, ROI_list = precompute_fc_mat_allplot(cond)

    #### thresh
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    data_cond_clean = copy.deepcopy(data_cond)

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for band in band_names:

            thresh_up = np.percentile(data_cond[mat_type][band].reshape(-1), percentile_thresh_up)
            thresh_down = np.percentile(data_cond[mat_type][band].reshape(-1), percentile_thresh_down)

            for x in range(data_cond_clean[mat_type][band].shape[1]):
                for y in range(data_cond_clean[mat_type][band].shape[1]):
                    if (data_cond_clean[mat_type][band][x,y] < thresh_up) & (data_cond_clean[mat_type][band][x,y] > thresh_down):
                        data_cond_clean[mat_type][band][x,y] = 0

    #### adjust scale
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_names:

            max_list = np.array(())

            max_list = np.append(max_list, data_cond[mat_type][band].max())
            max_list = np.append(max_list, np.abs(data_cond[mat_type][band].min()))

            scales_abs[mat_type][band] = max_list.max()

    print(f'######## PLOT {cond} DIFF ########')

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'FC', 'allcond'))
    mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band_used = {key : value for key, value in freq_band_list[band_prep_i].items() if key != 'whole'}

        n_rows = 1
        n_cols = len(freq_band_used)

        #### plot    
        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #### mat plot
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,15))
            plt.suptitle(mat_type)
            for c, band in enumerate(freq_band_used):
                ax = axs[c]
                ax.set_title(f'{cond} {band}')
                ax.matshow(data_cond[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                if c == 0:
                    ax.set_yticks(np.arange(ROI_list.shape[0]))
                    ax.set_yticklabels(ROI_list)
            # plt.show()
            fig.savefig(f'{cond}_MAT_DIFF_{mat_type}_{band_prep}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_rows, n_cols
            fig = plt.figure()
            _position = 0

            for c, band in enumerate(freq_band_used):

                _position += 1

                mne.viz.plot_connectivity_circle(data_cond[mat_type][band], node_names=ROI_list, n_lines=None, 
                                            title=f'{band}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

            plt.suptitle(f'{cond}_{mat_type}', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'{cond}_CIRCLE_DIFF_{mat_type}_{band_prep}.png')
            plt.close('all')

            #### THRESH
            #### mat plot
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,15))
            plt.suptitle(f'{mat_type} THRESH')
            for c, band in enumerate(freq_band_used):
                ax = axs[c]
                ax.set_title(f'{cond} {band}')
                ax.matshow(data_cond_clean[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                if c == 0:
                    ax.set_yticks(np.arange(ROI_list.shape[0]))
                    ax.set_yticklabels(ROI_list)
            # plt.show()
            fig.savefig(f'{cond}_MAT_DIFF_TRESH_{mat_type}_{band_prep}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_rows, n_cols
            fig = plt.figure()
            _position = 0

            for c, band in enumerate(freq_band_used):

                _position += 1

                mne.viz.plot_connectivity_circle(data_cond_clean[mat_type][band], node_names=ROI_list, n_lines=None, 
                                            title=f'{band}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

            plt.suptitle(f'{cond}_{mat_type}_THRESH', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'{cond}_CIRCLE_DIFF_THRESH_{mat_type}_{band_prep}.png')
            plt.close('all')






################################
######## SUMMARY ########
################################


def process_fc_res(cond_to_compute):

    print(f'######## SUMMARY FC ########')

    #### CONNECTIVITY PLOT ####

    #### extract band names
    band_names = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys()) if band_name_i != 'whole']

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_names[0]) != -1)]
    roi_names = xr.open_dataarray(file_to_load[0])['x'].data
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data

    #### load allcond data 
    allcond_data = {}
    allcond_scales_abs = {}
    allcond_ROI_list = {}

    #cond = 'RD_CV'
    for cond in cond_to_compute:
        #### load data
        allcond_data_i, ROI_list = precompute_fc_mat_allplot(cond)
        allcond_ROI_list[cond] = ROI_list

        #### scale abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_names:

                scales_abs[mat_type][band] = allcond_data_i[mat_type][band].max()

        allcond_scales_abs[cond] = scales_abs

        #### thresh
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean_i = copy.deepcopy(allcond_data_i)

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_names:

                thresh_up = np.percentile(allcond_data_i[mat_type][band].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(allcond_data_i[mat_type][band].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean_i[mat_type][band].shape[1]):
                    for y in range(mat_dfc_clean_i[mat_type][band].shape[1]):
                        if (mat_dfc_clean_i[mat_type][band][x,y] < thresh_up) & (mat_dfc_clean_i[mat_type][band][x,y] > thresh_down):
                            mat_dfc_clean_i[mat_type][band][x,y] = 0

        #### fill res containers
        allcond_data[cond] = mat_dfc_clean_i

    #### adjust scale
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_names:

            max_list = np.array(())

            for cond in cond_to_compute:

                max_list = np.append(max_list, allcond_scales_abs[cond][mat_type][band])

            scales_abs[mat_type][band] = max_list.max()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'FC', 'summary'))

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band_used = {key : value for key, value in freq_band_list[band_prep_i].items() if key != 'whole'}

        n_rows = len(freq_band_used)
        n_cols = len(cond_to_compute)

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #### mat
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,15))
            plt.suptitle(f'{mat_type} summary THRESH : inspi - expi')
            for r, band in enumerate(freq_band_used):
                for c, cond in enumerate(cond_to_compute):
                    ax = axs[r, c]
                    if c == 0:
                        ax.set_ylabel(band)
                    if r == 0:
                        ax.set_title(f'{cond}')

                    ax.matshow(allcond_data[cond][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    
                    if c == 0:
                        ax.set_yticks(np.arange(allcond_ROI_list[cond].shape[0]))
                        ax.set_yticklabels(allcond_ROI_list[cond])
                    elif c != 0 and allcond_ROI_list[cond].shape[0] != allcond_ROI_list[cond_to_compute[c-1]].shape[0]:
                        ax.set_yticks(np.arange(allcond_ROI_list[cond].shape[0]))
                        ax.set_yticklabels(allcond_ROI_list[cond])

            # plt.show()
            fig.savefig(f'summary_MAT_DIFF_TRESH_{mat_type}.png')
            plt.close('all')

            #### circle plot
            fig = plt.figure()
            _position = 0

            for r, band in enumerate(band_name_fc_dfc):

                for c, cond in enumerate(cond_to_compute):

                    _position += 1

                    mne.viz.plot_connectivity_circle(allcond_data[cond][mat_type][band], node_names=allcond_ROI_list[cond], n_lines=None, 
                                                title=f'{cond} {band}', show=False, padding=7, fig=fig, subplot=(n_rows, n_cols, _position),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')

            plt.suptitle(f'{mat_type}_THRESH : inspi - expi', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'summary_CIRCLE_DIFF_TRESH_{mat_type}.png')
            plt.close('all')









################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #### fo cond
    cond_to_compute = ['RD_CV', 'RD_SV', 'RD_FV', 'FR_CV']

    for cond in cond_to_compute:

        process_fc_res_for_cond(cond)

    #### summary
    process_fc_res(cond_to_compute)








