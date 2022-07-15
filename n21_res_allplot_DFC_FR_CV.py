

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






################################################
######## COMPUTE DATA RESPI PHASE ########
################################################


#data_dfc, pairs, roi_in_data = data_chunk.loc[cf_metric,:,:].data, data['pairs'].data, roi_in_data
def from_dfc_to_mat_conn_trpz(data_dfc, pairs, roi_in_data):

    #### fill mat
    mat_cf = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue
            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = data_dfc[pairs == pair_to_find]
            x_rev = data_dfc[pairs == pair_to_find_rev]

            x_mean = np.vstack([x, x_rev]).mean(axis=0)
            val_to_place = np.trapz(x_mean)

            mat_cf[x_i, y_i] = val_to_place

    if debug:
        plt.matshow(mat_cf)
        plt.show()

    return mat_cf





################################
######## PRECOMPUTE MAT ########
################################



def precompute_dfc_mat_allplot(cond):

    #### initiate containers
    mat_allplot = {}
    mat_pairs_allplot = {}

    #cf_metric = 'ispc'
    for cf_metric in ['ispc', 'wpli']:
        mat_allplot[cf_metric] = {}
        mat_pairs_allplot[cf_metric] = {}
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'beta', [10,40]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:

                    mat_allplot[cf_metric][band] = np.array([])
                    mat_pairs_allplot[cf_metric][band] = np.array([])

                    #sujet = sujet_list_FR_CV[0]
                    for sujet in sujet_list_FR_CV:

                        #### extract data
                        os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
                        xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
                        ROI = xr_dfc['x'].data

                        #### generate mat ROI names
                        for row_i, ROI_row_i in enumerate(ROI):
                            
                            mat_roi_row_i = np.array([])
                            
                            for ROI_col_i in ROI:
                                
                                pair_i = f'{ROI_row_i}-{ROI_col_i}'
                                mat_roi_row_i = np.append(mat_roi_row_i, pair_i)

                            if row_i == 0:
                                mat_roi = mat_roi_row_i
                            else:
                                mat_roi = np.vstack([mat_roi, mat_roi_row_i])

                        #### extract data from mat
                        mat = xr_dfc.loc[cf_metric,:,:].data
                        mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                        mat_pairs_names = mat_roi[np.triu_indices(mat_roi.shape[0], k=1)]

                        #### fill containers
                        mat_allplot[cf_metric][band] = np.append(mat_allplot[cf_metric][band], mat_values)
                        mat_pairs_allplot[cf_metric][band] = np.append(mat_pairs_allplot[cf_metric][band], mat_pairs_names)

    #### reduce vector
    mat_allplot_unique = {}

    pairs_unique = np.unique(mat_pairs_allplot[cf_metric][band])

    for cf_metric in ['ispc', 'wpli']:
        mat_allplot_unique[cf_metric] = {}
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'beta', [10,40]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:

                    mat_allplot_unique[cf_metric][band] = np.array([])

                    #pair_i = pairs_unique[0]
                    for pair_name_i in pairs_unique:
                        
                        mask_pair_i = np.where(mat_pairs_allplot[cf_metric][band] == pair_name_i)[0]
                        mat_allplot_unique[cf_metric][band] = np.append( mat_allplot_unique[cf_metric][band], np.mean(mat_allplot[cf_metric][band][mask_pair_i]) )

    #### identify missing pairs
    ROI_list = np.array([])

    for pair_i in pairs_unique:

        pair_A, pair_B = pair_i.split('-')[0], pair_i.split('-')[1]

        if pair_A not in ROI_list:
            ROI_list = np.append(ROI_list, pair_A)
        if pair_B not in ROI_list:
            ROI_list = np.append(ROI_list, pair_B)

    mat_verif = np.zeros(( ROI_list.shape[0], ROI_list.shape[0]))
    for ROI_row_i, ROI_row_name in enumerate(ROI_list):
        #ROI_col_i, ROI_col_name = 1, ROI_list[1]
        for ROI_col_i, ROI_col_name in enumerate(ROI_list):

            if ROI_row_name == ROI_col_name:
                mat_verif[ROI_row_i, ROI_col_i] = 1
                continue

            pair_name = f'{ROI_row_name}-{ROI_col_name}'
            pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

            if pair_name in pairs_unique:
                mat_verif[ROI_row_i, ROI_col_i] = 1
            if pair_name_rev in pairs_unique:
                mat_verif[ROI_row_i, ROI_col_i] = 1

    if debug:
        plt.matshow(mat_verif)
        plt.show()

    #### export missig matrix
    os.chdir(os.path.join(path_results, 'allplot', cond, 'DFC', 'ROI'))
    plt.matshow(mat_verif)
    plt.savefig('missing_mat.png')
    plt.close('all')

    #### generate reduced mat 

    mat_allplot_reduced = {}

    for cf_metric in ['ispc', 'wpli']:
        mat_allplot_reduced[cf_metric] = {}
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'beta', [10,40]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:

                    mat_allplot_reduced[cf_metric][band] = np.zeros(( ROI_list.shape[0], ROI_list.shape[0] ))

                    #ROI_row_i, ROI_row_name = 0, ROI_list[0]
                    for ROI_row_i, ROI_row_name in enumerate(ROI_list):
                        #ROI_col_i, ROI_col_name = 1, ROI_list[1]
                        for ROI_col_i, ROI_col_name in enumerate(ROI_list):

                            if ROI_row_name == ROI_col_name:
                                continue

                            pair_name = f'{ROI_row_name}-{ROI_col_name}'
                            pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                            if np.where(pairs_unique == pair_name)[0].shape[0] == 1:
                                dfc_val = mat_allplot_unique[cf_metric][band][np.where(pairs_unique == pair_name)[0]]
                            else:
                                dfc_val = mat_allplot_unique[cf_metric][band][np.where(pairs_unique == pair_name_rev)[0]]
                            
                            if dfc_val.shape[0] == 0:
                                continue
                            else:
                                mat_allplot_reduced[cf_metric][band][ROI_row_i, ROI_col_i] = dfc_val

                            if debug:
                                plt.matshow(mat_allplot_reduced)
                                plt.show()


    return mat_allplot_reduced, ROI_list



def precompute_dfc_mat_allplot_respi_phase(cond):

    respi_phase_list = ['inspi', 'expi']

    #### initiate containers
    mat_allplot = {}
    mat_pairs_allplot = {}

    for respi_phase in respi_phase_list:
        mat_allplot[respi_phase] = {}
        mat_pairs_allplot[respi_phase] = {}
        #cf_metric = 'ispc'
        for cf_metric in ['ispc', 'wpli']:
            mat_allplot[respi_phase][cf_metric] = {}
            mat_pairs_allplot[respi_phase][cf_metric] = {}
            #band_prep = 'lf'
            for band_prep in band_prep_list:
                #band, freq = 'beta', [10,40]
                for band, freq in freq_band_dict_FC_function[band_prep].items():

                    if band in ['beta', 'l_gamma', 'h_gamma']:

                        mat_allplot[respi_phase][cf_metric][band] = np.array([])
                        mat_pairs_allplot[respi_phase][cf_metric][band] = np.array([])

                        #sujet = sujet_list_FR_CV[0]
                        for sujet in sujet_list_FR_CV:

                            #### extract data
                            os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
                            xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
                            ROI = xr_dfc['x'].data

                            #### generate mat ROI names
                            for row_i, ROI_row_i in enumerate(ROI):
                                
                                mat_roi_row_i = np.array([])
                                
                                for ROI_col_i in ROI:
                                    
                                    pair_i = f'{ROI_row_i}-{ROI_col_i}'
                                    mat_roi_row_i = np.append(mat_roi_row_i, pair_i)

                                if row_i == 0:
                                    mat_roi = mat_roi_row_i
                                else:
                                    mat_roi = np.vstack([mat_roi, mat_roi_row_i])

                            #### generate mat for correct respi phase
                            xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                            if respi_phase == 'inspi':
                                mat = from_dfc_to_mat_conn_trpz(xr_dfc.loc[cf_metric,:,:int(stretch_point_TF*ratio_stretch_TF)].data, xr_dfc['pairs'], ROI)
                            if respi_phase == 'expi':
                                mat = from_dfc_to_mat_conn_trpz(xr_dfc.loc[cf_metric,:,int(stretch_point_TF*ratio_stretch_TF):].data, xr_dfc['pairs'], ROI)

                            #### extract data from mat
                            mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                            mat_pairs_names = mat_roi[np.triu_indices(mat_roi.shape[0], k=1)]

                            #### fill containers
                            mat_allplot[respi_phase][cf_metric][band] = np.append(mat_allplot[respi_phase][cf_metric][band], mat_values)
                            mat_pairs_allplot[respi_phase][cf_metric][band] = np.append(mat_pairs_allplot[respi_phase][cf_metric][band], mat_pairs_names)

    #### reduce vector
    mat_allplot_unique = {}

    pairs_unique = np.unique(mat_pairs_allplot[respi_phase][cf_metric][band])

    for respi_phase in respi_phase_list:

        mat_allplot_unique[respi_phase] = {}

        for cf_metric in ['ispc', 'wpli']:
            mat_allplot_unique[respi_phase][cf_metric] = {}
            #band_prep = 'lf'
            for band_prep in band_prep_list:
                #band, freq = 'beta', [10,40]
                for band, freq in freq_band_dict_FC_function[band_prep].items():

                    if band in ['beta', 'l_gamma', 'h_gamma']:

                        mat_allplot_unique[respi_phase][cf_metric][band] = np.array([])

                        #pair_i = pairs_unique[0]
                        for pair_name_i in pairs_unique:
                            
                            mask_pair_i = np.where(mat_pairs_allplot[respi_phase][cf_metric][band] == pair_name_i)[0]
                            mat_allplot_unique[respi_phase][cf_metric][band] = np.append( mat_allplot_unique[respi_phase][cf_metric][band], np.mean(mat_allplot[respi_phase][cf_metric][band][mask_pair_i]) )

    #### generate ROI list
    ROI_list = np.array([])

    for pair_i in pairs_unique:

        pair_A, pair_B = pair_i.split('-')[0], pair_i.split('-')[1]

        if pair_A not in ROI_list:
            ROI_list = np.append(ROI_list, pair_A)
        if pair_B not in ROI_list:
            ROI_list = np.append(ROI_list, pair_B)

    #### generate reduced mat 
    mat_allplot_reduced = {}

    for respi_phase in respi_phase_list:

        mat_allplot_reduced[respi_phase] = {}

        for cf_metric in ['ispc', 'wpli']:
            mat_allplot_reduced[respi_phase][cf_metric] = {}
            #band_prep = 'lf'
            for band_prep in band_prep_list:
                #band, freq = 'beta', [10,40]
                for band, freq in freq_band_dict_FC_function[band_prep].items():

                    if band in ['beta', 'l_gamma', 'h_gamma']:

                        mat_allplot_reduced[respi_phase][cf_metric][band] = np.zeros(( ROI_list.shape[0], ROI_list.shape[0] ))

                        #ROI_row_i, ROI_row_name = 0, ROI_list[0]
                        for ROI_row_i, ROI_row_name in enumerate(ROI_list):
                            #ROI_col_i, ROI_col_name = 1, ROI_list[1]
                            for ROI_col_i, ROI_col_name in enumerate(ROI_list):

                                if ROI_row_name == ROI_col_name:
                                    continue

                                pair_name = f'{ROI_row_name}-{ROI_col_name}'
                                pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                                if np.where(pairs_unique == pair_name)[0].shape[0] == 1:
                                    dfc_val = mat_allplot_unique[respi_phase][cf_metric][band][np.where(pairs_unique == pair_name)[0]]
                                else:
                                    dfc_val = mat_allplot_unique[respi_phase][cf_metric][band][np.where(pairs_unique == pair_name_rev)[0]]
                                
                                if dfc_val.shape[0] == 0:
                                    continue
                                else:
                                    mat_allplot_reduced[respi_phase][cf_metric][band][ROI_row_i, ROI_col_i] = dfc_val

                                if debug:
                                    plt.matshow(mat_allplot_reduced[respi_phase])
                                    plt.show()


    return mat_allplot_reduced, ROI_list





################################
######## SAVE FIG ########
################################

#mat = mat_allplot_reduced
def save_fig_dfc(cond, mat, ROI_list):

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_dfc[0]) != -1)]
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_dfc)

    print(f'######## PLOT {cond} ########')
    os.chdir(os.path.join(path_results, 'allplot', cond, 'DFC', 'ROI'))

    #### plot
    roi_names = ROI_list
    mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

    #### identify scales
    scales = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

        for band in band_name_dfc:

            # mat_scaled = mat[mat_type][band][mat[band][mat_type_i,:,:] != 0]
            mat_scaled = mat[mat_type][band]

            scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
            scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

        scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

    #### identify scales abs
    scales_abs = {}
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_name_dfc:

            max_list = np.array(())
            max_list = np.append(max_list, mat[mat_type][band].max())
            max_list = np.append(max_list, np.abs(mat[mat_type][band].min()))

            scales_abs[mat_type][band] = max_list.max()

    #mat_type = 'ispc'
    for mat_type in ['ispc', 'wpli']:

        #### mat plot
        fig, axs = plt.subplots(ncols=n_band, figsize=(15,15))
        plt.suptitle(mat_type)
        for c, band in enumerate(band_name_dfc):
            ax = axs[c]
            ax.set_title(band)
            ax.matshow(mat[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
            # ax.matshow(mat[mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
            # ax.matshow(mat[mat_type][band])
            if c == 0:
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        nrows, ncols = 1, n_band
        fig = plt.figure()
        for c, band in enumerate(band_name_dfc):
            # mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
            #                                 textcolor='k')
            # mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 colormap=mat_type_color[mat_type], facecolor='w', 
            #                                 textcolor='k')
            mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
        plt.suptitle(f'{cond}_{mat_type}', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_{cond}_{mat_type}.png')
        plt.close('all')


    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = mat.copy()

    for mat_type_i, mat_type in enumerate(['ispc', 'wpli']):

        for band in band_name_dfc:

            thresh_up = np.percentile(mat[mat_type][band].reshape(-1), percentile_thresh_up)
            thresh_down = np.percentile(mat[mat_type][band].reshape(-1), percentile_thresh_down)

            for x in range(mat_dfc_clean[mat_type][band].shape[1]):
                for y in range(mat_dfc_clean[mat_type][band].shape[1]):
                    if mat_type_i == 0:
                        if mat_dfc_clean[mat_type][band][x,y] < thresh_up:
                            mat_dfc_clean[mat_type][band][x,y] = 0
                    else:
                        if (mat_dfc_clean[mat_type][band][x,y] < thresh_up) & (mat_dfc_clean[mat_type][band][x,y] > thresh_down):
                            mat_dfc_clean[mat_type][band][x,y] = 0

    #### plot
    #mat_type = 'wpli'
    for mat_type in ['ispc', 'wpli']:

        #### mat plot
        fig, axs = plt.subplots(ncols=n_band, figsize=(15,15))
        plt.suptitle(f'{mat_type} : THRESH')
        for c, band in enumerate(band_name_dfc):
            ax = axs[c]
            ax.set_title(band)
            ax.matshow(mat_dfc_clean[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
            # ax.matshow(mat_dfc_clean[mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
            # ax.matshow(mat_dfc_clean[mat_type][band])
            if c == 0:
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_thresh_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        nrows, ncols = 1, n_band
        fig = plt.figure()
        for c, band in enumerate(band_name_dfc):
            # mne.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=cm.seismic, facecolor='w', 
            #                                 textcolor='k')
            # mne.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 colormap=mat_type_color[mat_type], facecolor='w', 
            #                                 textcolor='k')
            mne.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
        plt.suptitle(f'{cond}_{mat_type} : THRESH', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_thresh_{cond}_{mat_type}.png')
        plt.close('all')





#mat = mat_allplot_reduced
def save_fig_dfc_respi_phase(cond, mat, ROI_list):

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_dfc[0]) != -1)]
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_dfc)

    respi_phase_list = ['inspi', 'expi']
    n_rows = len(respi_phase_list)

    print(f'######## PLOT {cond} PHASE ########')

    #### identify scales
    scales = {}

    for respi_phase in respi_phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

            for band in band_name_dfc:

                # mat_scaled = mat[respi_phase][mat_type][band][mat[respi_phase][band][mat_type_i,:,:] != 0]
                mat_scaled = mat[respi_phase][mat_type][band]

                scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
                scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

            scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

    #### identify scales abs
    scales_abs = {}

    for respi_phase in respi_phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_dfc:

                max_list = np.array(())
                max_list = np.append(max_list, mat[respi_phase][mat_type][band].max())
                max_list = np.append(max_list, np.abs(mat[respi_phase][mat_type][band].min()))

                scales_abs[mat_type][band] = max_list.max()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', cond, 'DFC', 'ROI'))
    roi_names = ROI_list
    mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #### mat plot
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_band, figsize=(15,15))
        plt.suptitle(mat_type)
        for r, respi_phase in enumerate(respi_phase_list):
            for c, band in enumerate(band_name_dfc):
                ax = axs[r, c]
                if c == 0:
                    ax.set_ylabel(respi_phase)
                ax.set_title(band)
                ax.matshow(mat[respi_phase][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(mat[respi_phase][mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(mat[respi_phase][mat_type][band])
                if c == 0:
                    ax.set_yticks(np.arange(roi_names.shape[0]))
                    ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_PHASE_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        nrows, ncols = n_rows, n_band
        fig = plt.figure()
        for r, respi_phase in enumerate(respi_phase_list):
            if r == 1:
                r = len(band_name_dfc)
            for c, band in enumerate(band_name_dfc):
                # mne.viz.plot_connectivity_circle(mat[respi_phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                # mne.viz.plot_connectivity_circle(mat[respi_phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(mat[respi_phase][mat_type][band], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
        plt.suptitle(f'{cond}_{mat_type}', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_PHASE_{cond}_{mat_type}.png')
        plt.close('all')


    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = mat.copy()

    for respi_phase in respi_phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_name_dfc:

                thresh_up = np.percentile(mat[respi_phase][mat_type][band].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(mat[respi_phase][mat_type][band].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[respi_phase][mat_type][band].shape[1]):
                    for y in range(mat_dfc_clean[respi_phase][mat_type][band].shape[1]):
                        if mat_type_i == 0:
                            if mat_dfc_clean[respi_phase][mat_type][band][x,y] < thresh_up:
                                mat_dfc_clean[respi_phase][mat_type][band][x,y] = 0
                        else:
                            if (mat_dfc_clean[respi_phase][mat_type][band][x,y] < thresh_up) & (mat_dfc_clean[respi_phase][mat_type][band][x,y] > thresh_down):
                                mat_dfc_clean[respi_phase][mat_type][band][x,y] = 0

    #### plot
    #mat_type = 'wpli'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #### mat plot
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_band, figsize=(15,15))
        plt.suptitle(f'{mat_type} : THRESH')

        for r, respi_phase in enumerate(respi_phase_list):
            for c, band in enumerate(band_name_dfc):
                ax = axs[r, c]
                if c == 0:
                    ax.set_ylabel(respi_phase)
                    ax.yaxis.set_label_coords(-.5, .5)
                ax.set_title(band)
                ax.matshow(mat_dfc_clean[respi_phase][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(mat_dfc_clean[respi_phase][mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(mat_dfc_clean[respi_phase][mat_type][band])
                if c == 0:
                    ax.set_yticks(np.arange(roi_names.shape[0]))
                    ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_PHASE_thresh_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        nrows, ncols = n_rows, n_band
        fig = plt.figure()
        for r, respi_phase in enumerate(respi_phase_list):
            if r == 1:
                r = len(band_name_dfc)
            for c, band in enumerate(band_name_dfc):
                # mne.viz.plot_connectivity_circle(mat_dfc_clean[respi_phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                # mne.viz.plot_connectivity_circle(mat_dfc_clean[respi_phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(mat_dfc_clean[respi_phase][mat_type][band], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
        plt.suptitle(f'{cond}_{mat_type} : THRESH', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_PHASE_thresh_{cond}_{mat_type}.png')
        plt.close('all')





#mat = mat_allplot_reduced
def save_fig_dfc_respi_phase_diff(cond, mat, ROI_list):

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_dfc[0]) != -1)]
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_dfc)

    respi_phase_list = ['inspi', 'expi']
    n_rows = 1

    print(f'######## PLOT {cond} DIFF ########')

    #### compute diff
    mat_diff = {}
            
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        mat_diff[mat_type] = {}

        for band in band_name_dfc:

            mat_diff[mat_type][band] = mat['inspi'][mat_type][band] - mat['expi'][mat_type][band]

    mat_original = mat.copy()
    mat = mat_diff

    #### identify scales
    scales = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

        for band in band_name_dfc:

            # mat_scaled = mat[band][mat_type_i,:,:][allband_data[respi_phase_i][band][mat_type_i,:,:] != 0]
            mat_scaled = mat[mat_type][band]

            scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
            scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

        scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

    #### identify scales abs
    scales_abs = {}

    for respi_phase in respi_phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_dfc:

                max_list = np.array(())
                max_list = np.append(max_list, mat_original[respi_phase][mat_type][band].max())
                max_list = np.append(max_list, np.abs(mat_original[respi_phase][mat_type][band].min()))

                scales_abs[mat_type][band] = max_list.max()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', cond, 'DFC', 'ROI'))
    roi_names = ROI_list
    mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #### mat plot
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_band, figsize=(15,15))
        plt.suptitle(f'{mat_type} inspi - expi')
        for c, band in enumerate(band_name_dfc):
            ax = axs[c]
            ax.set_title(band)
            ax.matshow(mat[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
            # ax.matshow(mat[mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
            # ax.matshow(allband_data[band][mat_type])
            if c == 0:
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_DIFF_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        nrows, ncols = n_rows, n_band
        fig = plt.figure()
        for c, band in enumerate(band_name_dfc):
            # mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
            #                                 textcolor='k')
            # mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 colormap=mat_type_color[mat_type], facecolor='w', 
            #                                 textcolor='k')
            mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
        plt.suptitle(f'{cond}_{mat_type} inspi - expi', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_DIFF_{cond}_{mat_type}.png')
        plt.close('all')

    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = mat.copy()

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for band in band_name_dfc:

            thresh_up = np.percentile(mat[mat_type][band].reshape(-1), percentile_thresh_up)
            thresh_down = np.percentile(mat[mat_type][band].reshape(-1), percentile_thresh_down)

            for x in range(mat_dfc_clean[mat_type][band].shape[1]):
                for y in range(mat_dfc_clean[mat_type][band].shape[1]):
                    if (mat_dfc_clean[mat_type][band][x,y] < thresh_up) & (mat_dfc_clean[mat_type][band][x,y] > thresh_down):
                            mat_dfc_clean[mat_type][band][x,y] = 0

    #### plot
    #mat_type = 'wpli'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #### mat plot
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_band, figsize=(15,15))
        plt.suptitle(f'{mat_type} THRESH inspi - expi')

        for c, band in enumerate(band_name_dfc):
            ax = axs[c]
            if c == 0:
                ax.yaxis.set_label_coords(-.5, .5)
            ax.set_title(band)
            ax.matshow(mat_dfc_clean[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
            # ax.matshow(mat_dfc_clean[mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
            # ax.matshow(mat_dfc_clean[mat_type][band])
            if c == 0:
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_DIFF_thresh_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        nrows, ncols = n_rows, n_band
        fig = plt.figure()
        for c, band in enumerate(band_name_dfc):
            # mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
            #                                 textcolor='k')
            # mne.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band], node_names=roi_names, n_lines=None, 
            #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
            #                                 colormap=mat_type_color[mat_type], facecolor='w', 
            #                                 textcolor='k')
            mne.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band], node_names=roi_names, n_lines=None, 
                                                    title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')
        plt.suptitle(f'{cond}_{mat_type} THRESH inspi - expi', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_DIFF_thresh_{cond}_{mat_type}.png')
        plt.close('all')






################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    cond = 'FR_CV'

    mat, ROI_list = precompute_dfc_mat_allplot(cond)
    save_fig_dfc(cond, mat, ROI_list)

    mat, ROI_list = precompute_dfc_mat_allplot_respi_phase(cond)
    save_fig_dfc_respi_phase(cond, mat, ROI_list)
        
    save_fig_dfc_respi_phase_diff(cond, mat, ROI_list)








