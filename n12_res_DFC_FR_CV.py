
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
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






def get_data_for_respi_phase(sujet):

    #### get ROI list
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_dfc[0]) != -1)]
    roi_in_data = xr.open_dataarray(file_to_load[0])['x'].data

    #### load data 
    allband_data = {}

    #export_type = 'inspi'
    for export_type in ['inspi', 'expi']:

        allband_data[export_type] = {}

        #band = 'beta'
        for band in band_name_dfc:

            file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1)]
            data = xr.open_dataarray(file_to_load[0])

            if export_type == 'inspi':
                data_chunk = data[:, :, :int(stretch_point_TF*ratio_stretch_TF)]
            elif export_type == 'expi':
                data_chunk = data[:, :, int(stretch_point_TF*ratio_stretch_TF):]

            mat_cf = np.zeros(( data['mat_type'].shape[0], roi_in_data.shape[0], roi_in_data.shape[0] ))

            for cf_metric_i, cf_metric in enumerate(data['mat_type'].data):
                mat_cf[cf_metric_i,:,:] = from_dfc_to_mat_conn_trpz(data_chunk.loc[cf_metric,:,:].data, data['pairs'].data, roi_in_data)

            allband_data[export_type][band] = mat_cf

    return allband_data








################################
######## SAVE FIG ########
################################


def process_dfc_res(sujet, cond, export_type):

    print(f'######## {cond} DFC {export_type} ########')

    #### CONNECTIVITY PLOT ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_dfc[0]) != -1)]
    roi_names = xr.open_dataarray(file_to_load[0])['x'].data
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_dfc)

    if export_type == 'whole':
            
        #### load data 
        allband_data = {}

        for band in band_name_dfc:

            file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band) != -1)]
            allband_data[band] = xr.open_dataarray(file_to_load[0]).data

        #### go to results
        os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', cond))

        #### identify scales
        scales = {}
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

            for band in band_name_dfc:

                mat_zero_excluded = allband_data[band][mat_type_i,:,:][allband_data[band][mat_type_i,:,:] != 0]

                scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_zero_excluded.min())
                scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_zero_excluded.max())

            scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].mean(), scales[mat_type]['vmax'].mean()
            
        #### identify scales abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_dfc:

                max_list = np.array(())

                max_list = np.append(max_list, allband_data[band][mat_type_i,:,:].max())
                max_list = np.append(max_list, np.abs(allband_data[band][mat_type_i,:,:].min()))

                scales_abs[mat_type][band] = max_list.max()

        #### plot
        mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #### mat plot
            fig, axs = plt.subplots(ncols=n_band, figsize=(15,15))
            plt.suptitle(mat_type)
            for c, band in enumerate(band_name_dfc):
                ax = axs[c]
                ax.set_title(band)
                ax.matshow(allband_data[band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(mat_dfc_clean[band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(allband_data[band][mat_type_i,:,:])
                if c == 0:
                    ax.set_yticks(np.arange(roi_names.shape[0]))
                    ax.set_yticklabels(roi_names)
            # plt.show()
            fig.savefig(f'MAT_{sujet}_{cond}_{mat_type}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = 1, n_band
            fig = plt.figure()
            for c, band in enumerate(band_name_dfc):
                # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
                # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
            plt.suptitle(f'{cond}_{mat_type}', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'CIRCLE_{cond}_{mat_type}.png')
            plt.close('all')


        #### thresh on previous plot
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean = allband_data.copy()

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_name_dfc:

                thresh_up = np.percentile(allband_data[band][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(allband_data[band][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[band][mat_type_i,:,:].shape[1]):
                    for y in range(mat_dfc_clean[band][mat_type_i,:,:].shape[1]):
                        if mat_type_i == 0:
                            if mat_dfc_clean[band][mat_type_i,x,y] < thresh_up:
                                mat_dfc_clean[band][mat_type_i,x,y] = 0
                        else:
                            if (mat_dfc_clean[band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][mat_type_i,x,y] > thresh_down):
                                mat_dfc_clean[band][mat_type_i,x,y] = 0


        #### plot
        #mat_type = 'wpli'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #### mat plot
            fig, axs = plt.subplots(ncols=n_band, figsize=(15,15))
            plt.suptitle(f'{mat_type} : THRESH')
            for c, band in enumerate(band_name_dfc):
                ax = axs[c]
                ax.set_title(band)
                ax.matshow(allband_data[band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(mat_dfc_clean[band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(mat_dfc_clean[band][mat_type_i,:,:])
                if c == 0:
                    ax.set_yticks(np.arange(roi_names.shape[0]))
                    ax.set_yticklabels(roi_names)
            # plt.show()
            fig.savefig(f'MAT_thresh_{sujet}_{cond}_{mat_type}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = 1, n_band
            fig = plt.figure()
            for c, band in enumerate(band_name_dfc):
                # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                # mne.viz.plot_connectivity_circle(mat_dfc_clean[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
            plt.suptitle(f'{cond}_{mat_type} : THRESH', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'CIRCLE_thresh_{cond}_{mat_type}.png')
            plt.close('all')

    elif export_type == 'respi_phase':

        #### load data 
        allband_data = get_data_for_respi_phase(sujet)
        respi_phase_list = ['inspi', 'expi']
        n_rows = len(respi_phase_list)

        #### go to results
        os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', cond))

        #### plot
        mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

        #### identify scales
        scales = {}

        for respi_phase in respi_phase_list:

            for mat_type_i, mat_type in enumerate(cf_metrics_list):

                scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

                for band in band_name_dfc:

                    # mat_scaled = allband_data[respi_phase][band][mat_type_i,:,:][allband_data[respi_phase][band][mat_type_i,:,:] != 0]
                    mat_scaled = allband_data[respi_phase][band][mat_type_i,:,:]

                    scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
                    scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

                scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

        #### identify scales abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_dfc:

                max_list = np.array(())

                for respi_phase in respi_phase_list:

                    max_list = np.append(max_list, allband_data[respi_phase][band][mat_type_i,:,:].max())
                    max_list = np.append(max_list, np.abs(allband_data[respi_phase][band][mat_type_i,:,:].min()))

                scales_abs[mat_type][band] = max_list.max()

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
                    ax.matshow(allband_data[respi_phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    # ax.matshow(allband_data[respi_phase][band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                    # ax.matshow(allband_data[respi_phase][band][mat_type_i,:,:])
                    if c == 0:
                        ax.set_yticks(np.arange(roi_names.shape[0]))
                        ax.set_yticklabels(roi_names)
            # plt.show()
            fig.savefig(f'MAT_PHASE_{sujet}_{cond}_{mat_type}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_rows, n_band
            fig = plt.figure()
            for r, respi_phase in enumerate(respi_phase_list):
                if r == 1:
                    r = len(band_name_dfc)
                for c, band in enumerate(band_name_dfc):
                    # mne.viz.plot_connectivity_circle(allband_data[respi_phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    # mne.viz.plot_connectivity_circle(allband_data[respi_phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    mne.viz.plot_connectivity_circle(allband_data[respi_phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
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

        mat_dfc_clean = allband_data.copy()

        for respi_phase in respi_phase_list:

            for mat_type_i, mat_type in enumerate(cf_metrics_list):

                for band in band_name_dfc:

                    thresh_up = np.percentile(allband_data[respi_phase][band][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                    thresh_down = np.percentile(allband_data[respi_phase][band][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                    for x in range(mat_dfc_clean[respi_phase][band][mat_type_i,:,:].shape[1]):
                        for y in range(mat_dfc_clean[respi_phase][band][mat_type_i,:,:].shape[1]):
                            if mat_type_i == 0:
                                if mat_dfc_clean[respi_phase][band][mat_type_i,x,y] < thresh_up:
                                    mat_dfc_clean[respi_phase][band][mat_type_i,x,y] = 0
                            else:
                                if (mat_dfc_clean[respi_phase][band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[respi_phase][band][mat_type_i,x,y] > thresh_down):
                                    mat_dfc_clean[respi_phase][band][mat_type_i,x,y] = 0

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
                    ax.set_title(band)
                    ax.matshow(mat_dfc_clean[respi_phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    # ax.matshow(mat_dfc_clean[respi_phase][band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                    # ax.matshow(mat_dfc_clean[respi_phase][band][mat_type_i,:,:])
                    if c == 0:
                        ax.set_yticks(np.arange(roi_names.shape[0]))
                        ax.set_yticklabels(roi_names)
            # plt.show()
            fig.savefig(f'MAT_PHASE_thresh_{sujet}_{cond}_{mat_type}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_rows, n_band
            fig = plt.figure()
            for r, respi_phase in enumerate(respi_phase_list):
                if r == 1:
                    r = len(band_name_dfc)
                for c, band in enumerate(band_name_dfc):
                    # mne.viz.plot_connectivity_circle(mat_dfc_clean[respi_phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=f'{band} : {respi_phase}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    # mne.viz.plot_connectivity_circle(mat_dfc_clean[respi_phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=f'{band} : {respi_phase}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    mne.viz.plot_connectivity_circle(mat_dfc_clean[respi_phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
            plt.suptitle(f'{cond}_{mat_type} : THRESH', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'CIRCLE_PHASE_thresh_{cond}_{mat_type}.png')
            plt.close('all')

    elif export_type == 'respi_phase_diff':

        #### load data 
        allband_data = get_data_for_respi_phase(sujet)
        allband_data_original = allband_data.copy()
        respi_phase_list = ['inspi', 'expi']
        n_rows = 1

        #### substract data
        allband_data_diff = {}
            
        for band in band_name_dfc:

            allband_data_diff[band] = allband_data['inspi'][band] - allband_data['expi'][band]

        allband_data = allband_data_diff

        #### go to results
        os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', cond))

        #### plot
        mat_type_color = {'ispc' : cm.seismic, 'wpli' : cm.seismic}

        #### identify scales on data that have been substracted
        scales = {}

        for respi_phase in respi_phase_list:

            for mat_type_i, mat_type in enumerate(cf_metrics_list):

                scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

                for band in band_name_dfc:

                    # mat_scaled = allband_data_original[respi_phase][band][mat_type_i,:,:][allband_data_original[respi_phase][band][mat_type_i,:,:] != 0]
                    mat_scaled = allband_data_original[respi_phase][band][mat_type_i,:,:]

                    scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
                    scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

                scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

        #### identify scales abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_dfc:

                max_list = np.array(())

                for respi_phase in respi_phase_list:

                    max_list = np.append(max_list, allband_data_original[respi_phase][band][mat_type_i,:,:].max())
                    max_list = np.append(max_list, np.abs(allband_data_original[respi_phase][band][mat_type_i,:,:].min()))

                scales_abs[mat_type][band] = max_list.max()

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #### mat plot
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_band, figsize=(15,15))
            plt.suptitle(f'{mat_type} inspi-expi')
            for c, band in enumerate(band_name_dfc):
                ax = axs[c]
                ax.set_title(band)
                ax.matshow(allband_data[band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(allband_data[band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(allband_data[band][mat_type_i,:,:])
                if c == 0:
                    ax.set_yticks(np.arange(roi_names.shape[0]))
                    ax.set_yticklabels(roi_names)
            # plt.show()
            fig.savefig(f'MAT_DIFF_{sujet}_{cond}_{mat_type}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_rows, n_band
            fig = plt.figure()
            for c, band in enumerate(band_name_dfc):
                # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')
            plt.suptitle(f'{cond}_{mat_type}_inspi-expi', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'CIRCLE_DIFF_{cond}_{mat_type}.png')
            plt.close('all')


        #### thresh on previous plot
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean = allband_data_diff.copy()

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_name_dfc:

                thresh_up = np.percentile(allband_data_diff[band][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(allband_data_diff[band][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[band][mat_type_i,:,:].shape[1]):
                    for y in range(mat_dfc_clean[band][mat_type_i,:,:].shape[1]):
                        if (mat_dfc_clean[band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][mat_type_i,x,y] > thresh_down):
                            mat_dfc_clean[band][mat_type_i,x,y] = 0

        #### plot
        #mat_type = 'wpli'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #### mat plot
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_band, figsize=(15,15))
            plt.suptitle(f'{mat_type} : THRESH inspi-expi')

            for c, band in enumerate(band_name_dfc):
                ax = axs[c]
                ax.set_title(f'{band} inspi-expi')
                ax.matshow(mat_dfc_clean[band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(mat_dfc_clean[band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(mat_dfc_clean[band][mat_type_i,:,:])
                if c == 0:
                    ax.set_yticks(np.arange(roi_names.shape[0]))
                    ax.set_yticklabels(roi_names)
            
            # plt.show()
            fig.savefig(f'MAT_DIFF_thresh_{sujet}_{cond}_{mat_type}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_rows, n_band
            fig = plt.figure()
            for c, band in enumerate(band_name_dfc):
                # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                # mne.viz.plot_connectivity_circle(mat_dfc_clean[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(mat_dfc_clean[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                    title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')
            plt.suptitle(f'{cond}_{mat_type} THRESH inspi-expi', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'CIRCLE_DIFF_thresh_{cond}_{mat_type}.png')
            plt.close('all')






################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    for sujet in sujet_list_FR_CV:

        print(sujet)

        cond = 'FR_CV'

        for export_type in ['whole', 'respi_phase', 'respi_phase_diff']:
            process_dfc_res(sujet, cond, export_type)
        