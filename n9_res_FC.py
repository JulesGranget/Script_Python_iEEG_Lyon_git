
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
import xarray as xr


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





################################
######## SAVE FIG ########
################################


def process_fc_res(sujet):

    print(f'######## FC ########')

    #### CONNECTIVITY PLOT ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    prms = get_params(sujet)
    file_to_load = [i for i in os.listdir() if i.find('reducedpairs') != -1]
    roi_names = xr.open_dataarray(file_to_load[0])['x'].data
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
            
    #### load data 
    allband_data = {}
    #band_prep = 'lf'
    for band_prep in band_prep_list:
        #band = 'theta'
        for band in freq_band_dict_FC_function[band_prep]:

            allband_data[band] = {}
            
            for cond in prms['conditions']:

                file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band) != -1 and i.find(cond) != -1)]
                allband_data[band][cond] = xr.open_dataarray(file_to_load[0]).data 

    #### go to results
    os.chdir(os.path.join(path_results, sujet, 'FC'))

    #### identify scales
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band_prep in band_prep_list:
            #band = 'theta'
            for band in freq_band_dict_FC_function[band_prep]:

                max_list = np.array(())

                max_list = np.append(max_list, np.abs(allband_data[band][cond][mat_type_i,:,:].min()))
                max_list = np.append(max_list, allband_data[band][cond][mat_type_i,:,:].max())

                scales_abs[mat_type][band] = max_list.max()

    #### plot
    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):
        #band_prep = 'lf'
        for band_prep in band_prep_list:

            n_band = len(freq_band_dict_FC_function[band_prep])
            n_cond = len(prms['conditions'])

            #### mat plot
            fig, axs = plt.subplots(nrows=n_band ,ncols=n_cond, figsize=(15,15))
            plt.suptitle(mat_type)
            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                for c, cond in enumerate(prms['conditions']):

                    ax = axs[r, c]
                    ax.set_title(f'{band}_{cond}')
                    ax.matshow(allband_data[band][cond][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                    if c == 0:
                        ax.set_yticks(np.arange(roi_names.shape[0]))
                        ax.set_yticklabels(roi_names)

            # plt.show()
            fig.savefig(f'MAT_{sujet}_{mat_type}_{band_prep}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_band, n_cond

            fig = plt.figure()
            _position = 0

            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                for c, cond in enumerate(prms['conditions']):

                    _position += 1

                    mne.viz.plot_connectivity_circle(allband_data[band][cond][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                    title=f'{cond}_{band}', show=False, padding=7, fig=fig, subplot=(n_band, n_cond, _position),
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')

            plt.suptitle(f'{mat_type}', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'CIRCLE_{sujet}_{mat_type}_{band_prep}.png')
            plt.close('all')


    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = allband_data.copy()

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for band_prep in band_prep_list:
            #band = 'theta'
            for band in freq_band_dict_FC_function[band_prep]:

                for cond in prms['conditions']:

                    thresh_up = np.percentile(allband_data[band][cond][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                    thresh_down = np.percentile(allband_data[band][cond][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                    for x in range(mat_dfc_clean[band][cond][mat_type_i,:,:].shape[1]):
                        for y in range(mat_dfc_clean[band][cond][mat_type_i,:,:].shape[1]):
                            if mat_type_i == 0:
                                if mat_dfc_clean[band][cond][mat_type_i,x,y] < thresh_up:
                                    mat_dfc_clean[band][cond][mat_type_i,x,y] = 0
                            else:
                                if (mat_dfc_clean[band][cond][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][cond][mat_type_i,x,y] > thresh_down):
                                    mat_dfc_clean[band][cond][mat_type_i,x,y] = 0

    #### plot
    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):
        #band_prep = 'lf'
        for band_prep in band_prep_list:

            n_band = len(freq_band_dict_FC_function[band_prep])
            n_cond = len(prms['conditions'])

            #### mat plot
            fig, axs = plt.subplots(nrows=n_band ,ncols=n_cond, figsize=(15,15))
            plt.suptitle(f'{mat_type} : THRESH')
            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                for c, cond in enumerate(prms['conditions']):

                    ax = axs[r, c]
                    ax.set_title(f'{band}_{cond}')
                    ax.matshow(mat_dfc_clean[band][cond][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                    if c == 0:
                        ax.set_yticks(np.arange(roi_names.shape[0]))
                        ax.set_yticklabels(roi_names)

            # plt.show()
            fig.savefig(f'MAT_thresh_{sujet}_{mat_type}_{band_prep}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = n_band, n_cond

            fig = plt.figure()
            _position = 0

            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                for c, cond in enumerate(prms['conditions']):

                    _position += 1

                    mne.viz.plot_connectivity_circle(mat_dfc_clean[band][cond][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                    title=f'{cond}_{band}', show=False, padding=7, fig=fig, subplot=(n_band, n_cond, _position),
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')

            plt.suptitle(f'{mat_type} THRESH', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'CIRCLE_thresh_{sujet}_{mat_type}_{band_prep}.png')
            plt.close('all')



################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    for sujet in sujet_list_FR_CV:

        print(sujet)
        process_fc_res(sujet)
        