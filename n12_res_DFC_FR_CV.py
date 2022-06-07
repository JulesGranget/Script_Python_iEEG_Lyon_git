

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne

from matplotlib import cm

import pandas as pd
import joblib
import xarray as xr


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





################################
######## SAVE FIG ########
################################


def process_dfc_res(sujet, cond):


    print(f'######## {cond} DFC ########')

    #### CONNECTIVITY PLOT ####
    #### load params
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    df_loca = get_loca_df(sujet)

    #### load data 
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    allband_data = {}
    for pair_type in ['allpairs', 'reducedpairs']:
        
        allband_data[pair_type] = {}
        
        for band in band_name_dfc:
            if band == 'beta':
                continue
            file_to_load = [i for i in os.listdir() if ( i.find(pair_type) != -1 and i.find(band) != -1)]
            allband_data[pair_type][band] = xr.open_dataarray(file_to_load[0])

    #### go to results
    os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', cond))

    #### identify scales
    scales = {}
    for mat_type in ['ispc', 'pli']:

        scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

        for band in band_name_dfc:
            if band == 'beta':
                continue
            scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], allband_data['reducedpairs'][band].loc[mat_type,:,:].min())
            scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], allband_data['reducedpairs'][band].loc[mat_type,:,:].max())

        scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].mean(), scales[mat_type]['vmax'].mean()

    #### plot
    roi_names = allband_data['l_gamma']['x'].values

    for mat_type in ['ispc', 'pli']:

        #### mat plot
        # fig, axs = plt.subplots(ncols=3, figsize=(15,15))
        fig, axs = plt.subplots(ncols=2, figsize=(15,15))
        plt.suptitle(mat_type)
        for c, band in enumerate(band_name_dfc[1:]):
            ax = axs[c]
            ax.set_title(band)
            ax.matshow(allband_data['reducedpairs'][band].loc[mat_type,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
            if c == 0:
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_reduced_{sujet}_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        # nrows, ncols = 1, 3
        nrows, ncols = 1, 2
        fig = plt.figure()
        for c, band in enumerate(band_name_dfc[1:]):
            mne.viz.plot_connectivity_circle(allband_data['reducedpairs'][band].loc[mat_type,:,:].values, node_names=roi_names, n_lines=None, 
                                            title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                            vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')
        plt.suptitle(f'{cond}_{mat_type}', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_reduced_{cond}_{mat_type}.png')
        plt.close('all')


    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = allband_data['reducedpairs'].copy()

    for mat_type_i, mat_type in enumerate(['ispc', 'pli']):

        for band in band_name_dfc[1:]:

            thresh_up = np.percentile(allband_data['reducedpairs'][band].loc[mat_type,:,:].values.reshape(-1), percentile_thresh_up)
            thresh_down = np.percentile(allband_data['reducedpairs'][band].loc[mat_type,:,:].values.reshape(-1), percentile_thresh_down)

            for x in range(mat_dfc_clean[band].loc[mat_type,:,:].values.shape[1]):
                for y in range(mat_dfc_clean[band].loc[mat_type,:,:].values.shape[1]):
                    if (mat_dfc_clean[band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][mat_type_i,x,y] > thresh_down):
                        mat_dfc_clean[band][mat_type_i,x,y] = 0
                    if (mat_dfc_clean[band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][mat_type_i,x,y] > thresh_down):
                        mat_dfc_clean[band][mat_type_i,x,y] = 0

    #### plot
    for mat_type in ['ispc', 'pli']:

        #### mat plot
        # fig, axs = plt.subplots(ncols=3, figsize=(15,15))
        fig, axs = plt.subplots(ncols=2, figsize=(15,15))
        plt.suptitle(mat_type)
        for c, band in enumerate(band_name_dfc[1:]):
            ax = axs[c]
            ax.set_title(band)
            ax.matshow(mat_dfc_clean[band].loc[mat_type,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
            if c == 0:
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'MAT_thresh_{sujet}_{cond}_{mat_type}.png')
        plt.close('all')

        #### circle plot
        # nrows, ncols = 1, 3
        nrows, ncols = 1, 2
        fig = plt.figure()
        for c, band in enumerate(band_name_dfc[1:]):
            mne.viz.plot_connectivity_circle(mat_dfc_clean[band].loc[mat_type,:,:].values, node_names=roi_names, n_lines=None, 
                                            title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                                            vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')
        plt.suptitle(f'{cond}_{mat_type}', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'CIRCLE_thresh_{cond}_{mat_type}.png')
        plt.close('all')



################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    cond = 'FR_CV'
    
    process_dfc_res(sujet, cond)
        









