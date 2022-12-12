

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import copy
import mne_connectivity

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






################################################
######## COMPUTE DATA RESPI PHASE ########
################################################


#data_dfc, pairs, roi_in_data = data.loc[AL_i+1, cf_metric, :, select_time_vec], pairs, roi_in_data
def from_dfc_to_mat_conn_mean(data_dfc, pairs, roi_in_data):

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





def precompute_dfc_mat_AL_allplot(cond, monopol):

    #### initiate containers
    mat_pairs_allplot = {}

    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i, monopol)
        if cond in prms_i['conditions']:
            if cond == 'FR_CV':
                continue
            sujet_list_selected.append(sujet_i)

    #AL_i = 0
    for AL_i in range(3):
        mat_pairs_allplot[f'AL_{AL_i+1}'] = {}
        #cf_metric = 'ispc'
        for cf_metric in ['ispc', 'wpli']:
            mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric] = {}
            #band_prep = 'lf'
            for band_prep in band_prep_list:
                #band, freq = 'beta', [10,40]
                for band, freq in freq_band_dict_FC_function[band_prep].items():

                    if band in ['beta', 'l_gamma', 'h_gamma']:

                        print(AL_i, cf_metric, band)

                        #sujet = sujet_list_selected[0]
                        for sujet_i, sujet in enumerate(sujet_list_selected):

                            #### extract data
                            os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

                            if monopol:
                                xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                            else:
                                xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')
                            
                            #### concat pairs name
                            if sujet_i == 0:
                                pairs_allplot = xr_dfc['pairs'].data
                            else:
                                pairs_allplot = np.concatenate((pairs_allplot, xr_dfc['pairs'].data), axis=0)

                            #### extract data and concat
                            mat = xr_dfc.loc[AL_i+1, cf_metric,:,:,:].data

                            del xr_dfc

                            if sujet_i == 0:
                                mat_values_allplot = mat
                            else:
                                mat_values_allplot = np.concatenate((mat_values_allplot, mat), axis=0)

                            del mat

                            #### mean
                            pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pairs_allplot)
        
                            dfc_mean_pair = np.zeros(( pair_unique_allplot.shape[0], mat_values_allplot.shape[1], mat_values_allplot.shape[-1] ))

                            #x_i, x_name = 0, roi_in_data[0]
                            for x_i, x_name in enumerate(roi_in_data_allplot):
                                #y_i, y_name = 2, roi_in_data[2]
                                for y_i, y_name in enumerate(roi_in_data_allplot):
                                    if x_name == y_name:
                                        continue

                                    pair_to_find = f'{x_name}-{y_name}'
                                    pair_to_find_rev = f'{y_name}-{x_name}'
                                    
                                    x = mat_values_allplot[pairs_allplot == pair_to_find, :, :]
                                    x_rev = mat_values_allplot[pairs_allplot == pair_to_find_rev, :, :]

                                    x_mean = np.vstack([x, x_rev]).mean(axis=0)

                                    if np.isnan(x_mean).sum() > 1:
                                        continue

                                    #### identify pair name mean
                                    try:
                                        pair_position = np.where(pair_unique_allplot == pair_to_find)[0][0]
                                    except:
                                        pair_position = np.where(pair_unique_allplot == pair_to_find_rev)[0][0]

                                    dfc_mean_pair[pair_position, :, :] = x_mean

                            mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric][band] = dfc_mean_pair

                        del mat_values_allplot
                        
    #### identify missing pairs with clean ROI
    for ROI_to_clean in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel']:
        if ROI_to_clean in roi_in_data_allplot:
            roi_in_data_allplot = np.delete(roi_in_data_allplot, roi_in_data_allplot==ROI_to_clean)

    mat_verif = np.zeros(( roi_in_data_allplot.shape[0], roi_in_data_allplot.shape[0]))
    for ROI_row_i, ROI_row_name in enumerate(roi_in_data_allplot):
        #ROI_col_i, ROI_col_name = 1, ROI_list[1]
        for ROI_col_i, ROI_col_name in enumerate(roi_in_data_allplot):

            if ROI_row_name == ROI_col_name:
                mat_verif[ROI_row_i, ROI_col_i] = 1
                continue

            pair_name = f'{ROI_row_name}-{ROI_col_name}'
            pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

            if pair_name in pair_unique_allplot:
                mat_verif[ROI_row_i, ROI_col_i] = 1
            if pair_name_rev in pair_unique_allplot:
                mat_verif[ROI_row_i, ROI_col_i] = 1

    if debug:
        plt.matshow(mat_verif)
        plt.show()

    #### export missig matrix
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'allcond'))
    plt.matshow(mat_verif)
    plt.savefig('missing_mat.png')
    plt.close('all')

    #### clean pairs
    pairs_to_keep = []
    for pair_i, pair in enumerate(pair_unique_allplot):
        if pair.split('-')[0] not in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel'] and pair.split('-')[1] not in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel']:
            pairs_to_keep.append(pair_i)

    for AL_i in range(3):
        for cf_metric in ['ispc', 'wpli']:
            for band in mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric].keys():
                mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric][band] = mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric][band][pairs_to_keep, :, :].copy()

    pair_unique_allplot = pair_unique_allplot[pairs_to_keep]

    #### save
    os.chdir(os.path.join(path_precompute, 'allplot'))
    data_dfc_allpairs = np.zeros((3, 2, len(band_name_fc_dfc), pair_unique_allplot.shape[0], mat_pairs_allplot['AL_1']['ispc']['beta'].shape[1], mat_pairs_allplot['AL_1']['ispc']['beta'].shape[2]))
    
    for AL_i in range(3):
        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            for band_i, band in enumerate(mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric].keys()):
                data_dfc_allpairs[AL_i, cf_metric_i, band_i, :, :, :] = mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric][band]

    dims = ['AL_num', 'cf_metric', 'band', 'pairs', 'nfrex', 'time']
    coords = [range(3), ['ispc', 'wpli'], band_name_fc_dfc, pair_unique_allplot, range(nfrex_lf), range(mat_pairs_allplot['AL_1']['ispc']['beta'].shape[2])]
    xr_dfc_allpairs = xr.DataArray(data_dfc_allpairs, coords=coords, dims=dims)

    if monopol:
        xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs.nc')
    else:
        xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs_bi.nc')

    return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot







def precompute_dfc_mat_allplot(band_to_compute, cond, monopol):

    os.chdir(os.path.join(path_precompute, 'allplot'))

    if monopol and os.path.exists(f'allcond_dfc_{cond}_allpairs.nc'):

        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

        return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot

    if monopol == False and os.path.exists(f'allcond_dfc_{cond}_allpairs_bi.nc'):
        
        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs_bi.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

        return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot

    else:

        #### initiate containers
        mat_pairs_allplot = {}

        #### filter only sujet with correct cond
        sujet_list_selected = []
        for sujet_i in sujet_list:
            prms_i = get_params(sujet_i, monopol)
            if cond in prms_i['conditions']:
                if cond == 'FR_CV':
                    continue
                sujet_list_selected.append(sujet_i)

        #cf_metric = 'ispc'
        for cf_metric in ['ispc', 'wpli']:

            mat_pairs_allplot[cf_metric] = {}

            #band_prep = 'lf'
            for band in band_to_compute:

                print(cf_metric, band)

                #sujet = sujet_list_selected[0]
                for sujet_i, sujet in enumerate(sujet_list_selected):

                    #### extract data
                    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

                    if monopol:
                        xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs.nc')
                    else:
                        xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')
                    
                    #### concat pairs name
                    if sujet_i == 0:
                        pairs_allplot = xr_dfc['pairs'].data
                    else:
                        pairs_allplot = np.concatenate((pairs_allplot, xr_dfc['pairs'].data), axis=0)

                    #### extract data and concat
                    mat = xr_dfc.loc[cf_metric,:,:,:].data

                    del xr_dfc

                    if sujet_i == 0:
                        mat_values_allplot = mat
                    else:
                        mat_values_allplot = np.concatenate((mat_values_allplot, mat), axis=0)

                    del mat

                    #### mean
                    pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pairs_allplot)

                    dfc_mean_pair = np.zeros(( pair_unique_allplot.shape[0], mat_values_allplot.shape[1], mat_values_allplot.shape[-1] ))

                    #x_i, x_name = 0, roi_in_data[0]
                    for x_i, x_name in enumerate(roi_in_data_allplot):
                        #y_i, y_name = 2, roi_in_data[2]
                        for y_i, y_name in enumerate(roi_in_data_allplot):
                            if x_name == y_name:
                                continue

                            pair_to_find = f'{x_name}-{y_name}'
                            pair_to_find_rev = f'{y_name}-{x_name}'
                            
                            x = mat_values_allplot[pairs_allplot == pair_to_find, :, :]
                            x_rev = mat_values_allplot[pairs_allplot == pair_to_find_rev, :, :]

                            x_mean = np.vstack([x, x_rev]).mean(axis=0)

                            if np.isnan(x_mean).sum() > 1:
                                continue

                            #### identify pair name mean
                            try:
                                pair_position = np.where(pair_unique_allplot == pair_to_find)[0][0]
                            except:
                                pair_position = np.where(pair_unique_allplot == pair_to_find_rev)[0][0]

                            dfc_mean_pair[pair_position, :, :] = x_mean

                    mat_pairs_allplot[cf_metric][band] = dfc_mean_pair

                del mat_values_allplot
                            
        #### identify missing pairs with clean ROI
        for ROI_to_clean in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel']:
            if ROI_to_clean in roi_in_data_allplot:
                roi_in_data_allplot = np.delete(roi_in_data_allplot, roi_in_data_allplot==ROI_to_clean)

        mat_verif = np.zeros(( roi_in_data_allplot.shape[0], roi_in_data_allplot.shape[0]))
        for ROI_row_i, ROI_row_name in enumerate(roi_in_data_allplot):
            #ROI_col_i, ROI_col_name = 1, ROI_list[1]
            for ROI_col_i, ROI_col_name in enumerate(roi_in_data_allplot):

                if ROI_row_name == ROI_col_name:
                    mat_verif[ROI_row_i, ROI_col_i] = 1
                    continue

                pair_name = f'{ROI_row_name}-{ROI_col_name}'
                pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                if pair_name in pair_unique_allplot:
                    mat_verif[ROI_row_i, ROI_col_i] = 1
                if pair_name_rev in pair_unique_allplot:
                    mat_verif[ROI_row_i, ROI_col_i] = 1

        if debug:
            plt.matshow(mat_verif)
            plt.show()

        #### export missig matrix
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'FC', 'allcond'))
        plt.matshow(mat_verif)
        plt.savefig('missing_mat.png')
        plt.close('all')

        #### clean pairs
        pairs_to_keep = []
        for pair_i, pair in enumerate(pair_unique_allplot):
            if pair.split('-')[0] not in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel'] and pair.split('-')[1] not in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel']:
                pairs_to_keep.append(pair_i)

        for cf_metric in ['ispc', 'wpli']:
            for band in mat_pairs_allplot[cf_metric].keys():
                mat_pairs_allplot[cf_metric][band] = mat_pairs_allplot[cf_metric][band][pairs_to_keep, :, :].copy()

        pair_unique_allplot = pair_unique_allplot[pairs_to_keep]

        #### save
        os.chdir(os.path.join(path_precompute, 'allplot'))
        data_dfc_allpairs = np.zeros((2, len(band_to_compute), pair_unique_allplot.shape[0], mat_pairs_allplot['ispc']['beta'].shape[1], mat_pairs_allplot['ispc']['beta'].shape[2]))
        
        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            for band_i, band in enumerate(mat_pairs_allplot[cf_metric].keys()):
                data_dfc_allpairs[cf_metric_i, band_i, :, :, :] = mat_pairs_allplot[cf_metric][band]

        dims = ['cf_metric', 'band', 'pairs', 'phase', 'nfrex']
        coords = [['ispc', 'wpli'], band_to_compute, pair_unique_allplot, ['whole', 'inspi', 'expi'], range(mat_pairs_allplot['ispc']['beta'].shape[2])]
        xr_dfc_allpairs = xr.DataArray(data_dfc_allpairs, coords=coords, dims=dims)

        if monopol:
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs.nc')
        else:
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs_bi.nc')

    return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot




           



#dfc_data, pairs = mat.loc[cf_metric, band, :, phase, :].data, pair_unique_allplot
def dfc_pairs_to_mat(dfc_data, pairs, compute_mode, rscore_computation):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)
    
    #### mean pairs to mat
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

            x_mean = np.vstack([x, x_rev]).mean(axis=0)

            if np.isnan(x_mean).sum() > 1:
                continue

            if rscore_computation:
                x_mean_rscore = rscore_mat(x_mean)
            else:
                x_mean_rscore = x_mean

            if compute_mode == 'mean':
                val_to_place = x_mean_rscore.mean()
            if compute_mode == 'trapz':
                val_to_place = np.trapz(x_mean_rscore)

            mat_dfc[x_i, y_i] = val_to_place

    return mat_dfc
    









#mat, pairs = xr_pairs_allplot, pair_unique_allplot
def precompute_dfc_mat_allplot_phase(mat, pairs, band_to_compute, baselines, monopol, rscore_computation=False):

    #### define diff and phase to plo
    phase_list = mat['phase'].data

    srate = get_params(sujet_list[0], monopol)['srate']

    #### initiate containers
    mat_phase = {}

    #cf_metric_i, cf_metric = 0, 'ispc'
    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

        mat_phase[cf_metric] = {}

        #band_prep = 'lf'
        for band in band_to_compute:

            mat_phase[cf_metric][band] = {}

            #phase = 'pre'
            for phase in phase_list:

                #### fill mat
                mat_phase[cf_metric][band][phase] = dfc_pairs_to_mat(mat.loc[cf_metric, band, :, phase, :].data, pairs, 'mean', rscore_computation) - baselines[cf_metric][band][phase]

    return mat_phase














########################################
######## FR_CV BASELINES ########
########################################



def precompute_baselines_allplot(band_to_compute, monopol, rscore_computation=False):

    cond = 'FR_CV'

    os.chdir(os.path.join(path_precompute, 'allplot'))

    if monopol and os.path.exists(f'allcond_dfc_FR_CV_allpairs.nc'):

        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

    if monopol == False and os.path.exists(f'allcond_dfc_FR_CV_allpairs_bi.nc'):

        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs_bi.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

    else:

        #### initiate containers
        mat_pairs_allplot = {}

        #### filter only sujet with correct cond
        sujet_list_selected = []
        for sujet_i in sujet_list:
            prms_i = get_params(sujet_i, monopol)
            if cond in prms_i['conditions']:
                sujet_list_selected.append(sujet_i)

        #cf_metric = 'ispc'
        for cf_metric in ['ispc', 'wpli']:

            mat_pairs_allplot[cf_metric] = {}

            for band in band_to_compute:

                print(cf_metric, band)

                #sujet = sujet_list_selected[0]
                for sujet_i, sujet in enumerate(sujet_list_selected):

                    #### extract data
                    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

                    if monopol:
                        xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs.nc')
                    else:
                        xr_dfc = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')
                    
                    #### concat pairs name
                    if sujet_i == 0:
                        pairs_allplot = xr_dfc['pairs'].data
                    else:
                        pairs_allplot = np.concatenate((pairs_allplot, xr_dfc['pairs'].data), axis=0)

                    #### extract data and concat
                    mat = xr_dfc.loc[cf_metric,:,:,:].data

                    del xr_dfc

                    if sujet_i == 0:
                        mat_values_allplot = mat
                    else:
                        mat_values_allplot = np.concatenate((mat_values_allplot, mat), axis=0)

                    del mat

                    #### mean
                    pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pairs_allplot)

                    dfc_mean_pair = np.zeros(( pair_unique_allplot.shape[0], mat_values_allplot.shape[1], mat_values_allplot.shape[-1] ))

                    #x_i, x_name = 0, roi_in_data[0]
                    for x_i, x_name in enumerate(roi_in_data_allplot):
                        #y_i, y_name = 2, roi_in_data[2]
                        for y_i, y_name in enumerate(roi_in_data_allplot):
                            if x_name == y_name:
                                continue

                            pair_to_find = f'{x_name}-{y_name}'
                            pair_to_find_rev = f'{y_name}-{x_name}'
                            
                            x = mat_values_allplot[pairs_allplot == pair_to_find, :, :]
                            x_rev = mat_values_allplot[pairs_allplot == pair_to_find_rev, :, :]

                            x_mean = np.vstack([x, x_rev]).mean(axis=0)

                            if np.isnan(x_mean).sum() > 1:
                                continue

                            #### identify pair name mean
                            try:
                                pair_position = np.where(pair_unique_allplot == pair_to_find)[0][0]
                            except:
                                pair_position = np.where(pair_unique_allplot == pair_to_find_rev)[0][0]

                            dfc_mean_pair[pair_position, :, :] = x_mean

                    mat_pairs_allplot[cf_metric][band] = dfc_mean_pair

                del mat_values_allplot
                            
        #### identify missing pairs with clean ROI
        for ROI_to_clean in ['WM', 'ventricule', 'choroide plexus', 'not in freesurfer parcel']:
            if ROI_to_clean in roi_in_data_allplot:
                roi_in_data_allplot = np.delete(roi_in_data_allplot, roi_in_data_allplot==ROI_to_clean)

        mat_verif = np.zeros(( roi_in_data_allplot.shape[0], roi_in_data_allplot.shape[0]))
        for ROI_row_i, ROI_row_name in enumerate(roi_in_data_allplot):
            #ROI_col_i, ROI_col_name = 1, ROI_list[1]
            for ROI_col_i, ROI_col_name in enumerate(roi_in_data_allplot):

                if ROI_row_name == ROI_col_name:
                    mat_verif[ROI_row_i, ROI_col_i] = 1
                    continue

                pair_name = f'{ROI_row_name}-{ROI_col_name}'
                pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                if pair_name in pair_unique_allplot:
                    mat_verif[ROI_row_i, ROI_col_i] = 1
                if pair_name_rev in pair_unique_allplot:
                    mat_verif[ROI_row_i, ROI_col_i] = 1

        if debug:
            plt.matshow(mat_verif)
            plt.show()

        #### clean pairs
        pairs_to_keep = []
        for pair_i, pair in enumerate(pair_unique_allplot):
            if pair.split('-')[0] not in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel'] and pair.split('-')[1] not in ['WM', 'ventricule', 'choroide plexus', 'not in a freesurfer parcel']:
                pairs_to_keep.append(pair_i)

        for cf_metric in ['ispc', 'wpli']:
            for band in mat_pairs_allplot[cf_metric].keys():
                mat_pairs_allplot[cf_metric][band] = mat_pairs_allplot[cf_metric][band][pairs_to_keep, :, :].copy()

        pair_unique_allplot = pair_unique_allplot[pairs_to_keep]

        #### save
        os.chdir(os.path.join(path_precompute, 'allplot'))
        data_dfc_allpairs = np.zeros((2, len(band_to_compute), pair_unique_allplot.shape[0], mat_pairs_allplot['ispc']['beta'].shape[1], mat_pairs_allplot['ispc']['beta'].shape[2]))
        
        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            for band_i, band in enumerate(mat_pairs_allplot[cf_metric].keys()):
                data_dfc_allpairs[cf_metric_i, band_i, :, :] = mat_pairs_allplot[cf_metric][band]

        dims = ['cf_metric', 'band', 'pairs', 'phase', 'nfrex']
        coords = [['ispc', 'wpli'], band_to_compute, pair_unique_allplot, ['whole', 'inspi', 'expi'], range(mat_pairs_allplot['ispc']['beta'].shape[2])]
        xr_dfc_allpairs = xr.DataArray(data_dfc_allpairs, coords=coords, dims=dims)

        if monopol:
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs.nc')
        else:
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs_bi.nc')

    #### compute dfc mat
    mat_baselines = {}

    #cf_metric_i, cf_metric = 0, 'ispc'
    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

        mat_baselines[cf_metric] = {}

        for band in band_to_compute:

            mat_baselines[cf_metric][band] = {}
            
            #phase = 'whole'
            for phase in ['whole', 'inspi', 'expi']:

                mat_baselines[cf_metric][band][phase] = dfc_pairs_to_mat(xr_dfc_allpairs.loc[cf_metric, band, :, phase, :].data, pair_unique_allplot, 'mean', rscore_computation)

    return mat_baselines










################################
######## SAVE FIG ########
################################




def save_fig_dfc_allplot(mat_phase, cond, band_to_compute, roi_in_data_allplot, monopol, FR_CV_normalized=True, plot_circle_dfc=False):

    #### get params
    cf_metrics_list = ['ispc', 'wpli']
    n_band = len(band_to_compute)

    #### define diff and phase to plot
    phase_list = ['whole', 'inspi', 'expi']

    n_cols_raw = len(phase_list)

    #### put 0 to matrix center
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for phase_list_i in phase_list:
                    
                for band in band_to_compute:

                    mat_to_clean = mat_phase[mat_type][band][phase_list_i]

                    for roi_i in range(mat_to_clean.shape[0]):

                        mat_to_clean[roi_i,roi_i] = 0

                    mat_phase[mat_type][band][phase_list_i] = mat_to_clean.copy()

    #### identify scales
    scales = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

        for band in band_to_compute:
            
            for phase in phase_list:

                mat_scaled = mat_phase[mat_type][band][phase]

                scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
                scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

        scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

    #### identify scales abs
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_to_compute:

            max_list = np.array(())

            for phase in phase_list:

                max_list = np.append(max_list, mat_phase[mat_type][band][phase].max())
                max_list = np.append(max_list, np.abs(mat_phase[mat_type][band][phase].min()))

            scales_abs[mat_type][band] = max_list.max()

    #### thresh alldata
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = copy.deepcopy(mat_phase)

    for phase in phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_to_compute:

                thresh_up = np.percentile(mat_phase[mat_type][band][phase].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(mat_phase[mat_type][band][phase].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[mat_type][band][phase].shape[1]):
                    for y in range(mat_dfc_clean[mat_type][band][phase].shape[1]):
                        if mat_type_i == 0:
                            if mat_dfc_clean[mat_type][band][phase][x,y] < thresh_up:
                                mat_dfc_clean[mat_type][band][phase][x,y] = 0
                        else:
                            if (mat_dfc_clean[mat_type][band][phase][x,y] < thresh_up) & (mat_dfc_clean[mat_type][band][phase][x,y] > thresh_down):
                                mat_dfc_clean[mat_type][band][phase][x,y] = 0

    ######## PLOT #######

    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'FC', 'allcond'))

    #### RAW

    n_cols_raw = len(phase_list)

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #band_prep = 'lf'
        for band_prep in band_prep_list:

            ######## NO THRESH ########

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
                    
                    cax = ax.matshow(mat_phase[mat_type][band][phase], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                    fig.colorbar(cax, ax=ax)

                    ax.set_yticks(np.arange(roi_in_data_allplot.shape[0]))
                    ax.set_yticklabels(roi_in_data_allplot)
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

                        mne_connectivity.viz.plot_connectivity_circle(mat_phase[mat_type][band][phase], node_names=roi_in_data_allplot, n_lines=None, 
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
                
                cax = ax.matshow(mat_dfc_clean[mat_type][band][phase], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                fig.colorbar(cax, ax=ax)

                ax.set_yticks(np.arange(roi_in_data_allplot.shape[0]))
                ax.set_yticklabels(roi_in_data_allplot)
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

                    mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band][phase], node_names=roi_in_data_allplot, n_lines=None, 
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








            
################################
######## SUMMARY ########
################################



def process_dfc_res_summary(mat_phase_allcond, roi_in_data_allplot, cond_to_compute, band_to_compute, monopol, plot_circle_dfc=False):

    print(f'######## SUMMARY DFC ########')

    #### CONNECTIVITY PLOT ####
    
    cf_metrics_list = ['ispc', 'wpli']

    #### load allcond data 
    allcond_data_thresh = {}
    allcond_scales_abs = {}

    for cond in cond_to_compute:

        #### load data
        allcond_data_i = mat_phase_allcond[cond]

        #### define diff and phase to plot
        phase_list = ['whole', 'inspi', 'expi']

        #### scale abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_to_compute:

                max_list = np.array(())

                for phase in phase_list:

                    max_list = np.append(max_list, allcond_data_i[mat_type][band][phase].max())
                    max_list = np.append(max_list, np.abs(allcond_data_i[mat_type][band][phase].min()))

                scales_abs[mat_type][band] = max_list.max()

        allcond_scales_abs[cond] = scales_abs

        #### thresh
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean_i = copy.deepcopy(allcond_data_i)

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for phase in phase_list:

                for band in band_to_compute:

                    thresh_up = np.percentile(allcond_data_i[mat_type][band][phase].reshape(-1), percentile_thresh_up)
                    thresh_down = np.percentile(allcond_data_i[mat_type][band][phase].reshape(-1), percentile_thresh_down)

                    for x in range(mat_dfc_clean_i[mat_type][band][phase].shape[1]):
                        for y in range(mat_dfc_clean_i[mat_type][band][phase].shape[1]):
                            if (mat_dfc_clean_i[mat_type][band][phase][x,y] < thresh_up) & (mat_dfc_clean_i[mat_type][band][phase][x,y] > thresh_down):
                                mat_dfc_clean_i[mat_type][band][phase][x,y] = 0

        #### fill res containers
        allcond_data_thresh[cond] = mat_dfc_clean_i

    #### adjust scale
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_to_compute:

            max_list = np.array(())

            for cond in cond_to_compute:

                max_list = np.append(max_list, allcond_scales_abs[cond][mat_type][band])

            scales_abs[mat_type][band] = max_list.max()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'FC', 'summary'))

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for band_prep in band_prep_list:

            n_cols_raw = len(cond_to_compute)

            ######## NO THRESH ########
            #phase = 'whole'
            for phase in phase_list:

                #### mat plot raw 
                fig, axs = plt.subplots(nrows=len(freq_band_dict_FC_function[band_prep]), ncols=n_cols_raw, figsize=(15,15))

                if monopol:
                    plt.suptitle(f'{phase} {mat_type}')
                else:
                    plt.suptitle(f'{phase} {mat_type} bi')
                
                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                    for c, cond in enumerate(cond_to_compute):

                        ax = axs[r, c]

                        if c == 0:
                            ax.set_ylabel(band)
                        if r == 0:
                            ax.set_title(cond)
                        
                        cax = ax.matshow(mat_phase_allcond[cond][mat_type][band][phase], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                        fig.colorbar(cax, ax=ax)

                        ax.set_yticks(np.arange(roi_in_data_allplot.shape[0]))
                        ax.set_yticklabels(roi_in_data_allplot)
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

                        for c, cond in enumerate(cond_to_compute):

                            mne_connectivity.viz.plot_connectivity_circle(mat_phase_allcond[cond][mat_type][band][phase], node_names=roi_in_data_allplot, n_lines=None, 
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
                    for c, cond in enumerate(cond_to_compute):

                        ax = axs[r, c]

                        if c == 0:
                            ax.set_ylabel(band)
                        if r == 0:
                            ax.set_title(f'{phase}')
                        
                        cax = ax.matshow(allcond_data_thresh[cond][mat_type][band][phase], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                        fig.colorbar(cax, ax=ax)

                        ax.set_yticks(np.arange(roi_in_data_allplot.shape[0]))
                        ax.set_yticklabels(roi_in_data_allplot)
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

                        for c, cond in enumerate(cond_to_compute):

                            mne_connectivity.viz.plot_connectivity_circle(allcond_data_thresh[cond][mat_type][band][phase], node_names=roi_in_data_allplot, n_lines=None, 
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

    #monopol = True
    for monopol in [True, False]:

        FR_CV_normalized = True

        cond_to_compute = ['RD_CV', 'RD_SV', 'RD_FV']
        band_to_compute = ['theta', 'alpha', 'beta', 'l_gamma', 'h_gamma']

        #### get baselines
        baselines = precompute_baselines_allplot(band_to_compute, monopol, rscore_computation=False)

        mat_phase_allcond = {}

        #### save fig
        #cond = 'RD_CV'
        for cond in cond_to_compute:
            
            print(cond, monopol)
                
            xr_pairs_allplot, pair_unique_allplot, roi_in_data_allplot = precompute_dfc_mat_allplot(band_to_compute, cond, monopol)
            mat_phase = precompute_dfc_mat_allplot_phase(xr_pairs_allplot, pair_unique_allplot, band_to_compute, baselines, monopol, rscore_computation=False)

            save_fig_dfc_allplot(mat_phase, cond, band_to_compute, roi_in_data_allplot, monopol, FR_CV_normalized=True)

            mat_phase_allcond[cond] = mat_phase

        process_dfc_res_summary(mat_phase_allcond, roi_in_data_allplot, cond_to_compute, band_to_compute, monopol)








