



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

import frites

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False





########################################
######## PLI ISPC DFC FC ######## 
########################################

#cond, band_prep, band, freq, trial_i = 'FR_CV', 'hf', 'l_gamma', [50, 80], 0
def get_pli_ispc_fc_dfc_trial(sujet, cond, band_prep, band, freq, trial_i, monopol):

    #### load data
    data = load_data_sujet(sujet, band_prep, cond, trial_i, monopol)
    
    data_length = data.shape[-1]

    #### get params
    prms = get_params(sujet, monopol)

    wavelets = get_wavelets()

    respfeatures_allcond = load_respfeatures(sujet)

    #### initiate res
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_mnpol_{monopol}_dfc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data_length))

    #### generate fake convolutions
    # convolutions = np.random.random(len(prms['chan_list_ieeg']) * nfrex * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex, data.shape[1]) * 1j
    # convolutions += np.random.random(len(prms['chan_list_ieeg']) * nfrex * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex, data.shape[1]) 

    # convolutions = np.zeros((len(prms['chan_list_ieeg']), nfrex, data.shape[1])) 

    print('CONV')

    #nchan = 0
    def convolution_x_wavelets_nchan(nchan_i, nchan):

        print_advancement(nchan_i, len(prms['chan_list_ieeg']), steps=[25, 50, 75])
        
        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

        x = data[nchan_i,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan_i,:,:] = nchan_conv

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan_i, nchan) for nchan_i, nchan in enumerate(prms['chan_list_ieeg']))

    #### free memory
    del data        

    #### verif conv
    if debug:
        plt.plot(convolutions[0,0,:])
        plt.show()

    #### identify roi in data
    df_loca = get_loca_df(sujet, monopol)
    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    roi_in_data = df_sorted['ROI'].unique()

    #### compute index
    pairs_possible = []
    for pair_A_i, pair_A in enumerate(roi_in_data):
        for pair_B_i, pair_B in enumerate(roi_in_data[pair_A_i:]):
            if pair_A == pair_B:
                continue
            pairs_possible.append(f'{pair_A}-{pair_B}')

    pairs_to_compute = []
    pairs_to_compute_anat = []
    for pair_A in prms['chan_list_ieeg']:

        anat_A = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_A)]
        if monopol == False:
            pair_A = f"{pair_A.split('-')[0]}|{pair_A.split('-')[1]}"
        
        for pair_B in prms['chan_list_ieeg']:

            anat_B = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_B)]
            if monopol == False:
                pair_B = f"{pair_B.split('-')[0]}|{pair_B.split('-')[1]}"

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')
            pairs_to_compute_anat.append(f'{anat_A}-{anat_B}')

    pairs_to_compute_anat = np.array(pairs_to_compute_anat)

    ######## FC / DFC ########

    os.chdir(path_memmap)
    res_fc_phase = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_mnpol_{monopol}_dfc_phase.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), 3, nfrex))

    #pair_to_compute_i, pair_to_compute = 0, pairs_to_compute[0]
    def compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute):

        print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

        pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
        pair_A, pair_B = pair_A.replace('|', '-'), pair_B.replace('|', '-')
        pair_A_i, pair_B_i = prms['chan_list_ieeg'].index(pair_A), prms['chan_list_ieeg'].index(pair_B)

        as1 = convolutions[pair_A_i,:,:]
        as2 = convolutions[pair_B_i,:,:]

        #### stretch data
        as1_stretch = stretch_data_tf(respfeatures_allcond[cond][trial_i], stretch_point_TF, as1, prms['srate'])[0]
        as2_stretch = stretch_data_tf(respfeatures_allcond[cond][trial_i], stretch_point_TF, as2, prms['srate'])[0]

        #phase = 'inspi'
        for phase_i, phase in enumerate(['whole', 'inspi', 'expi']):

            #### chunk
            if phase == 'whole':
                as1_stretch_chunk =  as1_stretch.reshape((nfrex, -1))
                as2_stretch_chunk =  as2_stretch.reshape((nfrex, -1))

            if phase == 'inspi':
                as1_stretch_chunk =  as1_stretch[:,:,:int(stretch_point_TF*ratio_stretch_TF)].reshape((nfrex, -1))
                as2_stretch_chunk =  as2_stretch[:,:,:int(stretch_point_TF*ratio_stretch_TF)].reshape((nfrex, -1))

            if phase == 'expi':
                as1_stretch_chunk =  as1_stretch[:,:,int(stretch_point_TF*ratio_stretch_TF):].reshape((nfrex, -1))
                as2_stretch_chunk =  as2_stretch[:,:,int(stretch_point_TF*ratio_stretch_TF):].reshape((nfrex, -1))

            ##### collect "eulerized" phase angle differences
            cdd = np.exp(1j*(np.angle(as1_stretch_chunk)-np.angle(as2_stretch_chunk)))
            
            ##### compute ISPC and WPLI (and average over trials!)
            res_fc_phase[0, pair_to_compute_i, phase_i, :] = np.abs(np.mean(cdd, axis=1))
            # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
            res_fc_phase[1, pair_to_compute_i, phase_i, :] = np.abs( np.mean( np.imag(cdd), axis=1 ) ) / np.mean( np.abs( np.imag(cdd) ), axis=1 )

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli_dfc)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

    if debug:
        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):
            print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[10, 20, 50, 75])
            compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute)

    res_fc_phase_export = res_fc_phase.copy()

    #### remove memmap
    os.chdir(path_memmap)
    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_mnpol_{monopol}_dfc_convolutions.dat')
        del convolutions
    except:
        pass

    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_mnpol_{monopol}_dfc_phase.dat')
        del res_fc_phase
    except:
        pass

    return res_fc_phase_export








def get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, monopol):

    #### verif computation
    if monopol:
        if os.path.exists(os.path.join(path_precompute, sujet, 'DFC', f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')):
            print(f'ALREADY DONE DFC {cond} {band}')
    else:
        if os.path.exists(os.path.join(path_precompute, sujet, 'DFC', f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')):
            print(f'ALREADY DONE DFC {cond} {band}')

    if monopol:
        if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs.nc')):
            print(f'ALREADY DONE FC {cond} {band}')
            return
    else:
        if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')):
            print(f'ALREADY DONE FC {cond} {band}')
            return

    #### get n trial for cond
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    if monopol:
        n_trials = len([i for i in os.listdir() if i.find(cond) != -1 and i.find(band_prep) != -1 and i.find(f'_bi') == -1])
    else:
        n_trials = len([i for i in os.listdir() if i.find(cond) != -1 and i.find(band_prep) != -1 and i.find(f'_bi') != -1])

    #### identify anat info
    prms = get_params(sujet, monopol)
    df_loca = get_loca_df(sujet, monopol)

    pairs_to_compute = []
    pairs_to_compute_anat = []

    for pair_A in prms['chan_list_ieeg']:

        anat_A = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_A)]

        if monopol == False:
            pair_A = f"{pair_A.split('-')[0]}|{pair_A.split('-')[1]}"
        
        for pair_B in prms['chan_list_ieeg']:
            anat_B = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_B)]

            if monopol == False:
                pair_B = f"{pair_B.split('-')[0]}|{pair_B.split('-')[1]}"

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')
            pairs_to_compute_anat.append(f'{anat_A}-{anat_B}')

    #### compute
    mat_fc = []

    #### for dfc computation
    wavelets = get_wavelets()

    #trial_i = 1
    for trial_i in range(n_trials):
        res_fc_dfc = get_pli_ispc_fc_dfc_trial(sujet, cond, band_prep, band, freq, trial_i, monopol)
        mat_fc.append(res_fc_dfc)

    if debug:
        plt.plot(res_fc_dfc[1,0,0,:], label='whole')
        plt.plot(res_fc_dfc[1,0,1,:], label='inspi')
        plt.plot(res_fc_dfc[1,0,2,:], label='expi')
        plt.legend()
        plt.show()

    #### simulate data
    if debug:
        for trial_i in range(n_trials):
            mat_fc.append(np.random.random((2, pairs_to_compute_anat.shape[0], 50)))

    #### mean across trials
    for trial_i in range(n_trials):
        if trial_i == 0:
            mat_fc_mean = mat_fc[trial_i]

        else:
            mat_fc_mean += mat_fc[trial_i]

    mat_fc_mean /= n_trials

    #### export
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    if monopol:
        dict_xr = {'mat_type' : ['ispc', 'wpli'], 'pairs' : pairs_to_compute_anat, 'phase' : ['whole', 'inspi', 'expi'], 'nfrex' : range(nfrex)}
        xr_export = xr.DataArray(mat_fc_mean, coords=dict_xr.values(), dims=dict_xr.keys())
        xr_export.to_netcdf(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs.nc')
    else:
        dict_xr = {'mat_type' : ['ispc', 'wpli'], 'pairs' : pairs_to_compute_anat, 'phase' : ['whole', 'inspi', 'expi'], 'nfrex' : range(nfrex)}
        xr_export = xr.DataArray(mat_fc_mean, coords=dict_xr.values(), dims=dict_xr.keys())
        xr_export.to_netcdf(f'{sujet}_FC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')







################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    # sujet = 'CHEe'
    # sujet = 'GOBc'
    # sujet = 'MAZm'
    # sujet = 'TREt'

    #sujet = sujet_list[-1]
    for sujet in sujet_list:

        #monopol = True
        for monopol in [True, False]:

            prms = get_params(sujet, monopol)

            print('######## PRECOMPUTE DFC ########') 
            # cond = 'FR_CV'
            for cond in prms['conditions']:
                #band_prep = 'lf'
                for band_prep in band_prep_list:
                    #band, freq = 'theta', [4,8]
                    for band, freq in freq_band_dict_FC_function[band_prep].items():

                        # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, monopol)
                        execute_function_in_slurm_bash_mem_choice('n8_precompute_FC_DFC', 'get_wpli_ispc_fc_dfc', [sujet, cond, band_prep, band, freq, monopol], '35G')

        

