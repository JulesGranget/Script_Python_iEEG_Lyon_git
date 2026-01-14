



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *


debug = False






################################
######## WPLI ISPC ######## 
################################



#sujet, monopol = sujet_list_dfc[0], False
def get_ISPC_WPLI(sujet, monopol):
    
    # Check if results already exist
    if monopol:
        ispc_path = os.path.join(path_precompute, 'allplot', 'FC', f'ISPC_{sujet}_stretch_rscore.nc')
        wpli_path = os.path.join(path_precompute, 'allplot', 'FC', f'WPLI_{sujet}_stretch_rscore.nc')
    else:
        ispc_path = os.path.join(path_precompute, 'allplot', 'FC', f'ISPC_{sujet}_stretch_rscore_bi.nc')
        wpli_path = os.path.join(path_precompute, 'allplot', 'FC', f'WPLI_{sujet}_stretch_rscore_bi.nc')

    if os.path.exists(ispc_path) and os.path.exists(wpli_path):
        print(f'ALREADY DONE')
        return
    
    if sujet in sujet_list:
        cond_sel = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']
    else:
        cond_sel = ['FR_CV']
    
    #### identify anat info
    band_prep = 'wb'
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    df_loca = get_loca_df(sujet, monopol)

    df_loca = df_loca.query(f"ROI in {ROI_list_dfc}")

    pairs_to_compute = []
    pairs_to_compute_anat = []

    for pair_A_i, pair_A in enumerate(df_loca['name']):

        anat_A = df_loca['ROI'].iloc[pair_A_i]

        if monopol == False:
            pair_A = f"{pair_A.split('-')[0]}|{pair_A.split('-')[1]}"
        
        for pair_B_i, pair_B in enumerate(df_loca['name']):

            anat_B = df_loca['ROI'].iloc[pair_B_i]

            if monopol == False:
                pair_B = f"{pair_B.split('-')[0]}|{pair_B.split('-')[1]}"

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute or anat_A == anat_B:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')
            pairs_to_compute_anat.append(f'{anat_A}-{anat_B}')
    
    #pair_i = 0
    def compute_pair(pair_i):

        os.chdir(path_prep)
        # xr_norm_params = xr.open_dataarray('norm_params.nc')
        respfeatures_sujet = load_respfeatures(sujet)
    
        pair = pairs_to_compute[pair_i]
        pair_A, pair_B = pair.split('-')

        if not monopol:
            idx_A, idx_B = np.where(df_loca['name'] == pair_A.replace("|", "-"))[0][0], np.where(df_loca['name'] == pair_B.replace("|", "-"))[0][0]
        else:
            idx_A, idx_B = np.where(df_loca['name'] == pair_A)[0][0], np.where(df_loca['name'] == pair_B)[0][0]

        print(f"{pair} {int(pair_i*100/len(pairs_to_compute))}%", flush=True)
        
        ISPC_all_bands = np.zeros((len(cond_sel), len(freq_band_fc), nrespcycle_FC_FR_CV, stretch_point_FC))
        WPLI_all_bands = np.zeros((len(cond_sel), len(freq_band_fc), nrespcycle_FC_FR_CV, stretch_point_FC))
        
        for band_i, band in enumerate(freq_band_fc):

            wavelets = get_wavelets_fc(freq_band_fc[band])
            win_slide = ISPC_window_size[band]
            
            for cond_i, cond in enumerate(cond_sel):

                count_cycle_cond = 0

                for session_i in range(session_count[cond]):

                    if count_cycle_cond > nrespcycle_FC_FR_CV:
                        continue

                    # print(band, cond, session_i)

                    data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[[idx_A, idx_B],:]

                    respfeatures_sujet_cond = respfeatures_sujet[cond][session_i]

                    start_insert_cycle_resmat = count_cycle_cond 
                    count_cycle_cond += respfeatures_sujet_cond.shape[0]       
                    stop_insert_cycle_resmat = start_insert_cycle_resmat + respfeatures_sujet_cond.shape[0]
                    if stop_insert_cycle_resmat <= nrespcycle_FC_FR_CV:
                        stop_insert_cycle_computemat = respfeatures_sujet_cond.shape[0]   
                    else:
                        stop_insert_cycle_computemat = nrespcycle_FC_FR_CV - start_insert_cycle_resmat
                        stop_insert_cycle_resmat = nrespcycle_FC_FR_CV

                    data_rscore = data.copy()

                    for i in range(2):

                        data_rscore[i] = (data[i] - np.median(data[i])) * 0.6745 / scipy.stats.median_abs_deviation(data[i])

                        # data_rscore = (data - xr_norm_params.loc[sujet, 'CSD', 'median', [pair_A, pair_B]].values.reshape(-1,1))
                        #               * 0.6745 / xr_norm_params.loc[sujet, 'CSD', 'mad', [pair_A, pair_B]].values.reshape(-1,1)
                                    
                    convolutions = np.array([
                        np.array([scipy.signal.fftconvolve(data_rscore[ch, :], wavelet, 'same') for wavelet in wavelets])
                        for ch in [0, 1]
                    ])
                    
                    x_conv, y_conv = convolutions[0], convolutions[1]

                    #### slide
                    x_pad = np.pad(x_conv, ((0, 0), (int(win_slide*fc_win_overlap/2), int(win_slide*fc_win_overlap/2))), mode='reflect')
                    y_pad = np.pad(y_conv, ((0, 0), (int(win_slide*fc_win_overlap/2), int(win_slide*fc_win_overlap/2))), mode='reflect')

                    win_vec = np.arange(0, x_conv.shape[-1]-int(win_slide*fc_win_overlap/2), int(win_slide*fc_win_overlap/2)).astype('int')

                    ISPC_slide = np.zeros((x_conv.shape[0], win_vec.size))
                    WPLI_slide = np.zeros((x_conv.shape[0], win_vec.size))
                    for wavelet_i in range(x_conv.shape[0]):
                        for win_i in range(win_vec.size):
                            ISPC_slide[wavelet_i,win_i] = get_ISPC_2sig(x_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide], y_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide])
                            WPLI_slide[wavelet_i,win_i] = get_WPLI_2sig(x_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide], y_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide])

                    ISPC_slide = np.median(ISPC_slide, axis=0)
                    WPLI_slide = np.median(WPLI_slide, axis=0)

                    #### interpol
                    f = scipy.interpolate.interp1d(np.linspace(0, x_conv.shape[-1], ISPC_slide.size), ISPC_slide)
                    ISPC_slide_interp = f(np.arange(x_conv.shape[-1]))

                    f = scipy.interpolate.interp1d(np.linspace(0, x_conv.shape[-1], WPLI_slide.size), ISPC_slide)
                    WPLI_slide_interp = f(np.arange(x_conv.shape[-1]))
                    
                    ISPC_stretch = stretch_data_fc(respfeatures_sujet_cond, stretch_point_FC, ISPC_slide_interp, srate)[0]
                    WPLI_stretch = stretch_data_fc(respfeatures_sujet_cond, stretch_point_FC, WPLI_slide_interp, srate)[0]

                    if debug:

                        for cycle_i in range(ISPC_stretch.shape[0]):

                            plt.plot(ISPC_stretch[cycle_i,:], alpha=0.2)

                        plt.plot(np.median(ISPC_stretch, axis=0), color='r')
                        plt.show()
                    
                    ISPC_all_bands[cond_i, band_i, start_insert_cycle_resmat:stop_insert_cycle_resmat] = ISPC_stretch[:stop_insert_cycle_computemat,:]
                    WPLI_all_bands[cond_i, band_i, start_insert_cycle_resmat:stop_insert_cycle_resmat] = WPLI_stretch[:stop_insert_cycle_computemat,:]

        if debug:

            band_i = 0
            plt.plot(np.median(WPLI_all_bands[0,band_i], axis=0))
            plt.plot(np.median(WPLI_all_bands[1,band_i], axis=0))
            plt.show()
                
        return ISPC_all_bands, WPLI_all_bands
    
    results = joblib.Parallel(n_jobs=n_core, prefer='processes', batch_size=1)(
        joblib.delayed(compute_pair)(pair_i) for pair_i in range(len(pairs_to_compute))
    )

    # Preallocate results
    ISPC_sujet = np.zeros((len(pairs_to_compute), len(cond_sel), len(freq_band_fc), nrespcycle_FC_FR_CV, stretch_point_FC))
    WPLI_sujet = np.zeros((len(pairs_to_compute), len(cond_sel), len(freq_band_fc), nrespcycle_FC_FR_CV, stretch_point_FC))
    
    for pair_i in range(len(pairs_to_compute)):
        ISPC_sujet[pair_i] = results[pair_i][0]
        WPLI_sujet[pair_i] = results[pair_i][1]
    
    xr_dict = {'pair': pairs_to_compute_anat, 'cond': cond_sel, 'band': list(freq_band_fc.keys()), 'cycle': np.arange(nrespcycle_FC_FR_CV), 'time': np.arange(stretch_point_FC)}
    xr_ispc = xr.DataArray(data=ISPC_sujet, dims=xr_dict.keys(), coords=xr_dict.values())
    xr_wpli = xr.DataArray(data=WPLI_sujet, dims=xr_dict.keys(), coords=xr_dict.values())

    ISPC_sujet_rscore = ISPC_sujet.copy()
    WPLI_sujet_rscore = WPLI_sujet.copy()

    for cond_i, cond in enumerate(cond_sel):
        for pair_i, pair in enumerate(pairs_to_compute_anat):
            for band_i, band in enumerate(freq_band_fc):
                for cycle_i in range(nrespcycle_FC_FR_CV):
                    ISPC_sujet_rscore[pair_i, cond_i, band_i, cycle_i] = (ISPC_sujet[pair_i, cond_i, band_i, cycle_i, :] - np.median(ISPC_sujet[pair_i, cond_i, band_i, cycle_i, :])) * 0.6745 / scipy.stats.median_abs_deviation(ISPC_sujet[pair_i, cond_i, band_i, cycle_i, :])
                    WPLI_sujet_rscore[pair_i, cond_i, band_i, cycle_i] = (WPLI_sujet[pair_i, cond_i, band_i, cycle_i, :] - np.median(WPLI_sujet[pair_i, cond_i, band_i, cycle_i, :])) * 0.6745 / scipy.stats.median_abs_deviation(WPLI_sujet[pair_i, cond_i, band_i, cycle_i, :])
    
    xr_ispc_rscore = xr.DataArray(data=ISPC_sujet_rscore, dims=xr_dict.keys(), coords=xr_dict.values())
    xr_wpli_rscore = xr.DataArray(data=WPLI_sujet_rscore, dims=xr_dict.keys(), coords=xr_dict.values())

    if debug:
        
        pair_i = 0
        band_i = 0

        fc_diff = ISPC_sujet.values[pair_i, 1, band_i] - ISPC_sujet.values[pair_i, 0, band_i]
        fc_diff = WPLI_sujet.values[pair_i, 1, band_i] - WPLI_sujet.values[pair_i, 0, band_i]

        for cycle_i in range(nrespcycle_FC_FR_CV):

            plt.plot(fc_diff[cycle_i], alpha=0.2)
            
        plt.plot(np.median(fc_diff, axis=0), color='r')
        plt.show()

        fc_diff = ISPC_sujet_rscore.values[pair_i, 1, band_i] - ISPC_sujet_rscore.values[pair_i, 0, band_i]
        fc_diff = WPLI_sujet_rscore.values[pair_i, 1, band_i] - WPLI_sujet_rscore.values[pair_i, 0, band_i]

        for cycle_i in range(nrespcycle_FC_FR_CV):

            plt.plot(fc_diff[cycle_i], alpha=0.2)
            
        plt.plot(np.median(fc_diff, axis=0), color='r')
        plt.show()

        plt.plot(np.median(ISPC_sujet.values[pair_i, 1, band_i] - ISPC_sujet.values[pair_i, 0, band_i], axis=0), color='r')
        plt.plot(np.median(ISPC_sujet_rscore.values[pair_i, 1, band_i] - ISPC_sujet_rscore.values[pair_i, 0, band_i], axis=0), color='r')
        plt.show()

        plt.plot(np.median(WPLI_sujet.values[pair_i, 1, band_i] - WPLI_sujet.values[pair_i, 0, band_i], axis=0), color='r')
        plt.plot(np.median(WPLI_sujet_rscore.values[pair_i, 1, band_i] - WPLI_sujet_rscore.values[pair_i, 0, band_i], axis=0), color='r')
        plt.show()
        
    os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))

    if monopol:
        xr_ispc.to_netcdf(f'ISPC_{sujet}_stretch.nc')
        xr_ispc_rscore.to_netcdf(f'ISPC_{sujet}_stretch_rscore.nc')
        xr_wpli.to_netcdf(f'WPLI_{sujet}_stretch.nc')
        xr_wpli_rscore.to_netcdf(f'WPLI_{sujet}_stretch_rscore.nc')
    else:
        xr_ispc.to_netcdf(f'ISPC_{sujet}_stretch_bi.nc')
        xr_ispc_rscore.to_netcdf(f'ISPC_{sujet}_stretch_rscore_bi.nc')
        xr_wpli.to_netcdf(f'WPLI_{sujet}_stretch_bi.nc')
        xr_wpli_rscore.to_netcdf(f'WPLI_{sujet}_stretch_rscore_bi.nc')

    print('done')

    




################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    # sujet = 'CHEe'
    # sujet = 'GOBc'
    # sujet = 'MAZm'
    # sujet = 'TREt'
    # sujet = 'POTm'

    list_params = []
    #sujet = sujet_list_dfc[0]
    for sujet in sujet_list_dfc_FR_CV:
        #monopol = True
        for monopol in [True, False]:
            list_params.append([sujet, monopol])
            
    print('######## PRECOMPUTE DFC ########', flush=True) 
    execute_function_in_slurm_bash('n08_precompute_DFC', 'get_ISPC_WPLI', list_params, mem='20G')
    #sync_folders__push_to_crnldata()

    

