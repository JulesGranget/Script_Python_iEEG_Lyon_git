
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *

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
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))

        if monopol:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        else:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')

        chan_list_ieeg = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        #nchan = chan_list_ieeg_csv[0]
        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
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
######## COMPUTE STATS ########
################################


#tf, nchan = data_allcond[cond][odor_i], n_chan
def get_tf_stats(tf, min, max):

    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(tf.shape[0]):
        mask = np.logical_or(tf_thresh[wavelet_i, :] > max[wavelet_i], tf_thresh[wavelet_i, :] < min[wavelet_i])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    return tf_thresh




def precompute_tf_ROI_STATS(ROI, monopol):

    ######## INTRA ########
    print(f'#### COMPUTE TF STATS INTRA {ROI} ####', flush=True)

    #cond = conditions[1]
    for cond in conditions:

        #### identify if already computed
        os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

        if monopol:
            if os.path.exists(f'allsujet_{ROI}_tf_STATS_{cond}_intra.npy'):
                print(f'{cond} ALREADY COMPUTED', flush=True)
                continue
        else:
            if os.path.exists(f'allsujet_{ROI}_tf_STATS_{cond}_intra_bi.npy'):
                print(f'{cond} ALREADY COMPUTED', flush=True)
                continue

        #### params
        ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond, monopol)

        if len(ROI_dict_plots[ROI]) == 0:
            print('ROI only in FR_CV', flush=True)
            continue
        else:
            site_list = ROI_dict_plots[ROI]

        #### load baselines
        print(f'#### LOAD {cond} ####', flush=True)

        cycle_baseline_tot = 0

        #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
        for sujet_i, (sujet, chan_name) in enumerate(site_list):

            respfeatures_tot = load_respfeatures(sujet)[cond]

            for session_i in range(session_count[cond]):

                # print(sujet, respfeatures_tot[session_i]['select'].sum(), flush=True)
                cycle_baseline_tot += respfeatures_tot[session_i]['select'].sum()

        os.chdir(path_memmap)
        if monopol:
            tf_stretch_baselines = np.memmap(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}.dat', dtype=np.float32, mode='w+', 
                                            shape=(cycle_baseline_tot, nfrex, int(stretch_point_TF/2)))
            tf_stretch_cond = np.memmap(f'{ROI}_{cond}_tf_STATS_cond_{monopol}.dat', dtype=np.float32, mode='w+', 
                                        shape=(cycle_baseline_tot, nfrex, int(stretch_point_TF/2)))
        else:
            tf_stretch_baselines = np.memmap(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}_bi.dat', dtype=np.float32, mode='w+', 
                                            shape=(cycle_baseline_tot, nfrex, int(stretch_point_TF/2)))
            tf_stretch_cond = np.memmap(f'{ROI}_{cond}_tf_STATS_cond_{monopol}_bi.dat', dtype=np.float32, mode='w+', 
                                        shape=(cycle_baseline_tot, nfrex, int(stretch_point_TF/2)))
        
        cycle_baseline_i = 0

        #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
        for sujet_i, (sujet, chan_name) in enumerate(site_list):

            print_advancement(sujet_i, len(site_list), steps=[25, 50, 75])

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

            if sujet[:3] != 'pat':
                if monopol:
                    chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

            chan_i = chan_list_ieeg.index(chan_name)

            if monopol:
                tf_load = np.load(f'{sujet}_tf_conv_{cond}.npy')[chan_i,:,:,:]
            else:
                tf_load = np.load(f'{sujet}_tf_conv_{cond}_bi.npy')[chan_i,:,:,:]

            tf_load_cycle_n = tf_load.shape[0]

            tf_stretch_baselines[cycle_baseline_i:(cycle_baseline_i+tf_load_cycle_n),:,:] = tf_load[:,:,:int(stretch_point_TF/2)]
            tf_stretch_cond[cycle_baseline_i:(cycle_baseline_i+tf_load_cycle_n),:,:] = tf_load[:,:,int(stretch_point_TF/2):]

            cycle_baseline_i += tf_load_cycle_n

        del tf_load

        print(f'#### COMPUTE {cond} ####', flush=True)

        #### MA
        os.chdir(path_memmap)
        if monopol:
            pixel_based_distrib = np.memmap(f'{ROI}_{cond}_pixel_distrib_intra.dat', dtype=np.float32, mode='w+', shape=(nfrex, 2))
        else:
            pixel_based_distrib = np.memmap(f'{ROI}_{cond}_pixel_distrib_intra_bi.dat', dtype=np.float32, mode='w+', shape=(nfrex, 2))
        
        #### define ncycle
        n_cycle_baselines = tf_stretch_baselines.shape[0]
        n_cycle_cond = tf_stretch_cond.shape[0]

        #### space allocation
        _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
        pixel_based_distrib_i = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
        tf_shuffle = np.zeros((n_cycle_cond, nfrex, int(stretch_point_TF/2)))

        #surrogates_i = 0
        for surrogates_i in range(n_surrogates_tf):

            print_advancement(surrogates_i, n_surrogates_tf, steps=[25, 50, 75])

            #### random selection
            draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
            sel_baseline = np.random.randint(low=0, high=n_cycle_baselines, size=(draw_indicator == 1).sum())
            sel_cond = np.random.randint(low=0, high=n_cycle_cond, size=(draw_indicator == 0).sum())

            #### extract max min
            tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[sel_baseline, :, :]
            tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[sel_cond, :, :]

            _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
            
            pixel_based_distrib_i[:, surrogates_i, 0] = _min
            pixel_based_distrib_i[:, surrogates_i, 1] = _max

        min, max = np.median(pixel_based_distrib_i[:,:,0], axis=1), np.median(pixel_based_distrib_i[:,:,1], axis=1) 

        if debug:

            tf_nchan = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

            time = np.arange(tf_nchan.shape[-1])

            plt.pcolormesh(time, frex, tf_nchan, shading='gouraud', cmap='seismic')
            plt.contour(time, frex, get_tf_stats(tf_nchan, min, max), levels=0, colors='g')
            plt.yscale('log')
            plt.show()

            #wavelet_i = 0
            for wavelet_i in range(20):
                count, _, _ = plt.hist(tf_nchan[wavelet_i, :], bins=500)
                plt.vlines([min[wavelet_i], max[wavelet_i]], ymin=0, ymax=count.max(), color='r')
                plt.show()

        pixel_based_distrib[:,0] = min
        pixel_based_distrib[:,1] = max        

        #### plot 
        if debug:

            median_max_diff = np.abs(np.median(tf_stretch_cond, axis=1).reshape(-1) - np.median(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))).max()
            vmin = -median_max_diff
            vmax = median_max_diff

            for nchan, nchan_name in enumerate(chan_list_ieeg):

                tf_plot = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

                time = np.arange(tf_plot.shape[-1])

                plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
                plt.contour(time, frex, get_tf_stats(tf_plot, min, max), levels=0, colors='g', vmin=vmin, vmax=vmax)
                plt.yscale('log')
                plt.yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
                plt.title(nchan_name)
                plt.show()


        ######## SAVE ########

        print(f'SAVE', flush=True)

        os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

        if monopol:
            np.save(f'allsujet_{ROI}_tf_STATS_{cond}_intra.npy', pixel_based_distrib)
        else:
            np.save(f'allsujet_{ROI}_tf_STATS_{cond}_intra_bi.npy', pixel_based_distrib)    

        del tf_stretch_cond
        
        os.chdir(path_memmap)
        try:
            if monopol:
                os.remove(f'{ROI}_{cond}_pixel_distrib_intra.dat')
            else:
                os.remove(f'{ROI}_{cond}_pixel_distrib_intra_bi.dat')
            del pixel_based_distrib
        except:
            pass

        #### remove cond
        os.chdir(path_memmap)
        try:
            if monopol:
                os.remove(f'{ROI}_{cond}_tf_STATS_cond_{monopol}.dat')
            else:
                os.remove(f'{ROI}_{cond}_tf_STATS_cond_{monopol}_bi.dat')
            del tf_stretch_cond
        except:
            pass

        #### remove baseline
        os.chdir(path_memmap)
        try:
            if monopol:
                os.remove(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}.dat')
            else:
                os.remove(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}_bi.dat')
            del tf_stretch_baselines
        except:
            pass
    








    ######## INTER ########

    #### if only FR_CV no inter
    if len(conditions) == 1:

        return
    
    print(f'#### COMPUTE TF STATS INTER {ROI} ####', flush=True)

    #### params
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', monopol)

    if len(ROI_dict_plots[ROI]) == 0:
        print('ROI only in FR_CV', flush=True)
        return
    else:
        site_list = ROI_dict_plots[ROI] 

    #### load baselines
    print('#### LOAD BASELINES ####', flush=True)

    cycle_baseline_tot = 0
    cond = 'FR_CV'

    #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
    for sujet_i, (sujet, chan_name) in enumerate(site_list):

        respfeatures_tot = load_respfeatures(sujet)[cond]

        for session_i in range(session_count[cond]):

            cycle_baseline_tot += respfeatures_tot[session_i]['select'].sum()

    os.chdir(path_memmap)
    if monopol:
        tf_stretch_baselines = np.memmap(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}.dat', dtype=np.float32, mode='w+', 
                                shape=(cycle_baseline_tot, nfrex, int(stretch_point_TF/2)))
    else:
        tf_stretch_baselines = np.memmap(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}_bi.dat', dtype=np.float32, mode='w+', 
                                shape=(cycle_baseline_tot, nfrex, int(stretch_point_TF/2)))
    
    cycle_baseline_i = 0

    #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
    for sujet_i, (sujet, chan_name) in enumerate(site_list):

        print_advancement(sujet_i, len(site_list), steps=[25, 50, 75])

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

        if sujet[:3] != 'pat':
                if monopol:
                    chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

        chan_i = chan_list_ieeg.index(chan_name)

        if monopol:
            tf_load = np.load(f'{sujet}_tf_conv_{cond}.npy')[chan_i,:,:,:int(stretch_point_TF/2)]
        else:
            tf_load = np.load(f'{sujet}_tf_conv_{cond}_bi.npy')[chan_i,:,:,:int(stretch_point_TF/2)]

        tf_load_cycle_n = tf_load.shape[0]

        tf_stretch_baselines[cycle_baseline_i:(cycle_baseline_i+tf_load_cycle_n),:,:] = tf_load

        cycle_baseline_i += tf_load_cycle_n

    del tf_load

    #cond = conditions[0]
    for cond in conditions:

        if cond == 'FR_CV':
            continue

        #### identify if already computed
        os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

        if monopol:
            if os.path.exists(f'allsujet_{ROI}_tf_STATS_{cond}_inter.npy'):
                print(f'{cond} ALREADY COMPUTED', flush=True)
                continue
        else:
            if os.path.exists(f'allsujet_{ROI}_tf_STATS_{cond}_inter_bi.npy'):
                print(f'{cond} ALREADY COMPUTED', flush=True)
                continue

        #### params
        ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', monopol)
        site_list = ROI_dict_plots[ROI]

        #### load cond
        print(f'#### LOAD {cond} ####', flush=True)

        cycle_cond_tot = 0

        #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
        for sujet_i, (sujet, chan_name) in enumerate(site_list):

            respfeatures_tot = load_respfeatures(sujet)[cond]

            for session_i in range(session_count[cond]):

                cycle_cond_tot += respfeatures_tot[session_i]['select'].sum()

        os.chdir(path_memmap)
        if monopol:
            tf_stretch_cond = np.memmap(f'{ROI}_{cond}_tf_STATS_cond_{monopol}.dat', dtype=np.float32, mode='w+', 
                                        shape=(cycle_cond_tot, nfrex, int(stretch_point_TF/2)))
        else:
            tf_stretch_cond = np.memmap(f'{ROI}_{cond}_tf_STATS_cond_{monopol}_bi.dat', dtype=np.float32, mode='w+', 
                                        shape=(cycle_cond_tot, nfrex, int(stretch_point_TF/2)))
        
        cycle_cond_i = 0

        #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
        for sujet_i, (sujet, chan_name) in enumerate(site_list):

            print_advancement(sujet_i, len(site_list), steps=[25, 50, 75])

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

            if sujet[:3] != 'pat':
                if monopol:
                    chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

            chan_i = chan_list_ieeg.index(chan_name)

            if monopol:
                tf_load = np.load(f'{sujet}_tf_conv_{cond}.npy')[chan_i,:,:,int(stretch_point_TF/2):]
            else:
                tf_load = np.load(f'{sujet}_tf_conv_{cond}_bi.npy')[chan_i,:,:,int(stretch_point_TF/2):]

            tf_load_cycle_n = tf_load.shape[0]

            tf_stretch_cond[cycle_cond_i:(cycle_cond_i+tf_load_cycle_n),:,:] = tf_load

            cycle_cond_i += tf_load_cycle_n

        del tf_load

        #### MA
        os.chdir(path_memmap)
        if monopol:
            pixel_based_distrib = np.memmap(f'{ROI}_{cond}_pixel_distrib_inter.dat', dtype=np.float32, mode='w+', shape=(nfrex, 2))
        else:
            pixel_based_distrib = np.memmap(f'{ROI}_{cond}_pixel_distrib_inter_bi.dat', dtype=np.float32, mode='w+', shape=(nfrex, 2))
        
        #### define ncycle
        n_cycle_baselines = tf_stretch_baselines.shape[0]
        n_cycle_cond = tf_stretch_cond.shape[0]

        #### space allocation
        _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
        pixel_based_distrib_i = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
        tf_shuffle = np.zeros((n_cycle_cond, nfrex, int(stretch_point_TF/2)))

        #surrogates_i = 0
        for surrogates_i in range(n_surrogates_tf):

            #### random selection
            draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
            sel_baseline = np.random.randint(low=0, high=n_cycle_baselines, size=(draw_indicator == 1).sum())
            sel_cond = np.random.randint(low=0, high=n_cycle_cond, size=(draw_indicator == 0).sum())

            #### extract max min
            tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[sel_baseline, :, :]
            tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[sel_cond, :, :]

            _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
            
            pixel_based_distrib_i[:, surrogates_i, 0] = _min
            pixel_based_distrib_i[:, surrogates_i, 1] = _max

        min, max = np.median(pixel_based_distrib_i[:,:,0], axis=1), np.median(pixel_based_distrib_i[:,:,1], axis=1) 

        if debug:

            tf_nchan = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

            time = np.arange(tf_nchan.shape[-1])

            plt.pcolormesh(time, frex, tf_nchan, shading='gouraud', cmap='seismic')
            plt.contour(time, frex, get_tf_stats(tf_nchan, min, max), levels=0, colors='g')
            plt.yscale('log')
            plt.show()

            #wavelet_i = 0
            for wavelet_i in range(20):
                count, _, _ = plt.hist(tf_nchan[wavelet_i, :], bins=500)
                plt.vlines([min[wavelet_i], max[wavelet_i]], ymin=0, ymax=count.max(), color='r')
                plt.show()

        pixel_based_distrib[:,0] = min
        pixel_based_distrib[:,1] = max        

        #### plot 
        if debug:

            median_max_diff = np.abs(np.median(tf_stretch_cond, axis=1).reshape(-1) - np.median(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))).max()
            vmin = -median_max_diff
            vmax = median_max_diff

            for nchan, nchan_name in enumerate(chan_list_ieeg):

                tf_plot = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

                time = np.arange(tf_plot.shape[-1])

                plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
                plt.contour(time, frex, get_tf_stats(tf_plot, min, max), levels=0, colors='g', vmin=vmin, vmax=vmax)
                plt.yscale('log')
                plt.yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
                plt.title(nchan_name)
                plt.show()


        ######## SAVE ########

        print(f'SAVE', flush=True)

        os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

        if monopol:
            np.save(f'allsujet_{ROI}_tf_STATS_{cond}_inter.npy', pixel_based_distrib)
        else:
            np.save(f'allsujet_{ROI}_tf_STATS_{cond}_inter_bi.npy', pixel_based_distrib)    

        del tf_stretch_cond
        
        #### remove memmap
        os.chdir(path_memmap)
        try:
            if monopol:
                os.remove(f'{ROI}_{cond}_pixel_distrib_inter.dat')
            else:
                os.remove(f'{ROI}_{cond}_pixel_distrib_inter_bi.dat')
            del pixel_based_distrib
        except:
            pass

        try:
            if monopol:
                os.remove(f'{ROI}_{cond}_tf_STATS_cond_{monopol}.dat')
            else:
                os.remove(f'{ROI}_{cond}_tf_STATS_cond_{monopol}_bi.dat')
            del tf_stretch_cond
        except:
            pass

    #### remove baseline
    os.chdir(path_memmap)
    try:
        if monopol:
            os.remove(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}.dat')
        else:
            os.remove(f'{ROI}_{cond}_tf_STATS_baseline_{monopol}_bi.dat')
        del tf_stretch_baselines
    except:
        pass
    


                    
                
            
                

    




########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #monopol = True
    for monopol in [True, False]:

        #### load anat
        ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', monopol) 

        #ROI = ROI_to_include[5]
        for ROI in ROI_to_include:

            # precompute_tf_ROI_STATS(ROI, cond, monopol)
            execute_function_in_slurm_bash_mem_choice('n12bis_precompute_allplot_TF_STATS', 'precompute_tf_ROI_STATS', [ROI, monopol], '30G')

    
    



 