
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False












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



def precompute_tf_STATS(sujet, monopol):

    #### params
    if sujet not in sujet_list:

        conditions = ['FR_CV']

    else:

        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    ######## INTRA ########
    print(f'#### COMPUTE TF STATS INTRA {sujet} ####', flush=True)

    #cond = conditions[0]
    for cond in conditions:

        #### identify if already computed for all
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        if monopol:
            if os.path.exists(f'{sujet}_tf_STATS_{cond}_intra.npy'):
                print(f'{cond} ALREADY COMPUTED')
                continue
        else:
            if os.path.exists(f'{sujet}_tf_STATS_{cond}_intra_bi.npy'):
                print(f'{cond} ALREADY COMPUTED')
                continue

        ######## FOR INSPI ########
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))
        if monopol:
            tf_stretch_baselines = np.load(f'{sujet}_tf_conv_{cond}.npy', mmap_mode='r')[:,:,:,:int(stretch_point_TF/2)]
        else:
            tf_stretch_baselines = np.load(f'{sujet}_tf_conv_{cond}_bi.npy', mmap_mode='r')[:,:,:,:int(stretch_point_TF/2)]
            
        ######## FOR EXPI ########
        if monopol:
            tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}.npy', mmap_mode='r')[:,:,:,int(stretch_point_TF/2):]
        else:
            tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}_bi.npy', mmap_mode='r')[:,:,:,int(stretch_point_TF/2):]

        #### verif tf
        if debug:

            nchan = 0
            plt.pcolormesh(np.median(tf_stretch_baselines[nchan,:,:,:], axis=0))
            plt.show()

            plt.pcolormesh(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))
            plt.show()

            print(f'COMPUTE {cond}', flush=True)

        #### MA
        os.chdir(path_memmap)
        if monopol:
            pixel_based_distrib = np.memmap(f'{sujet}_{cond}_pixel_distrib_intra.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), nfrex, 2))
        else:
            pixel_based_distrib = np.memmap(f'{sujet}_{cond}_pixel_distrib_intra_bi.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), nfrex, 2))
        
        #nchan = 0
        def get_min_max_pixel_based_distrib(nchan):

            print_advancement(nchan, len(chan_list_ieeg), steps=[25, 50, 75])

            #### define ncycle
            n_cycle_baselines = tf_stretch_baselines.shape[1]
            n_cycle_cond = tf_stretch_cond.shape[1]

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
                tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[nchan, sel_baseline, :, :]
                tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[nchan, sel_cond, :, :]

                _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
                
                pixel_based_distrib_i[:, surrogates_i, 0] = _min
                pixel_based_distrib_i[:, surrogates_i, 1] = _max

                gc.collect()

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

            pixel_based_distrib[nchan,:,0] = min
            pixel_based_distrib[nchan,:,1] = max

            del min, max, tf_shuffle
        
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_min_max_pixel_based_distrib)(nchan) for nchan, _ in enumerate(chan_list_ieeg))

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

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        if monopol:
            np.save(f'{sujet}_tf_STATS_{cond}_intra.npy', pixel_based_distrib)
        else:
            np.save(f'{sujet}_tf_STATS_{cond}_intra_bi.npy', pixel_based_distrib)    

        del tf_stretch_cond
        
        os.chdir(path_memmap)
        try:
            if monopol:
                os.remove(f'{sujet}_{cond}_pixel_distrib_intra.dat')
            else:
                os.remove(f'{sujet}_{cond}_pixel_distrib_intra_bi.dat')
            del pixel_based_distrib
        except:
            pass

    #### remove baseline
    del tf_stretch_baselines








    #### pass subject that only have FR_CV
    if sujet not in sujet_list:

        return

    ######## INTER ########
    print(f'#### COMPUTE TF STATS INTER {sujet} ####', flush=True)

    ######## FOR BASELINE ########
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if monopol:
        tf_stretch_baselines = np.load(f'{sujet}_tf_conv_FR_CV.npy', mmap_mode='r')
    else:
        tf_stretch_baselines = np.load(f'{sujet}_tf_conv_FR_CV_bi.npy', mmap_mode='r')

    #cond = conditions[0]
    for cond in conditions:

        if cond == 'FR_CV':
            continue

        #### identify if already computed for all
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        if monopol:
            if os.path.exists(f'{sujet}_tf_STATS_{cond}_inter.npy'):
                print(f'{cond} ALREADY COMPUTED', flush=True)
                continue
        else:
            if os.path.exists(f'{sujet}_tf_STATS_{cond}_inter_bi.npy'):
                print(f'{cond} ALREADY COMPUTED', flush=True)
                continue
            
        ######## FOR COND ########
        if monopol:
            tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}.npy', mmap_mode='r')
        else:
            tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}_bi.npy', mmap_mode='r')

        #### verif tf
        if debug:

            nchan = 0
            plt.pcolormesh(np.median(tf_stretch_baselines[nchan,:,:,:], axis=0))
            plt.show()

            plt.pcolormesh(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))
            plt.show()

            print(f'COMPUTE {cond}', flush=True)

        #### MA
        os.chdir(path_memmap)
        if monopol:
            pixel_based_distrib = np.memmap(f'{sujet}_{cond}_pixel_distrib_inter.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), nfrex, 2))
        else:
            pixel_based_distrib = np.memmap(f'{sujet}_{cond}_pixel_distrib_inter_bi.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), nfrex, 2))
        
        #### compute
        #nchan = 0
        def get_min_max_pixel_based_distrib(nchan):

            print_advancement(nchan, len(chan_list_ieeg), steps=[25, 50, 75])

            #### define ncycle
            n_cycle_baselines = tf_stretch_baselines.shape[1]
            n_cycle_cond = tf_stretch_cond.shape[1]

            #### space allocation
            _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
            pixel_based_distrib_i = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
            tf_shuffle = np.zeros((n_cycle_cond, nfrex, stretch_point_TF))

            #surrogates_i = 0
            for surrogates_i in range(n_surrogates_tf):

                #### random selection
                draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
                sel_baseline = np.random.randint(low=0, high=n_cycle_baselines, size=(draw_indicator == 1).sum())
                sel_cond = np.random.randint(low=0, high=n_cycle_cond, size=(draw_indicator == 0).sum())

                #### extract max min
                tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[nchan, sel_baseline, :, :]
                tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[nchan, sel_cond, :, :]

                _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
                
                pixel_based_distrib_i[:, surrogates_i, 0] = _min
                pixel_based_distrib_i[:, surrogates_i, 1] = _max

                gc.collect()

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

            pixel_based_distrib[nchan,:,0] = min
            pixel_based_distrib[nchan,:,1] = max

            del min, max, tf_shuffle
        
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_min_max_pixel_based_distrib)(nchan) for nchan, _ in enumerate(chan_list_ieeg))

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

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        if monopol:
            np.save(f'{sujet}_tf_STATS_{cond}_inter.npy', pixel_based_distrib)
        else:
            np.save(f'{sujet}_tf_STATS_{cond}_inter_bi.npy', pixel_based_distrib)    

        del tf_stretch_cond
        
        os.chdir(path_memmap)
        try:
            if monopol:
                os.remove(f'{sujet}_{cond}_pixel_distrib_inter.dat')
            else:
                os.remove(f'{sujet}_{cond}_pixel_distrib_inter_bi.dat')
            del pixel_based_distrib
        except:
            pass

    #### remove baseline
    del tf_stretch_baselines













########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list_FR_CV[0]
    for sujet in sujet_list_FR_CV:   

        #monopol = True
        for monopol in [True, False]:
    
            # precompute_tf_STATS(sujet, monopol)
            execute_function_in_slurm_bash_mem_choice('n7bis_precompute_TF_STATS', 'precompute_tf_STATS', [sujet, monopol], '30G')

        







