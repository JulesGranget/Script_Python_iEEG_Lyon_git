
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import xarray as xr
import cv2

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False



########################################
######## ALLPLOT ANATOMY ######## 
########################################

def get_all_ROI_and_Lobes_name():

    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')
    
    #### fill dict with anat names
    anat_loca_dict = {}
    anat_lobe_dict = {}
    anat_loca_list = nomenclature_df['Our correspondances'].values
    anat_lobe_list_non_sorted = nomenclature_df['Lobes'].values
    for i in range(len(anat_loca_list)):
        anat_loca_dict[anat_loca_list[i]] = {'TF' : {}, 'ITPC' : {}}
        anat_lobe_dict[anat_lobe_list_non_sorted[i]] = {'TF' : {}, 'ITPC' : {}}

    return anat_loca_dict, anat_lobe_dict



########################################
######## PREP ALLPLOT ANALYSIS ########
########################################



def get_ROI_Lobes_list_and_Plots(cond, monopol):

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = list(np.unique(nomenclature_df['Our correspondances'].values))
    lobe_list = list(np.unique(nomenclature_df['Lobes'].values))

    #### fill dict with anat names
    ROI_dict = {}
    ROI_dict_plots = {}

    for ROI_i in ROI_list:
        ROI_dict[ROI_i] = 0
        ROI_dict_plots[ROI_i] = []

    lobe_dict = {}
    lobe_dict_plots = {}

    for lobe_i in lobe_list:
        lobe_dict[lobe_i] = 0
        lobe_dict_plots[lobe_i] = []

    #### search for ROI & lobe that have been counted
    if cond == 'FR_CV':
        sujet_list_selected = sujet_list_FR_CV
    else:
        sujet_list_selected = sujet_list

    #sujet_i = sujet_list_selected[0]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        if monopol:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        else:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')
            
        chan_list_ieeg = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            ROI_dict[ROI_tmp] = ROI_dict[ROI_tmp] + 1
            lobe_dict[lobe_tmp] = lobe_dict[lobe_tmp] + 1
            count_verif += 1

            ROI_dict_plots[ROI_tmp].append([sujet_i, nchan])
            lobe_dict_plots[lobe_tmp].append([sujet_i, nchan])

        #### verif count
        if count_verif != len(chan_list_ieeg):
            raise ValueError('ERROR : anatomical count is not correct, count != len chan_list')

    ROI_to_include = [ROI_i for ROI_i in ROI_list if ROI_dict[ROI_i] > 0]
    lobe_to_include = [Lobe_i for Lobe_i in lobe_list if lobe_dict[Lobe_i] > 0]

    return ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots









########################
######## STATS ########
########################



#tf_plot = data_allcond[cond].values
def get_tf_stats(tf_plot, pixel_based_distrib, nfrex, stats_type):

    tf_thresh = np.zeros(tf_plot.shape)

    if stats_type == 'inter':
            
        #wavelet_i = 0
        for wavelet_i in range(nfrex):
            mask = np.logical_or(tf_plot[wavelet_i, :] <= pixel_based_distrib[wavelet_i, 0], tf_plot[wavelet_i, :] >= pixel_based_distrib[wavelet_i, 1])
            tf_thresh[wavelet_i, mask] = 1
            tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    if stats_type == 'intra':
            
        #wavelet_i = 0
        for wavelet_i in range(nfrex):
            mask = np.logical_or(tf_plot[wavelet_i, :] <= pixel_based_distrib[wavelet_i, 0], tf_plot[wavelet_i, :] >= pixel_based_distrib[wavelet_i, 1])
            tf_thresh[wavelet_i, mask] = 1
            tf_thresh[wavelet_i, np.logical_not(mask)] = 0

        tf_thresh[:, :int(tf_plot.shape[-1]/2)] = 0

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### if empty return
    if tf_thresh.sum() == 0:

        return tf_thresh

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf_plot.shape[-1])

        plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh











########################################
######## COMPUTE TF FOR COND ######## 
########################################


def robust_zscore(data):
    
    _median = np.median(data) 
    MAD = np.median(np.abs(data-np.median(data)))
    data_zscore = (0.6745*(data-_median))/ MAD
        
    return data_zscore


#struct_name, cond, mat_type, anat_type = ROI_name, 'FR_CV', 'TF', 'ROI'
def open_TForITPC_data(struct_name, cond, mat_type, anat_type, monopol):
    
    #### open file
    os.chdir(os.path.join(path_precompute, 'allplot'))
    
    listdir = os.listdir()
    file_to_open = []

    if monopol:
        [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1 and file_i.find('bi') == -1]
    else:
        [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1 and file_i.find('bi') != -1]

    #### extract band names
    band_names = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]

    #### load matrix
    xr_TF = xr.open_dataarray(file_to_open[0])
    try:
        struct_xr = xr_TF.loc[struct_name, :, mat_type, :, :]
    except:
        print(f'{struct_name} {cond} not found')
        return 0, 0

    #### identify plot number
    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i, monopol)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### search for ROI & lobe that have been counted
    n_count = 0
    
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))

        if monopol:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        else:
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')

        if anat_type == 'ROI':
            n_count += np.sum(plot_loca_df['localisation_corrected'] == struct_name)
        if anat_type == 'Lobes':
            n_count += np.sum(plot_loca_df['lobes_corrected'] == struct_name)

    return struct_xr, n_count







#ROI_i, ROI = ROI_to_include.index('amygdala'), 'amygdala'
def compute_for_one_ROI_allcond(ROI_i, ROI, monopol):

    print(ROI, monopol)

    #### count ROI
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', monopol)
    ROI_count_FR_CV = len(ROI_dict_plots[ROI])

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('RD_SV', monopol)
    ROI_count_allcond = len(ROI_dict_plots[ROI])

    #### select cond
    if ROI_count_allcond == 0:
        conditions = ['FR_CV']
    else:
        conditions = ['FR_CV', 'RD_CV', 'RD_SV', 'RD_FV']

    #### scale
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))
    
    data_allcond = {}
    #cond = conditions[0]
    for cond in conditions:

        if monopol:
            data_allcond[cond] = xr.open_dataarray(f'allsujet_{cond}_ROI.nc').loc[ROI,:,:]
        else:
            data_allcond[cond] = xr.open_dataarray(f'allsujet_{cond}_ROI_bi.nc').loc[ROI,:,:]

    vals = np.array([])

    #cond = cond_to_plot[0]
    for cond in conditions:

        vals = np.append(vals, data_allcond[cond].values.reshape(-1))

    median_diff = np.percentile(np.abs(vals - np.median(vals)), 100-tf_plot_percentile_scale)

    vmin = np.median(vals) - median_diff
    vmax = np.median(vals) + median_diff

    del vals

    #stats_type = 'intra'
    for stats_type in ['inter', 'intra']:

        #### plot 
        fig, axs = plt.subplots(ncols=len(conditions))

        if monopol:
            plt.suptitle(f'{ROI}, stats:{stats_type}')
        else:
            plt.suptitle(f'{ROI}_bi, stats:{stats_type}')

        fig.set_figheight(5)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        #c, cond = 1, cond_to_plot[1]
        for c, cond in enumerate(conditions):

            if len(conditions) == 1:
                ax = axs
            else:
                ax = axs[c]

            if cond == 'FR_CV':
                ax.set_title(f'{cond}, n:{ROI_count_FR_CV}', fontweight='bold', rotation=0)
            else:
                ax.set_title(f'{cond}, n:{ROI_count_allcond}', fontweight='bold', rotation=0)
                
            #### generate time vec
            time_vec = np.arange(stretch_point_TF)

            #### plot
            ax.pcolormesh(time_vec, frex, data_allcond[cond].values, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.set_yscale('log')

            #### stats
            os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))
            if stats_type == 'inter' and cond != 'FR_CV':
                if monopol:
                    pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}_inter.npy')
                else:
                    pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}_inter_bi.npy')

                if get_tf_stats(data_allcond[cond].values, pixel_based_distrib, nfrex, stats_type).sum() != 0:
                    ax.contour(time_vec, frex, get_tf_stats(data_allcond[cond].values, pixel_based_distrib, nfrex, stats_type), levels=0, colors='g')

            if stats_type == 'intra':
                if monopol:
                    pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}_intra.npy')
                else:
                    pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}_intra_bi.npy')

                if get_tf_stats(data_allcond[cond].values, pixel_based_distrib, nfrex, stats_type).sum() != 0:
                    ax.contour(time_vec, frex, get_tf_stats(data_allcond[cond].values, pixel_based_distrib, nfrex, stats_type), levels=0, colors='g')

            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')

            ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

        #plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF'))

        if monopol:
            fig.savefig(f'{ROI}_{stats_type}.jpeg', dpi=150)
        else:
            fig.savefig(f'{ROI}_{stats_type}_bi.jpeg', dpi=150)

        fig.clf()
        plt.close('all')













# #Lobe_name, mat_type = 'Temporal', 'TF'
# def compute_for_one_Lobe_allcond(Lobe_name, mat_type, cond_to_compute, srate, monopol):

#     print(Lobe_name)

#     #### params
#     anat_type = 'Lobes'

#     #### extract band names
#     band_names = []
#     freq_values = []
#     for band_freq_i in freq_band_list:
#         [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
#         [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

#     #### get stats
#     STATS_for_Lobe_to_process = {}

#     for cond in cond_to_compute:

#         if cond != 'FR_CV':

#             STATS_for_Lobe_to_process[cond] = get_STATS_for_Lobes(Lobe_name, cond, monopol)

#     allcond_TF = {}
#     allcond_count = {}
#     #cond = 'FR_CV'
#     for cond in cond_to_compute:

#         allcond_TF[cond], allcond_count[cond] = open_TForITPC_data(Lobe_name, cond, mat_type, anat_type, monopol)

#     #### plot & save
#     if mat_type == 'TF':
#         os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'Lobes'))
#     if mat_type == 'ITPC':
#         os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'Lobes'))

#     #### plot
#     # band_prep_i, band_prep = 0, 'lf'
#     for band_prep_i, band_prep in enumerate(band_prep_list):

#         #### extract band to plot
#         freq_band = freq_band_dict[band_prep]

#         #### initiate fig
#         fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(cond_to_compute))

#         fig.set_figheight(10)
#         fig.set_figwidth(15)

#         #### for plotting l_gamma down
#         if band_prep == 'hf':
#             keys_list_reversed = list(freq_band.keys())
#             keys_list_reversed.reverse()
#             freq_band_reversed = {}
#             for key_i in keys_list_reversed:
#                 freq_band_reversed[key_i] = freq_band[key_i]
#             freq_band = freq_band_reversed

#         if monopol:
#             plt.suptitle(Lobe_name)
#         else:
#             plt.suptitle(f'{Lobe_name} bi')

#         #cond_i, cond = 0, 'FR_CV'
#         for c, cond in enumerate(cond_to_compute):

#             #### generate time vec
#             time_vec = np.arange(stretch_point_TF)
                        
#             # i, (band, freq) = 0, ('theta', [2 ,10])
#             for r, (band, freq) in enumerate(list(freq_band.items())) :

#                 TF_i = allcond_TF[cond].loc[band, :, :].data
#                 TF_count_i = allcond_count[cond]
#                 frex = np.linspace(freq[0], freq[1], TF_i.shape[0])
                
#                 ax = axs[r, c]
#                 if r == 0 :
#                     ax.set_title(f' {cond} : {TF_count_i}')
#                 if c == 0:
#                     ax.set_ylabel(band)
                    
#                 ax.pcolormesh(time_vec, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

#                 if cond != 'FR_CV':
#                     # ax.contour(time_vec, frex, STATS_for_Lobe_to_process[band], levels=0, colors='g')
#                     ax.contour(time_vec, frex, STATS_for_Lobe_to_process[cond][band])

#                 ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')

#         # plt.show()
                    
#         #### save
#         if monopol:
#             fig.savefig(f'{Lobe_name}_allcond_{band_prep}.jpeg', dpi=150)
#         else:
#             fig.savefig(f'{Lobe_name}_allcond_{band_prep}_bi.jpeg', dpi=150)

#         plt.close('all')






################################
######## COMPILATION ########
################################

def compilation_slurm(anat_type, mat_type, monopol):

    print(f'#### {anat_type} {mat_type} ####')

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', monopol)

    if anat_type == 'ROI':

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_ROI_allcond)(ROI_i, ROI, monopol) for ROI_i, ROI in enumerate(ROI_to_include))
        # for ROI_i in ROI_to_include:
        #     compute_for_one_ROI_allcond(ROI_i, mat_type, cond_to_compute, srate)

    # if anat_type == 'Lobes':

    #     joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_Lobe_allcond)(Lobe_i, mat_type, cond_to_compute, srate, monopol) for Lobe_i, Lobe in enumerate(lobe_to_include))
        # for Lobe_i in lobe_to_include:
        #     compute_for_one_Lobe_allcond(Lobe_i, mat_type, cond_to_compute, srate)






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #monopol = True
    for monopol in [True, False]:

        anat_type = 'ROI'
        mat_type = 'TF'
        
        compilation_slurm(anat_type, mat_type, monopol)
        # execute_function_in_slurm_bash('n16_res_allplot_TF', 'compilation_slurm', [anat_type, mat_type, monopol])




