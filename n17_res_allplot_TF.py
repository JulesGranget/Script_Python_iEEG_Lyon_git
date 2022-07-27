
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *

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


def get_ROI_Lobes_list_and_Plots(cond):

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
    sujet_list_selected = []
    for sujet_i in sujet_list_FR_CV:
        prms_i = get_params(sujet_i)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### search for ROI & lobe that have been counted
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')

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










########################################
######## COMPUTE TF FOR COND ######## 
########################################


def robust_zscore(data):
    
    _median = np.median(data) 
    MAD = np.median(np.abs(data-np.median(data)))
    data_zscore = (0.6745*(data-_median))/ MAD
        
    return data_zscore


#struct_name, cond, mat_type, anat_type = ROI_name, 'RD_CV', 'TF', 'ROI'
def open_TForITPC_data(struct_name, cond, mat_type, anat_type):
    
    #### open file
    os.chdir(os.path.join(path_precompute, 'allplot'))
    
    listdir = os.listdir()
    file_to_open = []
    [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1]

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
    for sujet_i in sujet_list_FR_CV:
        prms_i = get_params(sujet_i)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### search for ROI & lobe that have been counted
    n_count = 0
    
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')

        if anat_type == 'ROI':
            n_count += np.sum(plot_loca_df['localisation_corrected'] == struct_name)
        if anat_type == 'Lobe':
            n_count += np.sum(plot_loca_df['lobes_corrected'] == struct_name)

    return struct_xr, n_count





def robust_zscore(data):
    
    _median = np.median(data) 
    MAD = np.median(np.abs(data-np.median(data)))
    data_zscore = (0.6745*(data-_median))/ MAD
        
    return data_zscore



#ROI_name, mat_type = 'isthme', 'TF'
def compute_for_one_ROI_allcond(ROI_name, mat_type, cond_to_compute):

    print(ROI_name)

    #### params
    anat_type = 'ROI'

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    allcond_TF = {}
    allcond_count = {}
    #cond = 'FR_CV'
    for cond in cond_to_compute:

        allcond_TF[cond], allcond_count[cond] = open_TForITPC_data(ROI_name, cond, mat_type, anat_type)

    #### keep only cond that have plot
    cond_to_plot = [cond for cond in cond_to_compute if allcond_count[cond] != 0]

    if len(cond_to_plot) == 0:
        return

    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'ROI'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'ROI'))

    #### plot
    # band_prep_i, band_prep = 1, 'hf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### extract band to plot
        freq_band = freq_band_dict[band_prep]

        #### initiate fig
        fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(cond_to_plot))

        fig.set_figheight(10)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        if band_prep == 'hf':
            keys_list_reversed = list(freq_band.keys())
            keys_list_reversed.reverse()
            freq_band_reversed = {}
            for key_i in keys_list_reversed:
                freq_band_reversed[key_i] = freq_band[key_i]
            freq_band = freq_band_reversed

        plt.suptitle(ROI_name)

        #cond_i, cond = 0, 'FR_CV'
        for c, cond in enumerate(cond_to_plot):

            stretch_point = stretch_point_TF
            time = range(stretch_point)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for r, (band, freq) in enumerate(list(freq_band.items())) :

                TF_i = allcond_TF[cond].loc[band, :, :].data
                TF_count_i = allcond_count[cond]
                frex = np.linspace(freq[0], freq[1], TF_i.shape[0])
                
                if len(cond_to_plot) == 1:
                    ax = axs[r]
                    if r == 0 :
                        ax.set_title(f' {cond} : {TF_count_i}')
                    if c == 0:
                        ax.set_ylabel(band)
                    ax.pcolormesh(time, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))
                    ax.vlines(ratio_stretch_TF*stretch_point, ymin=freq[0], ymax=freq[1], colors='g')

                else:
                    ax = axs[r, c]
                    if r == 0 :
                        ax.set_title(f' {cond} : {TF_count_i}')
                    if c == 0:
                        ax.set_ylabel(band)
                    ax.pcolormesh(time, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                    ax.vlines(ratio_stretch_TF*stretch_point, ymin=freq[0], ymax=freq[1], colors='g')

        
        #plt.show()
                    
        #### save
        if band_prep == 'lf':
            fig.savefig(ROI_name + '_allcond_lf.jpeg', dpi=150)
        if band_prep == 'hf':
            fig.savefig(ROI_name + '_allcond_hf.jpeg', dpi=150)
        plt.close('all')









#Lobe_name, mat_type = 'Cingular', 'TF'
def compute_for_one_Lobe_allcond(Lobe_name, mat_type, cond_to_compute):

    print(Lobe_name)

    #### params
    anat_type = 'Lobe'

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    allcond_TF = {}
    allcond_count = {}
    #cond = 'FR_CV'
    for cond in cond_to_compute:

        allcond_TF[cond], allcond_count[cond] = open_TForITPC_data(Lobe_name, cond, mat_type, anat_type)

    #### keep only cond that have plot
    cond_to_plot = [cond for cond in cond_to_compute if allcond_count[cond] != 0]

    if len(cond_to_plot) == 0:
        return
    
    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'Lobes'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'Lobes'))

    #### plot
    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### extract band to plot
        freq_band = freq_band_dict[band_prep]

        #### initiate fig
        fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(cond_to_plot))

        fig.set_figheight(10)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        if band_prep == 'hf':
            keys_list_reversed = list(freq_band.keys())
            keys_list_reversed.reverse()
            freq_band_reversed = {}
            for key_i in keys_list_reversed:
                freq_band_reversed[key_i] = freq_band[key_i]
            freq_band = freq_band_reversed

        plt.suptitle(Lobe_name)

        #cond_i, cond = 0, 'FR_CV'
        for c, cond in enumerate(cond_to_plot):

            stretch_point = stretch_point_TF
            time = range(stretch_point)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for r, (band, freq) in enumerate(list(freq_band.items())) :

                TF_i = allcond_TF[cond].loc[band, :, :].data
                TF_count_i = allcond_count[cond]
                frex = np.linspace(freq[0], freq[1], TF_i.shape[0])
                
                if len(cond_to_plot) == 1:
                    ax = axs[r]
                    if r == 0 :
                        ax.set_title(f' {cond} : {TF_count_i}')
                    if c == 0:
                        ax.set_ylabel(band)
                    ax.pcolormesh(time, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))
                    ax.vlines(ratio_stretch_TF*stretch_point, ymin=freq[0], ymax=freq[1], colors='g')

                else:
                    ax = axs[r, c]
                    if r == 0 :
                        ax.set_title(f' {cond} : {TF_count_i}')
                    if c == 0:
                        ax.set_ylabel(band)
                    ax.pcolormesh(time, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                    ax.vlines(ratio_stretch_TF*stretch_point, ymin=freq[0], ymax=freq[1], colors='g')

        #plt.show()

        #### save
        if band_prep == 'lf':
            fig.savefig(Lobe_name + '_allcond_lf.jpeg', dpi=150)
        if band_prep == 'hf':
            fig.savefig(Lobe_name + '_allcond_hf.jpeg', dpi=150)
        plt.close()





################################
######## COMPILATION ########
################################

def compilation_slurm(anat_type, mat_type):

    #cond = 'RD_CV'
    cond_to_compute = ['RD_CV', 'RD_FV', 'RD_SV', 'FR_CV']

    print(f'{anat_type} {mat_type}')

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond_to_compute[0])

    if anat_type == 'ROI':

        # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_ROI_allcond)(ROI_i, mat_type) for ROI_i in ROI_to_include)
        #ROI_name = ROI_list[10]
        for ROI_name in ROI_list:
            compute_for_one_ROI_allcond(ROI_name, mat_type, cond_to_compute)

    if anat_type == 'Lobe':

        # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_Lobe_allcond)(Lobe_i, mat_type) for Lobe_i in lobe_to_include)
        for Lobe_name in lobe_list:
            compute_for_one_Lobe_allcond(Lobe_name, mat_type, cond_to_compute)




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #anat_type = 'ROI'
    for anat_type in ['ROI', 'Lobe']:
    
        #mat_type = 'TF'
        for mat_type in ['TF', 'ITPC']:
            
            #compilation_slurm(anat_type, mat_type)
            execute_function_in_slurm_bash('n17_res_allplot_TF', 'compilation_slurm', [anat_type, mat_type])




