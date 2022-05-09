
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from n0_config import *
from n0bis_analysis_functions import *

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


#cond = 'FR_CV'
def get_ROI_Lobes_list_and_Plots(cond):

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = list(nomenclature_df['Our correspondances'].values)
    lobe_list = []
    [lobe_list.append(lobe_i) for lobe_i in nomenclature_df['Lobes'].values if (lobe_i in lobe_list) == False]

    #### fill dict with anat names
    ROI_dict = {}
    ROI_dict_plots = {}
    lobe_dict = {}
    lobe_dict_plots = {}
    anat_lobe_list_non_sorted = nomenclature_df['Lobes'].values
    for i in range(len(ROI_list)):
        ROI_dict[ROI_list[i]] = 0
        ROI_dict_plots[ROI_list[i]] = []
        lobe_dict[anat_lobe_list_non_sorted[i]] = 0
        lobe_dict_plots[anat_lobe_list_non_sorted[i]] = []

    #### initiate for cond
    sujet_for_cond = []

    #### search for ROI & lobe that have been counted

    if cond == 'FR_CV' :
        sujet_list_selected = sujet_list_FR_CV
    else:
        sujet_list_selected = sujet_list

    #sujet_i = sujet_list_selected[0]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_prep, sujet_i, 'sections'))
        session_count_sujet_i = []
        for file_i in os.listdir():
            if file_i.find(cond) != -1:
                session_count_sujet_i.append(file_i)
            else:
                continue
        if len(session_count_sujet_i) == 0:
            continue
        else:
            sujet_for_cond.append(sujet_i)

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')

        chan_list_txt = open(sujet_i + '_chanlist_ieeg.txt', 'r')
        chan_list_txt_readlines = chan_list_txt.readlines()
        chan_list_ieeg = [i.replace('\n', '') for i in chan_list_txt_readlines]

        #### exclude Paris subjects
        if sujet_i[:3] == 'pat':
            chan_list_ieeg_csv = chan_list_ieeg
        else:
            chan_list_ieeg_csv, trash = modify_name(chan_list_ieeg)

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
            print('ERROR : anatomical count is not correct, count != len chan_list')
            exit()

    ROI_to_include = [ROI_i for ROI_i in ROI_list if ROI_dict[ROI_i] > 0]
    lobe_to_include = [Lobe_i for Lobe_i in lobe_list if lobe_dict[Lobe_i] > 0]

    return sujet_for_cond, ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots




########################################
######## COMPUTE TF FOR COND ######## 
########################################




#cond, sujet_i, plot_i = 'FR_CV', 'pat_02495_0949', 'AmT2_2'
def open_TForITPC_data(cond, sujet_i, plot_i, mat_type):

    #### open file
    if mat_type == 'TF':
        os.chdir(os.path.join(path_precompute, sujet_i, 'TF'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_precompute, sujet_i, 'ITPC'))
    listdir = os.listdir()
    file_to_open = []
    [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1]

    #### identify plot
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_i, conditions_allsubjects)
    if sujet_i[:3] != 'pat':
        chan_list_csv, trash = modify_name(chan_list)
    else:
        chan_list_csv = chan_list
    plot_index = chan_list_csv.index(plot_i)

    #### number of sessions
    n_session = []
    for file_i in file_to_open:
        n_session.append(int(file_i[-5:-4]))
    n_session = np.max(n_session)

    #### load matrix
    if n_session == 1:
        TF_mat = []
        for file_i in file_to_open:
            mat = np.load(file_i)
            mat_plot_i = mat[plot_index, :, :]
            TF_mat.append(mat_plot_i)

    else:
        TF_mat = []
        TF_mat_count = []
        for session_i in range(n_session):

            file_session_to_open = []
            [file_session_to_open.append(file_i) for file_i in file_to_open if int(file_i[-5:-4]) == session_i+1]

            if session_i == 0:
                for file_i in file_session_to_open:
                    mat = np.load(file_i)
                    mat_plot_i = mat[plot_index, :, :]
                    TF_mat.append(mat_plot_i)
                    TF_mat_count.append(1)
            
            else:
                #i, file_i = 0, file_session_to_open[0]
                for i, file_i in enumerate(file_session_to_open):
                    mat = np.load(file_i)
                    mat_plot_i = mat[plot_index, :, :]
                    TF_mat[i] = (TF_mat[i] + mat_plot_i)
                    TF_mat_count[i] += 1

        #### mean
        for i, _ in enumerate(TF_mat):
            TF_mat[i] /= TF_mat_count[i]

    #### verif
    #fig, axs = plt.subplots(ncols=len(TF_mat))
    #for i in range(len(TF_mat)):
    #    ax = axs[i]
    #    ax.pcolormesh(TF_mat[i])
    #plt.show()
                
    return TF_mat








#ROI_i, mat_type = 'amygdala', 'TF'
def compute_for_one_ROI_allcond(ROI_i, mat_type):

    print(ROI_i)

    ROI_dict_to_open = {}
    for cond in conditions_allsubjects:
        ROI_dict_to_open[cond] = []

    #cond = 'FR_CV'
    for cond in conditions_allsubjects:
        sujet_for_cond, ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond)
        ROI_dict_to_open[cond] = ROI_dict_plots[ROI_i]

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    #### generate dict
    ROI_mat_dict = {}
    ROI_mat_dict_count = {}
    for cond in conditions_allsubjects:
        ROI_mat_dict[cond] = {}
        ROI_mat_dict_count[cond] = {}
        
        for band_i, freq_i in zip(band_names, freq_values):
            if nfrex_hf == nfrex_lf:
                ROI_mat_dict[cond][band_i] = np.zeros((nfrex_hf, stretch_point_TF))
                ROI_mat_dict_count[cond][band_i] = 1
            else:
                raise ValueError('nfrex_hf != nfrex_lf')

    #### load mat
    for cond in conditions_allsubjects:

        #sujet_i, plot_i = 'GOBc', 'Ap01'
        for sujet_i, plot_i in ROI_dict_to_open[cond]:
            
            if len(ROI_dict_to_open[cond]) == 1:
                TF_mat = open_TForITPC_data(cond, sujet_i, plot_i, mat_type)
                for band_i, band in enumerate(band_names):
                    ROI_mat_dict[cond][band] = TF_mat[band_i]

            else:
                
                if sujet_i == ROI_dict_to_open[cond][0][0] and plot_i == ROI_dict_to_open[cond][0][1]:

                    TF_mat = open_TForITPC_data(cond, sujet_i, plot_i, mat_type)
                    for band_i, band in enumerate(band_names):
                        ROI_mat_dict[cond][band] = TF_mat[band_i]

                else:

                    TF_mat = open_TForITPC_data(cond, sujet_i, plot_i, mat_type)
                    for band_i, band in enumerate(band_names):
                        mat_to_add = (ROI_mat_dict[cond][band] + TF_mat[band_i])
                        ROI_mat_dict[cond][band] = mat_to_add
                        ROI_mat_dict_count[cond][band] += 1

    for cond in conditions_allsubjects:
        for band_i, band in enumerate(band_names):
            ROI_mat_dict[cond][band] /= ROI_mat_dict_count[cond][band]

    #### verif matrix 
    #fig, axs = plt.subplots(nrows=len(conditions_allsubjects), ncols=len(band_names))
    #for cond_i, cond in enumerate(conditions_allsubjects):
    #    for band_i, band in enumerate(band_names):
    #        ax = axs[band_i, cond_i]
    #        ax.pcolormesh(ROI_mat_dict[cond][band])
    #plt.show()

    #### compute number of plot for each cond
    cond_count = {}
    for cond in list(ROI_dict_to_open.keys()):
        str_cond_count = cond_count[cond] = len(ROI_dict_to_open[cond])
    
    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'ROI'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'ROI'))

    #### find scale
    scales_lf = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}
                        
    for cond in conditions_allsubjects:
        # i, (band, freq) = 0, ('theta', [2 ,10])

        for band in band_names[:4] :

            if band == 'whole' or band == 'l_gamma' :
                continue

            data = ROI_mat_dict[cond][band]

            scales_lf['vmin_val'] = np.append(scales_lf['vmin_val'], np.min(data))
            scales_lf['vmax_val'] = np.append(scales_lf['vmax_val'], np.max(data))
            scales_lf['median_val'] = np.append(scales_lf['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales_lf['vmin_val']) - np.median(scales_lf['median_val'])), np.abs(np.max(scales_lf['vmax_val']) - np.median(scales_lf['median_val']))])

    vmin_lf = np.median(scales_lf['median_val']) - median_diff
    vmax_lf = np.median(scales_lf['median_val']) + median_diff

    scales_hf = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}
                        
    for cond in conditions_allsubjects:
        # i, (band, freq) = 0, ('theta', [2 ,10])

        if cond == 'FR_CV':
            continue

        for band in band_names[4:] :

            if band == 'whole' or band == 'l_gamma' :
                continue

            data = ROI_mat_dict[cond][band]

            scales_hf['vmin_val'] = np.append(scales_hf['vmin_val'], np.min(data))
            scales_hf['vmax_val'] = np.append(scales_hf['vmax_val'], np.max(data))
            scales_hf['median_val'] = np.append(scales_hf['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales_hf['vmin_val']) - np.median(scales_hf['median_val'])), np.abs(np.max(scales_hf['vmax_val']) - np.median(scales_hf['median_val']))])

    vmin_hf = np.median(scales_hf['median_val']) - median_diff
    vmax_hf = np.median(scales_hf['median_val']) + median_diff

    del scales_lf, scales_hf

    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        if band_prep == 'lf':
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions_allsubjects))
            band_to_plot = band_names[:4]
        if band_prep == 'hf':
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions_allsubjects))
            band_to_plot = band_names[4:]

        plt.suptitle(ROI_i)
        dict_freq_to_plot = freq_band_list[band_prep_i]

        for cond_i, cond in enumerate(conditions_allsubjects):
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for i, (band, freq) in enumerate(list(dict_freq_to_plot.items())) :

                data = ROI_mat_dict[cond][band]

                frex = np.linspace(freq[0], freq[1], data.shape[0])
                time = np.arange(stretch_point_TF)
            
                ax = axs[i, cond_i]
                if i == 0 :
                    ax.set_title(cond + f' : {cond_count[cond]}')
                if cond_i == 0:
                    ax.set_ylabel(band)
                if band_prep == 'lf':
                    ax.pcolormesh(time, frex, data, vmin=vmin_lf, vmax=vmax_lf, shading='gouraud', cmap=plt.get_cmap('seismic'))
                if band_prep == 'hf':
                    ax.pcolormesh(time, frex, data, vmin=vmin_hf, vmax=vmax_hf, shading='gouraud', cmap=plt.get_cmap('seismic'))
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
                #plt.show()
                    
        #### save
        if band_prep == 'lf':
            fig.savefig(ROI_i + '_all_lf.jpeg', dpi=600)
        if band_prep == 'hf':
            fig.savefig(ROI_i + '_all_hf.jpeg', dpi=600)
        plt.close()









#Lobe_i = 'Frontal'
def compute_for_one_Lobe_allcond(Lobe_i, mat_type):

    print(Lobe_i)

    Lobe_dict_to_open = {}
    for cond in conditions_allsubjects:
        Lobe_dict_to_open[cond] = []

    #cond = 'FR_CV'
    for cond in conditions_allsubjects:
        sujet_for_cond, ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond)
        Lobe_dict_to_open[cond] = lobe_dict_plots[Lobe_i]

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    #### generate dict
    Lobe_mat_dict = {}
    Lobe_mat_dict_count = {}
    for cond in conditions_allsubjects:
        Lobe_mat_dict[cond] = {}
        Lobe_mat_dict_count[cond] = {}
        
        for band_i, freq_i in zip(band_names, freq_values):
            if nfrex_hf == nfrex_lf:
                Lobe_mat_dict[cond][band_i] = np.zeros((nfrex_hf, stretch_point_TF))
                Lobe_mat_dict_count[cond][band_i] = 1
            else:
                raise ValueError('nfrex_hf != nfrex_lf')

    #### load mat
    for cond in conditions_allsubjects:

        #sujet_i, plot_i = 'GOBc', 'Ap01'
        for sujet_i, plot_i in Lobe_dict_to_open[cond]:
            
            if len(Lobe_dict_to_open[cond]) == 1:
                TF_mat = open_TForITPC_data(cond, sujet_i, plot_i, mat_type)
                for band_i, band in enumerate(band_names):
                    Lobe_mat_dict[cond][band] = TF_mat[band_i]

            else:
                
                if sujet_i == Lobe_dict_to_open[cond][0][0] and plot_i == Lobe_dict_to_open[cond][0][1]:

                    TF_mat = open_TForITPC_data(cond, sujet_i, plot_i, mat_type)
                    for band_i, band in enumerate(band_names):
                        Lobe_mat_dict[cond][band] = TF_mat[band_i]

                else:

                    TF_mat = open_TForITPC_data(cond, sujet_i, plot_i, mat_type)
                    for band_i, band in enumerate(band_names):
                        mat_to_add = (Lobe_mat_dict[cond][band] + TF_mat[band_i])
                        Lobe_mat_dict[cond][band] = mat_to_add
                        Lobe_mat_dict_count[cond][band] += 1

    for cond in conditions_allsubjects:
        for band_i, band in enumerate(band_names):
            Lobe_mat_dict[cond][band] /= Lobe_mat_dict_count[cond][band]

    #### verif matrix 
    #fig, axs = plt.subplots(nrows=len(conditions_allsubjects), ncols=len(band_names))
    #for cond_i, cond in enumerate(conditions_allsubjects):
    #    for band_i, band in enumerate(band_names):
    #        ax = axs[band_i, cond_i]
    #        ax.pcolormesh(Lobe_mat_dict[cond][band])
    #plt.show()

    #### compute number of plot for each cond
    cond_count = {}
    for cond in list(Lobe_dict_to_open.keys()):
        str_cond_count = cond_count[cond] = len(Lobe_dict_to_open[cond])
    
    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'Lobes'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'Lobes'))

    #### find scale
    scales_lf = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}
                        
    for cond in conditions_allsubjects:
        # i, (band, freq) = 0, ('theta', [2 ,10])

        for band in band_names[:4] :

            if band == 'whole' or band == 'l_gamma' :
                continue

            data = Lobe_mat_dict[cond][band]

            scales_lf['vmin_val'] = np.append(scales_lf['vmin_val'], np.min(data))
            scales_lf['vmax_val'] = np.append(scales_lf['vmax_val'], np.max(data))
            scales_lf['median_val'] = np.append(scales_lf['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales_lf['vmin_val']) - np.median(scales_lf['median_val'])), np.abs(np.max(scales_lf['vmax_val']) - np.median(scales_lf['median_val']))])

    vmin_lf = np.median(scales_lf['median_val']) - median_diff
    vmax_lf = np.median(scales_lf['median_val']) + median_diff

    scales_hf = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}
                        
    for cond in conditions_allsubjects:
        # i, (band, freq) = 0, ('theta', [2 ,10])

        if cond == 'FR_CV':
            continue

        for band in band_names[4:] :

            if band == 'whole' or band == 'l_gamma' :
                continue

            data = Lobe_mat_dict[cond][band]

            scales_hf['vmin_val'] = np.append(scales_hf['vmin_val'], np.min(data))
            scales_hf['vmax_val'] = np.append(scales_hf['vmax_val'], np.max(data))
            scales_hf['median_val'] = np.append(scales_hf['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales_hf['vmin_val']) - np.median(scales_hf['median_val'])), np.abs(np.max(scales_hf['vmax_val']) - np.median(scales_hf['median_val']))])

    vmin_hf = np.median(scales_hf['median_val']) - median_diff
    vmax_hf = np.median(scales_hf['median_val']) + median_diff

    del scales_lf, scales_hf

    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        if band_prep == 'lf':
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions_allsubjects))
            band_to_plot = band_names[:4]
        if band_prep == 'hf':
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions_allsubjects))
            band_to_plot = band_names[4:]

        plt.suptitle(Lobe_i)
        dict_freq_to_plot = freq_band_list[band_prep_i]

        for cond_i, cond in enumerate(conditions_allsubjects):
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for i, (band, freq) in enumerate(list(dict_freq_to_plot.items())) :

                data = Lobe_mat_dict[cond][band]

                frex = np.linspace(freq[0], freq[1], data.shape[0])
                time = np.arange(stretch_point_TF)
            
                ax = axs[i, cond_i]
                if i == 0 :
                    ax.set_title(cond + f' : {cond_count[cond]}')
                if cond_i == 0:
                    ax.set_ylabel(band)
                if band_prep == 'lf':
                    ax.pcolormesh(time, frex, data, vmin=vmin_lf, vmax=vmax_lf, shading='gouraud', cmap=plt.get_cmap('seismic'))
                if band_prep == 'hf':
                    ax.pcolormesh(time, frex, data, vmin=vmin_hf, vmax=vmax_hf, shading='gouraud', cmap=plt.get_cmap('seismic'))
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
                #plt.show()
                    
        #### save
        if band_prep == 'lf':
            fig.savefig(Lobe_i + '_all_lf.jpeg', dpi=600)
        if band_prep == 'hf':
            fig.savefig(Lobe_i + '_all_hf.jpeg', dpi=600)
        plt.close()






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    anat_loca_dict, anat_lobe_dict = get_all_ROI_and_Lobes_name()
    ROI_list = list(anat_loca_dict.keys())
    Lobe_list = list(anat_lobe_dict.keys())

    #### compute ROI
    for mat_type in ['TF', 'ITPC']:
        print(f'#### {mat_type} ####')
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_ROI_allcond)(ROI_i, mat_type) for ROI_i in ROI_list)

    #### compute ROI
    for mat_type in ['TF', 'ITPC']:
        print(f'#### {mat_type} ####')
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_Lobe_allcond)(Lobe_i, mat_type) for Lobe_i in Lobe_list)






