
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib
import seaborn as sns

from n0_config import *
from n0bis_analysis_functions import *


debug = False


def count_all_plot_location():

    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')
    
    #### fill dict with anat names
    anat_loca_dict = {}
    anat_lobe_dict = {}
    anat_loca_list = nomenclature_df['Our correspondances'].values
    anat_lobe_list_non_sorted = nomenclature_df['Lobes'].values
    for i in range(len(anat_loca_list)):
        anat_loca_dict[anat_loca_list[i]] = 0
        anat_lobe_dict[anat_lobe_list_non_sorted[i]] = 0

    anat_ROI_noselect_dict = anat_loca_dict.copy()
    anat_lobe_noselect_dict = anat_lobe_dict.copy()

    #### search for all subjects
    #sujet_i = sujet_list[0]
    for sujet_i in sujet_list:

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

            loca_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            anat_loca_dict[loca_tmp] = anat_loca_dict[loca_tmp] + 1
            anat_lobe_dict[lobe_tmp] = anat_lobe_dict[lobe_tmp] + 1
            count_verif += 1

        #### verif count
        if count_verif != len(chan_list_ieeg):
            print('ERROR : anatomical count is not correct, count != len chan_list')
            exit()

    #### for all plot, i. e. not included
    os.chdir(os.path.join(path_anatomy))
    df_all_plot_noselect = pd.read_excel('plot_loca_all.xlsx')

    for i in df_all_plot_noselect.index.values:

        ROI_tmp = df_all_plot_noselect['localisation_corrected'][i]
        lobe_tmp = df_all_plot_noselect['lobes_corrected'][i]
        
        anat_ROI_noselect_dict[ROI_tmp] = anat_ROI_noselect_dict[ROI_tmp] + 1
        anat_lobe_noselect_dict[lobe_tmp] = anat_lobe_noselect_dict[lobe_tmp] + 1

    df_data_ROI = {'ROI' : list(anat_loca_dict.keys()), 'ROI_Count_No_Included' : list(anat_ROI_noselect_dict.values()), 'ROI_Count_Included' : list(anat_loca_dict.values())}
    df_data_Lobes = {'Lobes' : list(anat_lobe_dict.keys()), 'Lobes_Count_No_Included' : list(anat_lobe_noselect_dict.values()), 'Lobes_Count_Included' : list(anat_lobe_dict.values())}

    df_ROI_count = pd.DataFrame(df_data_ROI)
    df_lobes_count = pd.DataFrame(df_data_Lobes)

    #### save df
    os.chdir(os.path.join(path_results, 'all_plot'))

    if os.path.exists('ROI_count.xlsx'):
        os.remove('ROI_count.xlsx')

    if os.path.exists('Lobes_count.xlsx'):
        os.remove('Lobes_count.xlsx')  

    df_ROI_count.to_excel('ROI_count.xlsx')
    df_lobes_count.to_excel('Lobes_count.xlsx')

    #### save fig
    sns.catplot(x="ROI_Count_Included", y="ROI", kind='bar', palette="pastel", edgecolor=".6", data=df_ROI_count, height=10, aspect=1)
    plt.savefig('ROI_count_included.png', dpi=600)
    plt.close()
    
    sns.catplot(x="Lobes_Count_Included", y="Lobes", kind='bar', palette="pastel", edgecolor=".6", data=df_lobes_count, height=10, aspect=1)
    plt.savefig('Lobes_Count_Included.png', dpi=600)
    plt.close()

    sns.catplot(x="ROI_Count_No_Included", y="ROI", kind='bar', palette="pastel", edgecolor=".6", data=df_ROI_count, height=10, aspect=1)
    plt.savefig('ROI_Count_No_Included.png', dpi=600)
    plt.close()
    
    sns.catplot(x="Lobes_Count_No_Included", y="Lobes", kind='bar', palette="pastel", edgecolor=".6", data=df_lobes_count, height=10, aspect=1)
    plt.savefig('Lobes_Count_No_Included.png', dpi=600)
    plt.close()
    
    








def get_ROI_Lobes_list_and_Plots():

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

    #### search for ROI & lobe that have been counted
    #sujet_i = sujet_list[0]
    for sujet_i in sujet_list:

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

    return ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots



ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()

#### load all plot
os.chdir(os.path.join(path_anatomy))
df_all_plot_noselect = pd.read_excel('plot_loca_all.xlsx')
all_proccessed_plot = []
for list_i in list(ROI_dict_plots.values()):
    for i in range(len(list_i)):
        all_proccessed_plot.append(list_i[i])



os.chdir(path_memmap)
PxxRespi_Cxy_p = np.memmap('PxxRespi_Cxy_p.dat', dtype='float64', mode='w+', shape=(len(df_all_plot_noselect.index.values),4))
















def get_Coh_Respi_1plot(plot_i_to_process):

    print(plot_i_to_process)
    
    if plot_i_to_process/len(df_all_plot_noselect.index.values) % .2 <= .01:
        print('{:.2f}'.format(plot_i_to_process/len(df_all_plot_noselect.index.values)))

    #### identify if proccessed
    if (df_all_plot_noselect['subject'][plot_i_to_process] + '_' + df_all_plot_noselect['plot'][plot_i_to_process] in all_proccessed_plot) != True:
        return

    sujet_tmp = df_all_plot_noselect['subject'][plot_i_to_process]
    plot_tmp_mod = df_all_plot_noselect['plot'][plot_i_to_process]

    #### load subject params
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects)
    cond = 'FR_CV'
    band_prep = 'lf'

    #### load Cxy params
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    #### identify plot name in trc
    if sujet_tmp[:3] != 'pat':
        list_mod, list_trc = modify_name(chan_list_ieeg)
        plot_tmp = list_trc[list_mod.index(plot_tmp_mod)]
    else:
        plot_tmp = plot_tmp_mod

    #### identify session
    os.chdir(os.path.join(path_prep, sujet_tmp, 'sections'))
    listdir_file = os.listdir()
    file_to_load = [listdir_i for listdir_i in listdir_file if listdir_i.find(cond) != -1 and listdir_i.find(band_prep) != -1]
    session_count = len(file_to_load)

    #### compute Coh for all session
    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    ######
    if plot_i_to_process == 624:
        print('top') 
    ######


    Cxy_for_cond = np.zeros((session_count, len(hzCxy)))
    Pxx_respi = np.zeros((session_count, len(hzCxy)))

    for session_i in range(session_count):

        respi_chan_i = chan_list.index('nasal')
        plot_tmp_i = chan_list.index(plot_tmp)
        respi = load_data_sujet(sujet_tmp, band_prep, cond, session_i)[respi_chan_i,:]
        data_tmp = load_data_sujet(sujet_tmp, band_prep, cond, session_i)

        x = data_tmp[plot_tmp_i,:]
        y = respi
        hzPxx, Pxx = scipy.signal.welch(y,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

        Cxy_for_cond[session_i, :] = Cxy[mask_hzCxy]
        Pxx_respi[session_i, :] = Pxx[mask_hzCxy]

    #### load surrogates
    Cxy_surrogates = np.zeros((session_count, len(hzCxy)))
    os.chdir(os.path.join(path_precompute, sujet_tmp, 'PSD_Coh'))

    ######
    if plot_i_to_process == 624:
        print('top') 
    ######


    for session_i in range(session_count):

        data_load = np.load(sujet_tmp + '_' + cond + '_' + str(1) + '_Coh.npy')
        plot_tmp_i = chan_list.index(plot_tmp)
        Cxy_surrogates[session_i,:] = data_load[plot_tmp_i,:]


    #### reduce all sessions
    Pxx_respi_reduced = np.zeros((len(hzCxy)))
    Cxy_reduced = np.zeros((len(hzCxy)))
    Cxy_surrogates_reduced = np.zeros((len(hzCxy)))

    if session_count == 1:
        Pxx_respi_reduced = Pxx_respi[0,:]
        Cxy_reduced = Cxy_for_cond[0,:]
        Cxy_surrogates_reduced = Cxy_surrogates[0,:]

    else:

        for session_i in range(session_count):

            if session_i == 0:
                Pxx_respi_reduced = Pxx_respi[session_i,:]
                Cxy_reduced = Cxy_for_cond[session_i,:]
                Cxy_surrogates_reduced = Cxy_surrogates[session_i,:]

            else:
                Pxx_respi_reduced = (Pxx_respi_reduced + Pxx_respi[session_i,:])/2
                Cxy_reduced = (Cxy_reduced + Cxy_for_cond[session_i,:])/2
                Cxy_surrogates_reduced = (Cxy_surrogates_reduced + Cxy_surrogates[session_i,:])/2

    ######
    if plot_i_to_process == 624:
        print('top') 
    ######

    
    max_Cxy = hzCxy[np.argmax(Cxy_reduced)]
    max_respi = hzCxy[np.argmax(Pxx_respi_reduced)]
    Cxy_value = Cxy_reduced[np.argmax(Cxy_reduced)]
    if Cxy_reduced[np.argmax(Cxy_reduced)] >= Cxy_surrogates_reduced[np.argmax(Cxy_reduced)]:
        Cxy_significant = 1
    else:
        Cxy_significant = 0

    ######
    if plot_i_to_process == 624:
        print('top') 
    ######


    #### load into memmap
    PxxRespi_Cxy_p[plot_i_to_process,0] = max_respi
    PxxRespi_Cxy_p[plot_i_to_process,1] = max_Cxy
    PxxRespi_Cxy_p[plot_i_to_process,2] = Cxy_value
    PxxRespi_Cxy_p[plot_i_to_process,3] = Cxy_significant


#### compute CxyRespi all plot
joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_Coh_Respi_1plot)(plot_i_to_process) for plot_i_to_process in range(len(df_all_plot_noselect.index.values)))

data_df = {'sujet' : df_all_plot_noselect['subject'].values, 'plot' : df_all_plot_noselect['plot'].values, 'max_respi' : PxxRespi_Cxy_p[:,0], 'max_Cxy' : PxxRespi_Cxy_p[:,1], 'Cxy_value' : PxxRespi_Cxy_p[:,2], 'Cxy_significant' : PxxRespi_Cxy_p[:,3]} 

df_CxyRespi = pd.DataFrame(data_df)

size_marker = []
for plot_i_to_process in range(len(df_all_plot_noselect.index.values)):
    df_CxyRespi

#### plot CxyRespi
x = df_CxyRespi['max_respi'].values
y = df_CxyRespi['max_Cxy'].values
colors = df_CxyRespi['Cxy_significant'].values
area = df_CxyRespi['Cxy_value'].values * 100

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

#### save
os.chdir(os.path.join(path_results, 'all_plot'))
df_CxyRespi.to_excel('CxyRespi_all_plot.xlsx')

os.chdir(path_memmap)
os.remove('PxxRespi_Cxy_p.dat')






# ROI_to_process = ROI_list[1]
def get_CycleFreq_for_ROI(ROI_to_process):

    if plot_i_to_process/len(df_all_plot_noselect.index.values) % .2 <= .01:
        print('{:.2f}'.format(plot_i_to_process/len(df_all_plot_noselect.index.values)))

    #### identify if proccessed
    if (ROI_to_process in ROI_to_include) != True:
        return

    #### plot to compute
    plot_to_process = ROI_dict_plots[ROI_to_process]

    #### generate dict for loading TF
    dict_TF_for_ROI_to_process = {}
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_lf, stretch_point_TF))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_hf, stretch_point_TF))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### compute TF
    # plot_to_process_i = plot_to_process[0]    
    for plot_to_process_i in plot_to_process:
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        #### load subject params
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects)
        cond = 'FR_CV'

        #### identify plot name in trc
        if sujet_tmp[:3] != 'pat':
            list_mod, list_trc = modify_name(chan_list_ieeg)
            plot_tmp = list_trc[list_mod.index(plot_tmp_mod)]
        else:
            plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### count session number
        os.chdir(os.path.join(path_prep, sujet_tmp, 'sections'))
        listdir_file = os.listdir()
        file_to_load = [listdir_i for listdir_i in listdir_file if listdir_i.find(cond) != -1 and listdir_i.find('lf') != -1]
        session_count = len(file_to_load)

        #### load TF
        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))
        for band, freq in dict_freq_band.items():

            for session_i in range(session_count):
                
                TF_load = np.load(sujet_tmp + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i + 1) + '.npy')

                dict_TF_for_ROI_to_process[band] = (dict_TF_for_ROI_to_process[band] + TF_load[plot_tmp_i,:,:])/2

    #### plot
    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        if band_prep == 'lf':
            fig, axs = plt.subplots(nrows=4, ncols=1)
        if band_prep == 'hf':
            fig, axs = plt.subplots(nrows=2, ncols=1)
        
        plt.suptitle(ROI_to_process)
        dict_freq_to_plot = freq_band_list[band_prep_i]
                        
        # i, (band, freq) = 0, ('theta', [2 ,10])
        for i, (band, freq) in enumerate(list(dict_freq_to_plot.items())) :

            data = dict_TF_for_ROI_to_process[band]
            frex = np.linspace(freq[0], freq[1], data.shape[0])
            time = np.arange(stretch_point_TF)
        
            if i == 0 :

                ax = axs[i]
                ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data), shading='auto')
                ax.set_ylabel(band)
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                #plt.show()

            else :

                ax = axs[i]
                ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data), shading='auto')
                ax.set_ylabel(band)
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                #plt.show()
                          
        #### save
        if band_prep == 'lf':
            fig.savefig(ROI_to_process + '_lf.jpeg', dpi=600)
        if band_prep == 'hf':
            fig.savefig(ROI_to_process + '_hf.jpeg', dpi=600)
        plt.close()

            




