
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import seaborn as sns
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n10_res_allplot_analysis import ROI_list


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




################################################
######## XARRAY FOR EVERY SUBJECTS ########
################################################

#### open all plot
def identify_plot_to_compute_for_sujet(sujet_i):

    plot_to_compute = []

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

    #nchan = chan_list_ieeg_csv[0]
    for i, nchan in enumerate(chan_list_ieeg_csv):

        loca_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
        lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
        plot_tmp = chan_list_ieeg[i]

        plot_to_compute.append([sujet_i, plot_tmp, loca_tmp, lobe_tmp])

    return plot_to_compute






#sujet_tmp, plot_tmp, cond = 'GOBc', "B'10", 'RD_CV'
def compute_Pxx_Cxy_for_one_plot(sujet_tmp, plot_tmp, cond):

    #### load Cxy params
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects)
    band_prep = 'lf'
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    #### load data
    os.chdir(os.path.join(path_prep, sujet_tmp, 'sections'))
    listdir_file = os.listdir()
    file_to_load = [listdir_i for listdir_i in listdir_file if listdir_i.find(cond) != -1 and listdir_i.find(band_prep) != -1]
    session_count = len(file_to_load)

    #### compute Coh for all session
    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    Cxy_for_cond = np.zeros((session_count, len(hzCxy)))
    Pxx_for_cond = np.zeros((session_count, len(hzPxx)))

    Cxy_surrogates = np.zeros((session_count, len(hzCxy)))
    os.chdir(os.path.join(path_precompute, sujet_tmp, 'PSD_Coh'))

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
        Pxx_for_cond[session_i, :] = Pxx

        #### surrogates
        data_load = np.load(sujet_tmp + '_' + cond + '_' + str(1) + '_Coh.npy')
        plot_tmp_i = chan_list.index(plot_tmp)
        Cxy_surrogates[session_i,:] = data_load[plot_tmp_i,:]

    #### reduce all sessions
    Cxy_for_cond = np.mean(Cxy_for_cond, axis=0)
    Pxx_for_cond = np.mean(Pxx_for_cond, axis=0)
    Cxy_surrogates = np.mean(Cxy_surrogates, axis=0)

    #### adjust for srate
    if srate != 512 :
        nwind, nfft, noverlap, hannw = get_params_spectral_analysis(512)
        hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
        Pxx_for_cond = Pxx_for_cond[:len(hzPxx)]

    return Pxx_for_cond, Cxy_for_cond, Cxy_surrogates




#sujet_i = sujet_list[0]
def compute_PxxCxy_all_cond_for_1_subject(sujet_i):

    print('#### ' + sujet_i)

    #### verif computation
    os.chdir(os.path.join(path_precompute, sujet_i, 'PSD_Coh'))
    if os.path.exists(os.path.join(path_precompute, sujet_i, 'PSD_Coh', f'{sujet_i}_xr_Pxx.nc')) and os.path.exists(os.path.join(path_precompute, sujet_i, 'PSD_Coh', f'{sujet_i}_xr_Cxy.nc')):
        print('ALREADY COMPUTED')
        return 
    
    #### load subject params
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_i, conditions_allsubjects)
    if srate != 512:
        srate = 512
    band_prep = 'lf'

    plot_to_compute = identify_plot_to_compute_for_sujet(sujet_i)

    ROI_list = [plot_to_compute_i[2] for plot_to_compute_i in plot_to_compute]

    #### matrix preparation
    #### load Cxy params
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)
    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    Pxx_sujet_tmp = np.zeros((len(conditions), len(plot_to_compute), len(hzPxx)))
    Cxy_sujet_tmp = np.zeros((len(conditions), len(plot_to_compute), 2, len(hzCxy)))

    for cond in conditions:

        print(cond)

        PxxCxy_cond_tmp = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_Pxx_Cxy_for_one_plot)(sujet_i, plot_tmp_info[1], cond) for plot_tmp_info in plot_to_compute)

        # reorganize result
        for i in range(len(PxxCxy_cond_tmp)):
            Pxx_sujet_tmp[conditions.index(cond), i,:] = PxxCxy_cond_tmp[i][0]
            Cxy_sujet_tmp[conditions.index(cond), i, 0, :] = PxxCxy_cond_tmp[i][1]
            Cxy_sujet_tmp[conditions.index(cond), i, 1, :] = PxxCxy_cond_tmp[i][2]
            
        del PxxCxy_cond_tmp

    #### generate xarray

    data_Pxx = Pxx_sujet_tmp
    data_Cxy = Cxy_sujet_tmp

    dims_Pxx = ['cond', 'ROI', 'freq']
    dims_Cxy = ['cond', 'ROI', 'data', 'freq']
    coords_Pxx = {'cond':conditions, 'ROI':ROI_list, 'freq':hzPxx}
    coords_Cxy = {'cond':conditions, 'ROI':ROI_list, 'data':['Cxy', 'surrogates'], 'freq':hzCxy}
    name = f'xr_PxxCxy_{sujet_i}'

    xr_Pxx = xr.DataArray(data=data_Pxx, dims=dims_Pxx, coords=coords_Pxx)
    xr_Cxy = xr.DataArray(data=data_Cxy, dims=dims_Cxy, coords=coords_Cxy)

    #### save
    os.chdir(os.path.join(path_precompute, sujet_i, 'PSD_Coh'))
    xr_Pxx.to_netcdf(f'{sujet_i}_xr_Pxx.nc')
    xr_Cxy.to_netcdf(f'{sujet_i}_xr_Cxy.nc')







################################################
######## XARRAY FOR ALL SUBJECTS ########
################################################

def load_allsubject_array():

    #### load data
    #data_type = 'Pxx'
    for data_type in ['Pxx', 'Cxy']:

        if data_type == 'Pxx':

            sujet_i = sujet_list[0]
            os.chdir(os.path.join(path_precompute, sujet_i, 'PSD_Coh'))
            xr_Pxx_allsubject = xr.open_dataset(sujet_i + '_xr_Pxx.nc')

            #sujet_i = 'GOBc'
            for sujet_i in sujet_list[1:]:

                os.chdir(os.path.join(path_precompute, sujet_i, 'PSD_Coh'))
                xr_tmp = xr.open_dataset(sujet_i + '_xr_Pxx.nc')

                xr_Pxx_allsubject = xr.concat([xr_Pxx_allsubject, xr_tmp], 'ROI')

        if data_type == 'Cxy':

            sujet_i = sujet_list[0]
            os.chdir(os.path.join(path_precompute, sujet_i, 'PSD_Coh'))
            xr_Cxy_allsubject = xr.open_dataset(sujet_i + '_xr_Cxy.nc')

            #sujet_i = 'GOBc'
            for sujet_i in sujet_list[1:]:

                os.chdir(os.path.join(path_precompute, sujet_i, 'PSD_Coh'))
                xr_tmp = xr.open_dataset(sujet_i + '_xr_Cxy.nc')

                xr_Cxy_allsubject = xr.concat([xr_Cxy_allsubject, xr_tmp], 'ROI')

    xr_Pxx_allsubject = xr_Pxx_allsubject.to_array()
    xr_Cxy_allsubject = xr_Cxy_allsubject.to_array()

    return xr_Pxx_allsubject, xr_Cxy_allsubject

        










################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #### compute xarray
    for sujet_i in sujet_list:

        compute_PxxCxy_all_cond_for_1_subject(sujet_i)

    #### xarray all sujet   
    xr_Pxx_allsubject, xr_Cxy_allsubject = load_allsubject_array()

    #### plot

    anat_loca_dict, anat_lobe_dict = get_all_ROI_and_Lobes_name()
    ROI_list = list(anat_loca_dict.keys())
    ROI_list = xr_Pxx_allsubject.groupby('ROI').mean('ROI')['ROI'].data


    #### Cxy 

    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'PSD_Coh'))

    xr_Cxy_groupby = xr_Cxy_allsubject.groupby('ROI').mean('ROI')

    #ROI = ROI_list[0]
    for ROI in ROI_list:

        fig, axs = plt.subplots(ncols=6)

        for cond_i, cond in enumerate(conditions_allsubjects):

            xr_Cxy_groupby.sel(cond=cond, data='Cxy', ROI=ROI).plot(label='Cxy', ax=axs[cond_i], ylim=(0,1))
            xr_Cxy_groupby.sel(cond=cond, data='surrogates', ROI=ROI).plot(label='surrogates', ax=axs[cond_i], ylim=(0,1))
            ax = axs[cond_i]
            ax.set_title(cond)

        plt.suptitle(ROI)
        plt.legend()
        fig.set_figwidth(15)
        #plt.show()

        fig.savefig(ROI + '_Cxy.jpeg')

        plt.close('all')

    #### Pxx

    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'PSD_Coh'))

    xr_Pxx_groupby = xr_Pxx_allsubject.groupby('ROI').mean('ROI')

    band_to_analyze = {'theta' : [4, 8], 'alpha' : [8, 12], 'beta' : [12, 30]}

    #band, freq = 'theta', [4, 8]
    for ROI_i_i, ROI_i in enumerate(ROI_list):

        df_band_dict = {'cond' : [], 'band' : [], 'Pxx' : np.array(())}
            
        for cond_i, cond in enumerate(conditions_allsubjects):

            Pxx_raw = []

            for band_i, (band, freq) in enumerate(band_to_analyze.items()):

                Pxx_tmp = xr_Pxx_groupby.sel(cond=cond, ROI=ROI_i, freq=slice(freq[0], freq[1])).data
                Pxx_raw.append(Pxx_tmp[0])

            #### normalize
            for band_i, (band, freq) in enumerate(band_to_analyze.items()):

                Pxx_fill = Pxx_raw[band_i] / np.mean(Pxx_raw[-1])
                df_band_dict['Pxx'] = np.append(df_band_dict['Pxx'], Pxx_fill)
                [df_band_dict['cond'].append(cond) for i in range(Pxx_fill.shape[0])]
                [df_band_dict['band'].append(band) for i in range(Pxx_fill.shape[0])]

        df_band = pd.DataFrame(df_band_dict, columns=['Pxx', 'cond', 'band'])

        fig = plt.figure()
        
        sns.barplot(data=df_band, x="cond", y='Pxx', hue="band")
        plt.suptitle(ROI_i + '\n norm beta band')
        #plt.show()

        fig.savefig(ROI_i + '_Pxx.jpeg')

        plt.close('all')











