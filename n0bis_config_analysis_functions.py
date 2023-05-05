

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import sys
import stat
import subprocess
import physio
import xarray as xr


from bycycle.cyclepoints import find_extrema
import neurokit2 as nk

from n0_config_params import *


debug = False








########################################
######## GENERATE FOLDERS ########
########################################


#os.getcwd()
def create_folder(folder_name, construct_token):
    if os.path.exists(folder_name) == False:
        os.mkdir(folder_name)
        print('create : ' + folder_name)
        construct_token += 1
    return construct_token

def generate_folder_structure(sujet):

    construct_token = 0

    os.chdir(path_general)
    
    construct_token = create_folder('Analyses', construct_token)
    construct_token = create_folder('Data', construct_token)
    construct_token = create_folder('Mmap', construct_token)

    #### Analyses
    os.chdir(os.path.join(path_general, 'Analyses'))
    construct_token = create_folder('preprocessing', construct_token)
    construct_token = create_folder('precompute', construct_token)
    construct_token = create_folder('anatomy', construct_token)
    construct_token = create_folder('results', construct_token)
    construct_token = create_folder('protocole', construct_token)
    
        #### preprocessing
    os.chdir(os.path.join(path_general, 'Analyses', 'preprocessing'))
    construct_token = create_folder(sujet, construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'preprocessing', sujet))
    construct_token = create_folder('sections', construct_token)
    construct_token = create_folder('info', construct_token)

        #### precompute
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute'))
    construct_token = create_folder(sujet, construct_token)
    construct_token = create_folder('allplot', construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute', sujet))
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('baselines', construct_token)
    construct_token = create_folder('FC', construct_token)

        #### anatomy
    os.chdir(os.path.join(path_general, 'Analyses', 'anatomy'))
    construct_token = create_folder(sujet, construct_token)

        #### results
    os.chdir(os.path.join(path_general, 'Analyses', 'results'))
    construct_token = create_folder(sujet, construct_token)
    construct_token = create_folder('allplot', construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet))
    construct_token = create_folder('RESPI', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('FC', construct_token)
    construct_token = create_folder('HRV', construct_token)
    construct_token = create_folder('df', construct_token)

            #### TF
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'TF'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

            #### PSD_Coh
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'PSD_Coh'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

            #### ITPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'ITPC'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

            #### FC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'FC'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)
    construct_token = create_folder('verif', construct_token)

        #### allplot
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot'))
    construct_token = create_folder('allcond', construct_token)
    construct_token = create_folder('FR_CV', construct_token)
    construct_token = create_folder('anatomy', construct_token)
    construct_token = create_folder('df', construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond'))
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('FC', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)

            #### FR_CV
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'FR_CV'))
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('stats', construct_token)

            #### TF
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'FR_CV', 'TF'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)            
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond', 'TF'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)

            #### PSD_Coh
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'FR_CV', 'PSD_Coh'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)            
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond', 'PSD_Coh'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)

            #### ITPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'FR_CV', 'ITPC'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)            
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond', 'ITPC'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)

            #### FC          
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond', 'FC'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token) 

    #### Data
    os.chdir(os.path.join(path_general, 'Data'))
    construct_token = create_folder('raw_data', construct_token)

        #### raw_data
    os.chdir(os.path.join(path_general, 'Data', 'raw_data'))    
    construct_token = create_folder(sujet, construct_token)
    
            #### anatomy
    os.chdir(os.path.join(path_general, 'Data', 'raw_data', sujet))    
    construct_token = create_folder('anatomy', construct_token)

    return construct_token













################################
######## SLURM EXECUTE ########
################################


#name_script, name_function, params = 'test', 'slurm_test',  ['Pilote', 2]
def execute_function_in_slurm(name_script, name_function, params):

    python = sys.executable

    #### params to print in script
    params_str = ""
    for params_i in params:
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if params_i == params[0] :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for params_i in params:

        str_i = str(params_i)

        if params_i == params[0] :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run_function_{name_function}_{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
        
    subprocess.Popen(['sbatch', f'{slurm_script_name}', f'-cpus-per-task={n_core_slurms}', f'-mem={mem_crnl_cluster}']) 

    # wait subprocess to lauch before removing
    #time.sleep(3)
    #os.remove(slurm_script_name)

    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')






#name_script, name_function, params = 'n7_precompute_TF', 'precompute_tf', [cond, session_i, freq_band_list, band_prep_list]
def execute_function_in_slurm_bash(name_script, name_function, params):

    scritp_path = os.getcwd()
    
    python = sys.executable

    #### params to print in script
    params_str = ""
    for i, params_i in enumerate(params):
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if i == 0 :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for i, params_i in enumerate(params):

        str_i = str(params_i)

        if i == 0 :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i

    #### remove all txt that block name save
    for txt_remove_i in ["'", "[", "]", "{", "}", ":", " ", ","]:
        if txt_remove_i == " " or txt_remove_i == ",":
            params_str_name = params_str_name.replace(txt_remove_i, '_')
        else:
            params_str_name = params_str_name.replace(txt_remove_i, '')
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run__{name_function}__{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
    
    #### script text
    lines = ['#!/bin/bash']
    lines += [f'#SBATCH --job-name={name_function}']
    lines += [f'#SBATCH --output=%slurm_{name_function}_{params_str_name}.log']
    lines += [f'#SBATCH --cpus-per-task={n_core_slurms}']
    lines += [f'#SBATCH --mem={mem_crnl_cluster}']
    lines += [f'srun {python} {os.path.join(path_slurm, slurm_script_name)}']
        
    #### write script and execute
    slurm_bash_script_name =  f"bash__{name_function}__{params_str_name}.batch" #add params
        
    with open(slurm_bash_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()

    #### execute bash
    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')
    subprocess.Popen(['sbatch', f'{slurm_bash_script_name}']) 

    # wait subprocess to lauch before removing
    #time.sleep(4)
    #os.remove(slurm_script_name)
    #os.remove(slurm_bash_script_name)

    #### get back to original path
    os.chdir(scritp_path)




#name_script, name_function, params = 'n9_fc_analysis', 'compute_pli_ispc_allband', [sujet]
def execute_function_in_slurm_bash_mem_choice(name_script, name_function, params, mem_required):

    scritp_path = os.getcwd()
    
    python = sys.executable

    #### params to print in script
    params_str = ""
    for i, params_i in enumerate(params):
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if i == 0 :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for i, params_i in enumerate(params):

        str_i = str(params_i)

        if i == 0 :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i

    #### remove all txt that block name save
    for txt_remove_i in ["'", "[", "]", "{", "}", ":", " ", ","]:
        if txt_remove_i == " " or txt_remove_i == ",":
            params_str_name = params_str_name.replace(txt_remove_i, '_')
        else:
            params_str_name = params_str_name.replace(txt_remove_i, '')
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run__{name_function}__{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
    
    #### script text
    lines = ['#!/bin/bash']
    lines += [f'#SBATCH --job-name={name_function}']
    lines += [f'#SBATCH --output=%slurm_{name_function}_{params_str_name}.log']
    lines += [f'#SBATCH --cpus-per-task={n_core_slurms}']
    lines += [f'#SBATCH --mem={mem_required}']
    lines += [f'srun {python} {os.path.join(path_slurm, slurm_script_name)}']
        
    #### write script and execute
    slurm_bash_script_name =  f"bash__{name_function}__{params_str_name}.batch" #add params
        
    with open(slurm_bash_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()

    #### execute bash
    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')
    subprocess.Popen(['sbatch', f'{slurm_bash_script_name}']) 

    # wait subprocess to lauch before removing
    #time.sleep(4)
    #os.remove(slurm_script_name)
    #os.remove(slurm_bash_script_name)

    #### get back to original path
    os.chdir(scritp_path)














################################
######## WAVELETS ########
################################


def get_wavelets():

    #### compute wavelets
    wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

    # create Morlet wavelet family
    for fi in range(nfrex):
        
        s = cycles[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    if debug:

        plt.plot(np.sum(np.abs(wavelets),axis=1))
        plt.show()

        plt.pcolormesh(np.real(wavelets))
        plt.show()

        plt.plot(np.real(wavelets)[0,:])
        plt.show()

    return wavelets












############################
######## LOAD DATA ########
############################


def get_params(sujet, monopol):

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    respi_ratio_allcond = get_all_respi_ratio(sujet)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    params = {'conditions' : conditions, 'chan_list' : chan_list, 'chan_list_ieeg' : chan_list_ieeg, 'srate' : srate, 
    'nwind' : nwind, 'nfft' : nfft, 'noverlap' : noverlap, 'hannw' : hannw, 'respi_ratio_allcond' : respi_ratio_allcond}

    return params

    

def get_chanlist(sujet, monopol):

    path_source = os.getcwd()
    
    #### select conditions to keep
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    
    if monopol:
        raw = mne.io.read_raw_fif(f'{sujet}_FR_CV_1_wb.fif', verbose='critical')
    else:
        raw = mne.io.read_raw_fif(f'{sujet}_FR_CV_1_wb_bi.fif', verbose='critical')
    
    chan_list = raw.info['ch_names']
    chan_list_ieeg = chan_list[:-4] # on enlève : nasal, ventral, ECG, ECG_cR

    #### correct chan_list for paris sujet
    if sujet[:3] == 'pat' and monopol == False:
        for nchan_i, nchan in enumerate(chan_list_ieeg):
            if len(nchan.split('-')) == 3:
                chan_list_ieeg[nchan_i] = f"{nchan.split('-')[0]}-{nchan.split('-')[1]}"

    #### go back to path source
    os.chdir(path_source)

    return chan_list, chan_list_ieeg






def load_data_sujet(sujet, band_prep, cond, session_i, monopol):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if monopol:
            if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ) & ( session_name.find('bi') == -1 ):
                load_i.append(i)
        else:
            if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ) & ( session_name.find('bi') != -1 ):
                load_i.append(i)

    load_list = [os.listdir()[i] for i in load_i if os.listdir()[i].find(str(session_i+1)) != -1]

    raw = mne.io.read_raw_fif(load_list[0], preload=True, verbose='critical')

    data = raw.get_data()

    if sujet[:3] == 'pat' and sujet_respi_adjust[sujet] == 'inverse':
        data[-3,:] = data[-3,:]*-1
        data[-4,:] = data[-4,:]*-1

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return data



def get_srate(sujet):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw = mne.io.read_raw_fif(sujet + '_FR_CV_1_lf.fif', preload=True, verbose='critical')
    
    srate = int(raw.info['sfreq'])

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return srate





def organize_raw(sujet, raw, monopol):

    #### extract chan_list
    chan_list_clean = []
    chan_list = raw.info['ch_names']
    srate = int(raw.info['sfreq'])
    [chan_list_clean.append(nchan[23:]) for nchan in chan_list]

    #### extract data
    data = raw.get_data()

    del raw

    #### identify aux chan
    nasal_i = chan_list_clean.index(aux_chan[sujet]['nasal'])
    ventral_i = chan_list_clean.index(aux_chan[sujet]['ventral'])
    ecg_i = chan_list_clean.index(aux_chan[sujet]['ECG'])

    data_aux = np.vstack((data[nasal_i,:], data[ventral_i,:], data[ecg_i,:]))

    if debug:
        plt.plot(data_aux[0,:])
        plt.plot(data_aux[1,:])
        plt.plot(data_aux[2,:])
        plt.show()

    #### remove from data
    data_ieeg = data.copy()

    # remove other aux
    for aux_name in aux_chan[sujet].keys():

        aux_i = chan_list_clean.index(aux_chan[sujet][aux_name])
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_clean.remove(aux_chan[sujet][aux_name])

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']
    chan_list_ieeg = chan_list_clean


    return data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate



########################################
######## LOAD RESPI FEATURES ########
########################################

def load_respfeatures(sujet):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_respfeatures, sujet, 'RESPI'))

    if sujet in sujet_list:

        conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

    else:

        conditions = ['FR_CV']

    #### get respi features
    respfeatures_allcond = {}

    for cond in conditions:

        respfeatures_cond = []

        for session_i in range(session_count[cond]):

            respfeatures_cond.append(pd.read_excel(f'{sujet}_{cond}_{session_i+1}_respfeatures.xlsx'))

        respfeatures_allcond[cond] = respfeatures_cond
    
    #### go back to path source
    os.chdir(path_source)

    return respfeatures_allcond



def get_all_respi_ratio(sujet):
    
    respfeatures_allcond = load_respfeatures(sujet)
    
    respi_ratio_allcond = {}

    for cond in conditions_allsubjects:

        if len(respfeatures_allcond[cond]) == 1:

            mean_cycle_duration = np.mean(respfeatures_allcond[cond][0][['inspi_duration', 'expi_duration']].values, axis=0)
            mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

            respi_ratio_allcond[cond] = [ mean_inspi_ratio ]

        elif len(respfeatures_allcond[cond]) > 1:

            data_to_short = []
            data_to_short_count = 1

            for session_i in range(len(respfeatures_allcond[cond])):   
                
                if session_i == 0 :

                    mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['inspi_duration', 'expi_duration']].values, axis=0)
                    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
                    data_to_short = [ mean_inspi_ratio ]

                elif session_i > 0 :

                    mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['inspi_duration', 'expi_duration']].values, axis=0)
                    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

                    data_replace = [(data_to_short[0] + mean_inspi_ratio)]
                    data_to_short_count += 1

                    data_to_short = data_replace.copy()
            
            # to put in list
            respi_ratio_allcond[cond] = data_to_short[0] / data_to_short_count

    return respi_ratio_allcond









################################
######## STRETCH ########
################################


#resp_features, nb_point_by_cycle, data = respfeatures_i, stretch_point_surrogates_MVL_Cxy, x_zscore
def stretch_data(resp_features, nb_point_by_cycle, data, srate):

    #### params
    cycle_times = resp_features[['inspi_time', 'expi_time', 'next_inspi_time']].values
    mean_cycle_duration = np.mean(resp_features[['inspi_duration', 'expi_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,data.shape[0])/srate

    #### stretch
    if stretch_TF_auto:

        cycles = physio.deform_traces_to_cycle_template(data.reshape(-1,1), times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=mean_inspi_ratio, output_mode='stacked')
    else:
        
        cycles = physio.deform_traces_to_cycle_template(data.reshape(-1,1), times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=ratio_stretch_TF, output_mode='stacked')

    #### clean
    mask = resp_features[resp_features['select'] == 1].index.values
    cycle_clean = cycles[mask, :, :]

    #### reshape
    if np.iscomplex(data[0]):
        data_stretch = np.zeros(( cycle_clean.shape[0], nb_point_by_cycle ), dtype='complex')
    else:
        data_stretch = np.zeros(( cycle_clean.shape[0], nb_point_by_cycle ))

    for cycle_i in range(cycle_clean.shape[0]):

        data_stretch[cycle_i, :] = cycle_clean[cycle_i,:].reshape(-1)

    #### inspect
    if debug == True:

        plt.plot(data_stretch.mean(axis=0))
        plt.show()

    return data_stretch, mean_inspi_ratio

    




#resp_features, nb_point_by_cycle, data, srate = respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[n_chan,:,:], srate
def stretch_data_tf(resp_features, nb_point_by_cycle, data, srate):

    #### params
    cycle_times = resp_features[['inspi_time', 'expi_time', 'next_inspi_time']].values
    mean_cycle_duration = np.mean(resp_features[['inspi_duration', 'expi_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,data.shape[1])/srate

    #### stretch
    if stretch_TF_auto:

        cycles = physio.deform_traces_to_cycle_template(data.T, times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=mean_inspi_ratio, output_mode='stacked')
    else:
        
        cycles = physio.deform_traces_to_cycle_template(data.T, times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=ratio_stretch_TF, output_mode='stacked')

    #### clean
    mask = resp_features[resp_features['select'] == 1].index.values
    cycle_clean = cycles[mask, :, :]

    #### reshape
    if np.iscomplex(data[0,0]):
        data_stretch = np.zeros(( cycle_clean.shape[0], data.shape[0], nb_point_by_cycle ), dtype='complex')
    else:
        data_stretch = np.zeros(( cycle_clean.shape[0], data.shape[0], nb_point_by_cycle ))

    for cycle_i in range(cycle_clean.shape[0]):

        data_stretch[cycle_i, :, :] = cycle_clean[cycle_i,:,:].T

    #### inspect
    if debug == True:

        plt.pcolormesh(np.mean(data_stretch, axis=0))
        plt.show()

    return data_stretch, mean_inspi_ratio












########################################
######## LOAD LOCALIZATION ########
########################################



def get_loca_df(sujet, monopol):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_anatomy, sujet))

    if monopol:
        file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')
    else:
        file_plot_select = pd.read_excel(sujet + '_plot_loca_bi.xlsx')

    chan_list_ieeg_trc = file_plot_select['plot'][file_plot_select['select'] == 1].values.tolist()

    if sujet[:3] == 'pat':
        chan_list_ieeg_csv = chan_list_ieeg_trc.copy()
    else:
        if monopol:
            chan_list_ieeg_csv, trash = modify_name(chan_list_ieeg_trc)
        else:
            chan_list_ieeg_csv = chan_list_ieeg_trc
        chan_list_ieeg_csv.sort()

    ROI_ieeg = []
    lobes_ieeg = []
    for chan_name in chan_list_ieeg_csv:
        ROI_ieeg.append( file_plot_select['localisation_corrected'].loc[file_plot_select['plot'] == chan_name].values.tolist()[0] )
        lobes_ieeg.append( file_plot_select['lobes_corrected'].loc[file_plot_select['plot'] == chan_name].values.tolist()[0] )

    dict_loca = {'name' : chan_list_ieeg_trc,
                'ROI' : ROI_ieeg,
                'lobes' : lobes_ieeg
                }

    df_loca = pd.DataFrame(dict_loca, columns=dict_loca.keys())

    os.chdir(path_source)

    return df_loca


def get_mni_loca(sujet):

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_txt = open(sujet + '_chanlist_ieeg.txt', 'r')
    chan_list_txt_readlines = chan_list_txt.readlines()
    chan_list_ieeg = [i.replace('\n', '') for i in chan_list_txt_readlines]
    chan_list_ieeg, trash = modify_name(chan_list_ieeg)
    chan_list_ieeg.sort()

    mni_loc = file_plot_select['MNI']

    dict_mni = {}
    for chan_name in chan_list_ieeg:
        mni_nchan = file_plot_select['MNI'].loc[file_plot_select['plot'] == chan_name].values[0]
        mni_nchan = mni_nchan[1:-1]
        mni_nchan_convert = [float(mni_nchan.split(',')[0]), float(mni_nchan.split(',')[1]), float(mni_nchan.split(',')[2])]
        dict_mni[chan_name] = mni_nchan_convert

    return dict_mni










########################################
######## CHANGE NAME CSV TRC ########
########################################

#chan_list = prms['chan_list_ieeg']
def modify_name(chan_list):
    
    chan_list_modified = []
    chan_list_keep = []

    #nchan = ".A'1"
    for nchan in chan_list:

        #### what we remove
        if nchan.find("+") != -1:
            continue

        if np.sum([str.isalpha(str_i) for str_i in nchan]) >= 2 and nchan.find('p') == -1:
            continue

        if nchan.find('ECG') != -1:
            continue

        if nchan.find('. ') != -1:
            continue

        if nchan.find('*') != -1:
            continue

        if nchan.find('@') != -1:
            continue

        #### what we do to chan we keep
        else:

            nchan = nchan.replace('.', '')
            nchan_mod = nchan.replace(' ', '')
            nchan_mod = nchan_mod.replace("'", 'p')

            if nchan_mod.find('p') != -1:
                split = nchan_mod.split('p')
                letter_chan = split[0]

                if len(split[1]) == 1:
                    num_chan = '0' + split[1] 
                else:
                    num_chan = split[1]

                chan_list_modified.append(letter_chan + 'p' + num_chan)
                chan_list_keep.append(nchan)

                continue

            if nchan_mod.find('p') == -1:
                letter_chan = nchan_mod[0]

                split = nchan_mod[1:]

                if len(split) == 1:
                    num_chan = '0' + split
                else:
                    num_chan = split

                chan_list_modified.append(letter_chan + num_chan)
                chan_list_keep.append(nchan)
                continue


    return chan_list_modified, chan_list_keep












########################################
######## MI ANALYSIS FUNCTIONS ########
########################################



def shuffle_CycleFreq(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift
    

def shuffle_Cxy(x):
   half_size = x.shape[0]//2
   ind = np.random.randint(low=0, high=half_size)
   x_shift = x.copy()
   
   x_shift[ind:ind+half_size] *= -1
   if np.random.rand() >=0.5:
       x_shift *= -1

   return x_shift


def Kullback_Leibler_Distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def Shannon_Entropy(a):
    a = np.asarray(a, dtype=float)
    return - np.sum(np.where(a != 0, a * np.log(a), 0))

def Modulation_Index(distrib, show=False, verbose=False):
    distrib = np.asarray(distrib, dtype = float)
    
    if verbose:
        if np.sum(distrib) != 1:
            print(f'(!)  The sum of all bins is not 1 (sum = {round(np.sum(distrib), 2)})  (!)')
        
    N = distrib.size
    uniform_distrib = np.ones(N) * (1/N)
    mi = Kullback_Leibler_Distance(distrib, uniform_distrib) / np.log(N)
    
    if show:
        bin_width_deg = 360 / N
        
        doubled_distrib = np.concatenate([distrib,distrib] )
        x = np.arange(0, doubled_distrib.size*bin_width_deg, bin_width_deg)
        fig, ax = plt.subplots(figsize = (8,4))
        
        doubled_uniform_distrib = np.concatenate([uniform_distrib,uniform_distrib] )
        ax.scatter(x, doubled_uniform_distrib, s=2, color='r')
        
        ax.bar(x=x, height=doubled_distrib, width = bin_width_deg/1.1, align = 'edge')
        ax.set_title(f'Modulation Index = {round(mi, 4)}')
        ax.set_xlabel(f'Phase (Deg)')
        ax.set_ylabel(f'Amplitude (Normalized)')
        ax.set_xticks([0,360,720])

    return mi

def Shannon_MI(a):
    a = np.asarray(a, dtype = float)
    N = a.size
    kl_divergence_shannon = np.log(N) - Shannon_Entropy(a)
    return kl_divergence_shannon / np.log(N)



def get_MVL(x):
    
    _phase = np.arange(0, x.shape[0])*2*np.pi/x.shape[0]
    complex_vec = x*np.exp(1j*_phase)

    MVL = np.abs(np.mean(complex_vec))
    
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.scatter(complex_vec.real, complex_vec.imag)
        ax.scatter(np.mean(complex_vec.real), np.mean(complex_vec.imag), linewidth=3, color='r')
        plt.show()

    return MVL











########################################
######## SCRIPT ADVANCEMENT ########
########################################


def print_advancement(i, i_final, steps=[25, 50, 75]):

    steps_i = {}
    for step in steps:

        step_i = 0
        while (step_i/i_final*100) < step:
            step_i += 1

        steps_i[step] = step_i

    for step, step_i in steps_i.items():

        if i == step_i:
            print(f'{step}%', flush=True)









################################
######## NORMALIZATION ########
################################


def zscore(x):

    x_zscore = (x - x.mean()) / x.std()

    return x_zscore




def zscore_mat(x):

    _zscore_mat = (x - x.mean(axis=1).reshape(-1,1)) / x.std(axis=1).reshape(-1,1)

    return _zscore_mat



def rscore(x):

    mad = np.median( np.abs(x-np.median(x)) ) # median_absolute_deviation

    rzscore_x = (x-np.median(x)) * 0.6745 / mad

    return rzscore_x
    



def rscore_mat(x):

    mad = np.median(np.abs(x-np.median(x, axis=1).reshape(-1,1)), axis=1) # median_absolute_deviation

    _rscore_mat = (x-np.median(x, axis=1).reshape(-1,1)) * 0.6745 / mad.reshape(-1,1)

    return _rscore_mat






#tf_conv = tf_median_cycle[nchan, :, :]
def norm_tf(sujet, tf_conv, electrode_recording_type, norm_method):

    path_source = os.getcwd()

    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

    if norm_method not in ['rscore', 'zscore']:

        #### load baseline
        os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

        if electrode_recording_type == 'monopolaire':
            baselines = xr.open_dataarray(f'{sujet}_baselines.nc')
        if electrode_recording_type == 'bipolaire':
            baselines = xr.open_dataarray(f'{sujet}_baselines_bi.nc')

    if norm_method == 'dB':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = 10*np.log10(tf_conv[n_chan_i,:,:] / baselines.loc[n_chan, :, 'median'].values.reshape(-1,1))

    if norm_method == 'zscore_baseline':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = (tf_conv[n_chan_i,:,:] - baselines.loc[n_chan,:,'mean'].values.reshape(-1,1)) / baselines.loc[n_chan,:,'std'].values.reshape(-1,1)
                
    if norm_method == 'rscore_baseline':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = (tf_conv[n_chan_i,:,:] - baselines.loc[n_chan,:,'median'].values.reshape(-1,1)) * 0.6745 / baselines.loc[n_chan,:,'mad'].values.reshape(-1,1)

    if norm_method == 'zscore':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = zscore_mat(tf_conv[n_chan_i,:,:])
                
    if norm_method == 'rscore':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = rscore_mat(tf_conv[n_chan_i,:,:])


    #### verify baseline
    if debug:

        nchan = 0
        nchan_name = chan_list_ieeg[nchan]

        fig, axs = plt.subplots(ncols=2)
        axs[0].set_title('mean std')
        axs[0].plot(baselines.loc[nchan_name,:,'mean'], label='mean')
        axs[0].plot(baselines.loc[nchan_name,:,'std'], label='std')
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[1].set_title('median mad')
        axs[1].plot(baselines.loc[nchan_name,:,'median'], label='median')
        axs[1].plot(baselines.loc[nchan_name,:,'mad'], label='mad')
        axs[1].legend()
        axs[1].set_yscale('log')
        plt.show()

        tf_test = tf_conv[nchan,:,:int(tf_conv.shape[-1]/10)].copy()

        fig, axs = plt.subplots(nrows=6)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        percentile_sel = 0

        vmin = np.percentile(tf_test.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_test.reshape(-1),100-percentile_sel)
        im = axs[0].pcolormesh(tf_test, vmin=vmin, vmax=vmax)
        axs[0].set_title('raw')
        fig.colorbar(im, ax=axs[0])

        tf_baseline = 10*np.log10(tf_test / baselines.loc[chan_list_ieeg[nchan], :, 'median'].values.reshape(-1,1))
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[1].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[1].set_title('db')
        fig.colorbar(im, ax=axs[1])

        tf_baseline = (tf_test - baselines.loc[chan_list_ieeg[nchan],:,'mean'].values.reshape(-1,1)) / baselines.loc[chan_list_ieeg[nchan],:,'std'].values.reshape(-1,1)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[2].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[2].set_title('zscore')
        fig.colorbar(im, ax=axs[2])

        tf_baseline = (tf_test - baselines.loc[chan_list_ieeg[nchan],:,'median'].values.reshape(-1,1)) / baselines.loc[chan_list_ieeg[nchan],:,'mad'].values.reshape(-1,1)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[3].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[3].set_title('rscore')
        fig.colorbar(im, ax=axs[3])

        tf_baseline = zscore_mat(tf_test)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[4].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[4].set_title('zscore_mat')
        fig.colorbar(im, ax=axs[4])

        tf_baseline = rscore_mat(tf_test)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[5].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[5].set_title('rscore_mat')
        fig.colorbar(im, ax=axs[5])

        plt.show()

    os.chdir(path_source)

    return tf_conv






def get_mad(data, axis=0):

    med = np.median(data, axis=axis)
    mad = np.median(np.abs(data - med), axis=axis) / 0.6744897501960817

    return mad





########################################
######## HRV ANALYSIS HOMEMADE ########
########################################

#### params
def get_params_hrv_homemade(srate_resample_hrv):
    
    nwind_hrv = int( 128*srate_resample_hrv )
    nfft_hrv = nwind_hrv
    noverlap_hrv = np.round(nwind_hrv/90)
    win_hrv = scipy.signal.windows.hann(nwind_hrv)
    f_RRI = (.1, .5)

    return nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, f_RRI


#### RRI, IFR
#ecg_i, ecg_cR, srate, srate_resample = ecg_i, ecg_cR, srate, srate_resample_hrv
def get_RRI_IFR(ecg_i, ecg_cR, srate, srate_resample) :

    cR_sec = ecg_cR # cR in sec

    # RRI computation
    RRI = np.diff(cR_sec)
    RRI = np.insert(RRI, 0, np.median(RRI))
    IFR = (1/RRI)

    # interpolate
    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic', fill_value="extrapolate")
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    #plt.plot(cR_sec, RRI, label='old')
    #plt.plot(cR_sec_resample, RRI_resample, label='new')
    #plt.legend()
    #plt.show()

    return RRI, RRI_resample, IFR

def get_fig_RRI_IFR(ecg_i, ecg_cR, RRI, IFR, srate, srate_resample):

    cR_sec = ecg_cR # cR in sec
    times = np.arange(0,len(ecg_i))/srate # in sec

    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic', fill_value="extrapolate")
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    fig, ax = plt.subplots()
    ax = plt.subplot(411)
    plt.plot(times, ecg_i)
    plt.title('ECG')
    plt.ylabel('a.u.')
    plt.xlabel('s')
    plt.vlines(cR_sec, ymin=min(ecg_i), ymax=max(ecg_i), colors='k')
    plt.subplot(412, sharex=ax)
    plt.plot(cR_sec, RRI)
    plt.title('RRI')
    plt.ylabel('s')
    plt.subplot(413, sharex=ax)
    plt.plot(cR_sec_resample, RRI_resample)
    plt.title('RRI_resampled')
    plt.ylabel('Hz')
    plt.subplot(414, sharex=ax)
    plt.plot(cR_sec, IFR)
    plt.title('IFR')
    plt.ylabel('Hz')
    #plt.show()

    # in this plot one RRI point correspond to the difference value between the precedent RR
    # the first point of RRI is the median for plotting consideration

    return fig

    

#### LF / HF

#RRI_resample, srate_resample, nwind, nfft, noverlap, win = RRI_resample, srate_resample, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv
def get_PSD_LF_HF(RRI_resample, srate_resample, nwind, nfft, noverlap, win, VLF, LF, HF):

    # DETREND
    RRI_detrend = RRI_resample-np.median(RRI_resample)

    # FFT WELCH
    hzPxx, Pxx = scipy.signal.welch(RRI_detrend, fs=srate_resample, window=win, nperseg=nwind, noverlap=noverlap, nfft=nfft)

    AUC_LF = np.trapz(Pxx[(hzPxx>VLF) & (hzPxx<LF)])
    AUC_HF = np.trapz(Pxx[(hzPxx>LF) & (hzPxx<HF)])
    LF_HF_ratio = AUC_LF/AUC_HF

    return AUC_LF, AUC_HF, LF_HF_ratio, hzPxx, Pxx


def get_fig_PSD_LF_HF(Pxx, hzPxx, VLF, LF, HF):

    # PLOT
    fig = plt.figure()
    plt.plot(hzPxx,Pxx)
    plt.ylim(0, np.max(Pxx[hzPxx>0.01]))
    plt.xlim([0,.6])
    plt.vlines([VLF, LF, HF], ymin=min(Pxx), ymax=max(Pxx), colors='r')
    #plt.show()
    
    return fig


#### SDNN, RMSSD, NN50, pNN50
# RR_val = RRI
def get_stats_descriptors(RR_val) :
    SDNN = np.std(RR_val)

    RMSSD = np.sqrt(np.mean((np.diff(RR_val)*1e3)**2))

    NN50 = []
    for RR in range(len(RR_val)) :
        if RR == len(RR_val)-1 :
            continue
        else :
            NN = abs(RR_val[RR+1] - RR_val[RR])
            NN50.append(NN)

    NN50 = np.array(NN50)*1e3
    pNN50 = np.sum(NN50>50)/len(NN50)

    return SDNN, RMSSD, NN50, pNN50

#SDNN_CV, RMSSD_CV, NN50_CV, pNN50_CV = get_stats_descriptors(RRI_CV)


#### Poincarré

def get_poincarre(RRI):
    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    SD1_val = []
    SD2_val = []
    for RR in range(len(RRI)) :
        if RR == len(RRI)-1 :
            continue
        else :
            SD1_val_tmp = (RRI[RR+1] - RRI[RR])/np.sqrt(2)
            SD2_val_tmp = (RRI[RR+1] + RRI[RR])/np.sqrt(2)
            SD1_val.append(SD1_val_tmp)
            SD2_val.append(SD2_val_tmp)

    SD1 = np.std(SD1_val)
    SD2 = np.std(SD2_val)
    Tot_HRV = SD1*SD2*np.pi

    return SD1, SD2, Tot_HRV


        
def get_fig_poincarre(RRI):

    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    fig = plt.figure()
    plt.scatter(RRI, RRI_1)
    plt.xlabel('RR (ms)')
    plt.ylabel('RR+1 (ms)')
    plt.title('Poincarré ')
    plt.xlim(.600,1.)
    plt.ylim(.600,1.)

    return fig
    
#### DeltaHR

#RRI, srate_resample, f_RRI, condition = result_struct[keys_result[0]][1], srate_resample, f_RRI, cond 
def get_dHR(RRI_resample, srate_resample, f_RRI):
    
    times = np.arange(0,len(RRI_resample))/srate_resample

        # stairs method
    #RRI_stairs = np.array([])
    #len_cR = len(cR) 
    #for RR in range(len(cR)) :
    #    if RR == 0 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1]))])
    #    elif RR != 0 and RR != len_cR-1 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1] - cR[RR]))])
    #    elif RR == len_cR-1 :
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(len(ecg) - cR[RR]))])


    peaks, troughs = find_extrema(RRI_resample, srate_resample, f_RRI)
    peaks_RRI, troughs_RRI = RRI_resample[peaks], RRI_resample[troughs]
    peaks_troughs = np.stack((peaks_RRI, troughs_RRI), axis=1)

    fig_verif = plt.figure()
    plt.plot(times, RRI_resample)
    plt.vlines(peaks/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='b')
    plt.vlines(troughs/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='r')
    #plt.show()

    dHR = np.diff(peaks_troughs/srate_resample, axis=1)*1e3

    fig_dHR = plt.figure()
    ax = plt.subplot(211)
    plt.plot(times, RRI_resample*1e3)
    plt.title('RRI')
    plt.ylabel('ms')
    plt.subplot(212, sharex=ax)
    plt.plot(troughs/srate_resample, dHR)
    plt.hlines(np.median(dHR), xmin=min(times), xmax=max(times), colors='m', label='median = {:.3f}'.format(np.median(dHR)))
    plt.legend()
    plt.title('dHR')
    plt.ylabel('ms')
    plt.vlines(peaks/srate_resample, ymin=0, ymax=0.01, colors='b')
    plt.vlines(troughs/srate_resample, ymin=0, ymax=0.01, colors='r')
    plt.tight_layout()
    #plt.show()


    return fig_verif, fig_dHR


def ecg_analysis_homemade(ecg_i, srate, srate_resample_hrv, fig_token=False):

    #### load params
    nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, f_RRI = get_params_hrv_homemade(srate_resample_hrv)

    #### load cR
    ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srate*0.5)[0]
    ecg_cR = ecg_cR/srate

    #### verif
    if debug:
        times = np.arange(ecg_i.shape[0])/srate
        plt.plot(times, ecg_i)
        plt.vlines(ecg_cR, ymin=np.min(ecg_i) ,ymax=np.max(ecg_i), colors='r')
        plt.show()


    #### initiate metrics names
    res_list = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

    #### RRI
    RRI, RRI_resample, IFR = get_RRI_IFR(ecg_i, ecg_cR, srate, srate_resample_hrv)

    HRV_MeanNN = np.mean(RRI)
    
    #### PSD
    VLF, LF, HF = .04, .15, .4
    AUC_LF, AUC_HF, LF_HF_ratio, hzPxx, Pxx = get_PSD_LF_HF(RRI_resample, srate_resample_hrv, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, VLF, LF, HF)

    #### descriptors
    SDNN, RMSSD, NN50, pNN50 = get_stats_descriptors(RRI)

    #### poincarré
    SD1, SD2, Tot_HRV = get_poincarre(RRI)

    #### df
    res_tmp = [HRV_MeanNN*1e3, SDNN*1e3, RMSSD, pNN50*100, AUC_LF/10, AUC_HF/10, LF_HF_ratio, SD1*1e3, SD2*1e3, Tot_HRV*1e6]
    data_df = {}
    for i, dv in enumerate(res_list):
        data_df[dv] = [res_tmp[i]]

    hrv_metrics_homemade = pd.DataFrame(data=data_df)

    #### for figures

    #### dHR
    if fig_token:
        fig_verif, fig_dHR = get_dHR(RRI_resample, srate_resample_hrv, f_RRI)

    #### fig
    if fig_token:
        fig_RRI = get_fig_RRI_IFR(ecg_i, ecg_cR, RRI, IFR, srate, srate_resample_hrv)
        fig_PSD = get_fig_PSD_LF_HF(Pxx, hzPxx, VLF, LF, HF) 
        fig_poincarre = get_fig_poincarre(RRI)

        fig_list = [fig_RRI, fig_PSD, fig_poincarre, fig_verif, fig_dHR]

        plt.close('all')

        return hrv_metrics_homemade, fig_list

    else:

        return hrv_metrics_homemade










################################
######## NEUROKIT ######## 
################################

#ecg_i = xr_chunk[sujet_i, cond_i, trial_i, :].data
def nk_analysis(ecg_i, srate):

    ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srate*0.5)[0]
    peaks_dict = {'ECG_R_Peaks' : ecg_cR}
    ecg_peaks = pd.DataFrame(peaks_dict)

    hrv_metrics = nk.hrv(ecg_peaks, sampling_rate=srate, show=False)

    hrv_metrics_name = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

    col_to_drop = []
    col_hrv = list(hrv_metrics.columns.values) 
    for metric_name in col_hrv :
        if (metric_name in hrv_metrics_name) == False :
            col_to_drop.append(metric_name)

    hrv_metrics_short = hrv_metrics.copy()
    hrv_metrics_short = hrv_metrics_short.drop(col_to_drop, axis=1)

    return hrv_metrics_short









