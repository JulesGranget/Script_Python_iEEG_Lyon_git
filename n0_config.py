
import numpy as np
import scipy.signal

################################
######## MODULES ########
################################

# anaconda
# neurokit2 as nk
# respirationtools
# mne
# neo
# bycycle
# pingouin

################################
######## GENERAL PARAMS ######## 
################################

enable_big_execute = True

#### whole protocole
sujet = 'CHEe' #OK
#sujet = 'GOBc' #OK
#sujet = 'MAZm' #OK
#sujet = 'TREt' 

#### FR_CV only
#sujet = 'MUGa'
#sujet = 'BANc'
#sujet = 'KOFs'
#sujet = 'LEMl'
#sujet = 'pat_02459_0912'
#sujet = 'pat_02476_0929'
#sujet = 'pat_02495_0949'

#sujet = 'DEBUG'

#### whole protocole
sujet_list = ['CHEe', 'GOBc', 'MAZm', 'TREt']

#### FR_CV
sujet_list_FR_CV = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'MUGa', 'BANc', 'KOFs', 'LEMl', 'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949']

conditions_allsubjects = ['RD_CV', 'RD_FV', 'RD_SV', 'RD_AV', 'FR_CV', 'FR_MV']

condition_diff = [['FV','SV']]

band_prep_list = ['lf', 'hf']
freq_band_list = [{'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]}, {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}]




########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()

if PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Home'
    path_main_workdir = 'C:\\Users\\jules\\Desktop\\Codage Informatique\\Script_Python_iEEG_Lyon_git'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon\\iEEG'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\iEEG\\Mmap'
    n_core = 4

if PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    path_main_workdir = 'C:\\Users\\jules\\Desktop\\Script_Python_iEEG_Lyon_git'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Mmap'
    n_core = 2

if PC_ID == 'pc-jules':

    PC_working = 'Jules_Labo_Linux'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Script_Python_iEEG_Lyon_git'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Mmap'
    n_core = 6

if PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Script_Python_iEEG_Lyon_git'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Mmap'
    n_core = 6

if PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Script_Python_iEEG_Lyon_git'
    path_general = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 40



path_data = os.path.join(path_general, 'Data', 'raw_data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_respfeatures = os.path.join(path_general, 'Analyses', 'results') 
path_anatomy = os.path.join(path_general, 'Analyses', 'anatomy') 



################################################
######## ELECTRODES REMOVED BEFORE LOCA ######## 
################################################

electrodes_to_remove = {

'CHEe' : [],
'GOBc' : ['Bp'], # because the neurosurgeon didnt indicates the electrode's real size
'MAZm' : [],
'MUGa' : [],
'TREt' : []

}



################################
######## PREP INFO ######## 
################################


conditions_trig = {
'RD_CV' : ['31', '32'], # RespiDriver Comfort Ventilation
'RD_FV' : ['51', '52'], # RespiDriver Fast Ventilation  
'RD_SV' : ['11', '12'], # RespiDriver Slow Ventilation
'RD_AV' : ['61', '62'], # RespiDriver Ample Ventilation
'FR_CV' : ['CV_start', 'CV_stop'], # FreeVentilation Comfort Ventilation
'FR_MV' : ['MV_start', 'MV_stop'], # FreeVentilation Mouth Ventilation
}


aux_chan = {
'CHEe' : {'nasal': 'p7+', 'ventral' : 'p8+', 'ECG' : 'ECG'}, # OK
'GOBc' : {'nasal': 'p13+', 'ventral' : 'p14+', 'ECG' : 'ECG'}, # OK
'MAZm' : {'nasal': 'p7+', 'ventral' : 'p8+', 'ECG' : 'ECG'}, # OK
'TREt' : {'nasal': 'p19+', 'ventral' : 'p20+', 'ECG' : 'ECG1'}, # OK
'MUGa' : {'nasal': 'p20+', 'ventral' : 'p19+', 'ECG' : 'ECG'}, # OK
'BANc' : {'nasal': 'p19+', 'ventral' : None, 'ECG' : 'ECG'}, # OK
'KOFs' : {'nasal': 'p7+', 'ventral' : None, 'ECG' : 'ECG'}, # OK
'LEMl' : {'nasal': 'p17+', 'ventral' : None, 'ECG' : 'ECG1'}, # OK

'DEBUG' : {'nasal': 'p20+', 'ventral' : 'p19+', 'ECG' : 'ECG'}, # OK

}


################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'CHEe' : 'inverse',
'GOBc' : 'inverse',
'MAZm' : 'inverse',
'TREt' : 'normal',
'MUGa' : 'normal',
'BANc' : 'inverse',
'KOFs' : 'normal',
'LEMl' : 'inverse',
}


hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']




################################
######## PREP PARAMS ########
################################ 

prep_step_lf = {
'mean_centered_detrend' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': False},
}

prep_step_hf = {
'mean_centered_detrend' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': True, 'params' : {'l_freq' : 55, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': False},
}





########################################
######## PARAMS SURROGATES ########
########################################

#### Pxx Cxy

zero_pad_coeff = 15

def get_params_spectral_analysis(srate):
    nwind = int( 20*srate ) # window length in seconds*srate
    nfft = nwind*zero_pad_coeff # if no zero padding nfft = nwind
    noverlap = np.round(nwind/2) # number of points of overlap here 50%
    hannw = scipy.signal.windows.hann(nwind) # hann window

    return nwind, nfft, noverlap, hannw

#### plot Pxx Cxy  
if zero_pad_coeff - 5 <= 0:
    remove_zero_pad = 0
remove_zero_pad = zero_pad_coeff - 5

#### stretch
stretch_point_surrogates = 1000

#### coh
n_surrogates_coh = 1000
freq_surrogates = [0, 2]
percentile_coh = .95

#### cycle freq
n_surrogates_cyclefreq = 1000
percentile_cyclefreq_up = .99
percentile_cyclefreq_dw = .01






################################
######## PRECOMPUTE TF ########
################################

#### stretch
stretch_point_TF = 1000
stretch_TF_auto = False
ratio_stretch_TF = 0.45

#### TF & ITPC
nfrex_hf = 50
nfrex_lf = 50
ncycle_list_lf = [7, 15]
ncycle_list_hf = [20, 30]
srate_dw = 10



################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi


################################
######## FC ANALYSIS ########
################################

#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}




################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)





########################################
######## COMPUTATIONAL NOTES ######## 
########################################

#### CHEe
#

#### GOBc
#

#### MAZm
#

#### MUGa
#

#### TREt
#

