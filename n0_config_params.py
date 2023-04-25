
import numpy as np
import scipy.signal

################################
######## MODULES ########
################################

# anaconda (numpy, scipy, pandas, matplotlib, glob2, joblib, xlrd)
# neurokit2 as nk
# respirationtools
# mne
# neo
# bycycle
# pingouin

################################
######## GENERAL PARAMS ######## 
################################

perso_repo_computation = False

#### electrode recording type
# monopol = True
monopol = False

#### whole protocole
# sujet = 'CHEe'
# sujet = 'GOBc'
# sujet = 'MAZm'
# sujet = 'TREt'
sujet = 'POTm'

#sujet = 'DEBUG'

srate = 500

#### whole protocole
sujet_list = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'POTm']

#### FR_CV
sujet_list_FR_CV =  ['CHEe', 'GOBc', 'MAZm', 'TREt', 'POTm', 'BANc', 'KOFs', 'LEMl', 'MUGa',
                    'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949',
                    'pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601',
                    'pat_03146_1608', 'pat_03174_1634'
                    ]

sujet_list_paris_only_FR_CV = ['pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949',
                    'pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601',
                    'pat_03146_1608', 'pat_03174_1634']

conditions_allsubjects = ['RD_CV', 'RD_FV', 'RD_SV', 'RD_AV', 'FR_CV', 'FR_MV']
conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

condition_diff = [['FV','SV']]

band_prep_list = ['lf', 'hf']
freq_band_list = [{'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]}, {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}]
freq_band_whole = {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

freq_band_list_precompute = [{'theta_1' : [2,10], 'theta_2' : [4,8], 'alpha_1' : [8,12], 'alpha_2' : [8,14], 'beta_1' : [12,40], 'beta_2' : [10,40], 'whole_1' : [2,50]}, {'l_gamma_1' : [50, 80], 'h_gamma_1' : [80, 120]}]

freq_band_dict_FC = {'wb' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40]},
                'lf' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40], 'whole' : [2,50]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

freq_band_dict = {'wb' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40]},
                'lf' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

freq_band_dict_FC_function = {'lf' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [10,40]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]},
                'wb' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [10,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}  }


session_count =    {'FR_CV' : 1, 'RD_CV' : 2, 'RD_FV' : 2, 'RD_SV' : 3}

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
    if perso_repo_computation:
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon\\iEEG'
    else:    
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon\\iEEG'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon\\iEEG'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\iEEG\\Mmap'
    n_core = 4

elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    if perso_repo_computation:
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    else:    
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Mmap'
    n_core = 2

elif PC_ID == 'pc-jules':

    PC_working = 'Jules_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Script_Python_iEEG_Lyon_git'
    else:    
        path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Mmap'
    n_core = 6

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/valentin/Bureau/perso_repo_computation/Script_Python_iEEG_Lyon_git'
    else:    
        path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Mmap'
    n_core = 6

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Script_Python_iEEG_Lyon_git'
    path_general = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 15

else:

    PC_working = 'crnl_cluster'
    path_main_workdir = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Script_Python_iEEG_Lyon_git'
    path_general = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 10

path_data = os.path.join(path_general, 'Data', 'raw_data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_respfeatures = os.path.join(path_general, 'Analyses', 'results') 
path_anatomy = os.path.join(path_general, 'Analyses', 'anatomy') 

path_slurm = os.path.join(path_general, 'Script_slurm')

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10


################################################
######## ELECTRODES REMOVED BEFORE LOCA ######## 
################################################

electrodes_to_remove = {

'CHEe' : [],
'GOBc' : ['Bp'], # because the neurosurgeon didnt indicates the electrode's real size
'MAZm' : [],
'MUGa' : [],
'TREt' : [],
'POTm' : [],
'BANc' : [], 
'KOFs' : [], 
'LEMl' : [], 

'pat_02459_0912' : [], 
'pat_02476_0929' : [], 
'pat_02495_0949' : [],

'pat_03083_1527' : [], 
'pat_03105_1551' : [], 
'pat_03128_1591' : [], 
'pat_03138_1601' : [],
'pat_03146_1608' : [], 
'pat_03174_1634' : []

}



################################
######## PREP INFO ######## 
################################

dw_srate = 500 # new srate

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
'POTm' : {'nasal': 'p16+', 'ventral' : 'p17+', 'ECG' : 'ECG1'}, # OK
'MUGa' : {'nasal': 'p20+', 'ventral' : 'p19+', 'ECG' : 'ECG'}, # OK
'BANc' : {'nasal': 'p19+', 'ventral' : None, 'ECG' : 'ECG'}, # OK
'KOFs' : {'nasal': 'p7+', 'ventral' : None, 'ECG' : 'ECG'}, # OK
'LEMl' : {'nasal': 'p17+', 'ventral' : None, 'ECG' : 'ECG1'}, # OK

'pat_02459_0912' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'}, # OK
'pat_02476_0929' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'}, # OK
'pat_02495_0949' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'}, # OK

'pat_03083_1527' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'}, # OK
'pat_03105_1551' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'}, # OK
'pat_03128_1591' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'}, # OK
'pat_03138_1601' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'}, # OK
'pat_03146_1608' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'},
'pat_03174_1634' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'},

'DEBUG' : {'nasal': 'p20+', 'ventral' : 'p19+', 'ECG' : 'ECG'}, # OK

}



prep_step_lf = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': True},
}

prep_step_hf = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': True, 'params' : {'l_freq' : 55, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': True},
}


prep_step_wb = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'average_reref' : {'execute': True},
}


################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'CHEe' : 'inverse',
'GOBc' : 'inverse',
'MAZm' : 'inverse',
'TREt' : 'normal',
'POTm' : 'normal',
'MUGa' : 'normal',
'BANc' : 'inverse',
'KOFs' : 'normal',
'LEMl' : 'inverse',
}


hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']




################################
######## RESPI PARAMS ########
################################ 



#### INSPI DOWN
sujet_respi_adjust = {
'CHEe' : 'inverse',
'GOBc' : 'inverse',
'MAZm' : 'inverse',
'TREt' : 'normal',
'POTm' : 'normal',
'MUGa' : 'normal',
'BANc' : 'inverse',
'KOFs' : 'inverse',
'LEMl' : 'inverse',

'pat_02459_0912' : 'inverse',
'pat_02476_0929' : 'normal',
'pat_02495_0949' : 'inverse',

'pat_03083_1527' : 'normal',
'pat_03105_1551' : 'inverse',
'pat_03128_1591' : 'normal',
'pat_03138_1601' : 'normal',
'pat_03146_1608' : 'normal',
'pat_03174_1634' : 'normal'

}


SD_delete_cycles_freq = 3
SD_delete_cycles_amp = 3

#### filter params
f_theta = (0.1, 2)
l_freq = 0
h_freq = 2

#### sujet that need more filter for respi sig
sujet_manual_detection = ['pat_03105_1551', 'pat_03128_1591']
sujet_for_more_filter = ['pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949', 'pat_03138_1601',
                        'pat_03146_1608', 'pat_03174_1634']


respi_scale_cond = {'FR_CV' : [0.1, 0.35],
                'RD_CV' : [0.1, 0.35],
                'RD_SV' : [0.1, 0.2],
                'RD_FV' : [0.35, 0.60]}

outlier_coeff_removing_cond = {'FR_CV' : 6,
                'RD_CV' : 4,
                'RD_SV' : 4,
                'RD_FV' : 4}

cycle_detection_params = {
'exclusion_metrics' : 'med',
'outlier_coeff_removing' : outlier_coeff_removing_cond,
'metric_coeff_exclusion' : 3,
'respi_scale' : respi_scale_cond, #Hz
}



########################################
######## PARAMS SURROGATES ########
########################################

#### Pxx Cxy

zero_pad_coeff = 2

def get_params_spectral_analysis(srate):
    nwind = int( 50*srate ) # window length in seconds*srate
    nfft = nwind*zero_pad_coeff # if no zero padding nfft = nwind
    noverlap = np.round(nwind/2) # number of points of overlap here 50%
    hannw = scipy.signal.windows.hann(nwind) # hann window

    return nwind, nfft, noverlap, hannw

#### plot Pxx Cxy  
remove_zero_pad = 5

#### stretch
stretch_point_surrogates_MVL_Cxy = 1000
stretch_point_IE = [300, 500]
stretch_point_EI = [900, 100]
stretch_point_I = [100, 300]
stretch_point_E = [600, 800]


#### coh
n_surrogates_coh = 1000
freq_surrogates = [0, 2]
percentile_coh = .95

#### cycle freq
n_surrogates_cyclefreq = 1000
percentile_cyclefreq_up = .99
percentile_cyclefreq_dw = .01


#### n bin for MI computation
MI_n_bin = 18


################################
######## PRECOMPUTE TF ########
################################

#### stretch
stretch_point_TF = 1000
stretch_TF_auto = False
ratio_stretch_TF = 0.50

#### TF & ITPC
nfrex = 150
ncycle_list = [7, 41]
freq_list = [2, 150]
srate_dw = 10
wavetime = np.arange(-3,3,1/srate)
frex = np.logspace(np.log10(freq_list[0]), np.log10(freq_list[1]), nfrex) 
cycles = np.logspace(np.log10(ncycle_list[0]), np.log10(ncycle_list[1]), nfrex).astype('int')
Pxx_wavelet_norm = 1000

#### STATS
n_surrogates_tf = 1000
tf_percentile_sel_stats = 2 # for both side
tf_stats_percentile_cluster = 95
norm_method = 'rscore'# 'zscore', 'dB'
tf_stats_percentile_cluster_allplot = 99
df_extraction_Cxy = 0.02 #Hz aroutd respi median


#### plot
tf_plot_percentile_scale = 1 #for one side






################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi


################################
######## FC ANALYSIS ########
################################

#### ROI for DFC
ROI_for_DFC_df =    ['orbitofrontal', 'cingulaire ant rostral', 'cingulaire ant caudal', 'cingulaire post',
                    'insula ant', 'insula post', 'parahippocampique', 'amygdala', 'hippocampus']
ROI_for_DFC_plot =    ['orbitofrontal', 'cingulaire ant rostral', 'cingulaire ant caudal', 'cingulaire post',
                    'insula ant', 'insula post', 'parahippocampique', 'amygdala', 'hippocampus', 'temporal inf',
                    'temporal med', 'temporal sup']

#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

percentile_thresh = 90

#### for DFC
slwin_dict = {'theta' : 5, 'alpha' : 3, 'beta' : 1, 'l_gamma' : .3, 'h_gamma' : .3} # seconds
slwin_step_coeff = .1  # in %, 10% move

band_name_fc_dfc = ['beta', 'l_gamma', 'h_gamma']


################################
######## HRV ANALYSIS ########
################################

cond_HRV = ['FR_CV', 'RD_CV', 'RD_SV', 'RD_FV']

srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)



