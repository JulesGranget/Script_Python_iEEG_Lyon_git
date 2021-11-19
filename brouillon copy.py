

import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import respirationtools
import mne
import scipy.fftpack
import scipy.signal
import joblib


from n0_config import *
from n0bis_analysis_functions import *







sujet = 'DEBUG'
aux_chan = {
'CHEe' : {'nasal': 'p7+', 'ventral' : 'p8+', 'ECG' : 'ECG'}, # OK
'GOBc' : {'nasal': 'p13+', 'ventral' : 'p14+', 'ECG' : 'ECG'}, # OK
'MAZm' : {'nasal': 'p7+', 'ventral' : 'p8+', 'ECG' : 'ECG'}, # OK
'TREt' : {'nasal': 'p19+', 'ventral' : 'p20+', 'ECG' : 'ECG1'}, # OK
'MUGa' : {'nasal': 'p20+', 'ventral' : 'p19+', 'ECG' : 'ECG'}, # OK
'BANc' : {'nasal': 'p19+', 'ventral' : None, 'ECG' : 'ECG'}, # OK
'KOFs' : {'nasal': 'p7+', 'ventral' : None, 'ECG' : 'ECG'}, # OK
'LEMl' : {'nasal': 'p17+', 'ventral' : None, 'ECG' : 'ECG1'}, # OK

'DEBUG' : {'nasal': 'p7+', 'ventral' : 'p8+', 'ECG' : 'ECG'}, # OK

}




os.chdir(os.path.join(path_data,sujet))

debug = False

#### identify number of trc file
trc_file_names = glob.glob('*.TRC')

#### sort order TRC
trc_file_names_ordered = []
[trc_file_names_ordered.append(file_name) for file_name in trc_file_names if file_name.find('FR_CV') != -1]
[trc_file_names_ordered.append(file_name) for file_name in trc_file_names if file_name.find('PROTOCOLE') != -1]

#### extract file one by one
print('#### EXTRACT TRC ####')
data_whole = []
chan_list_whole = []
srate_whole = []
events_name_whole = []
events_time_whole = []
#file_i, file_name = 1, trc_file_names[1]
for file_i, file_name in enumerate(trc_file_names_ordered):

    #### current file
    print(file_name)

    #### extract segment with neo
    reader = neo.MicromedIO(filename=file_name)
    seg = reader.read_segment()
    print('len seg : ' + str(len(seg.analogsignals)))

    #### extract data
    data_whole_file = []
    chan_list_whole_file = []
    srate_whole_file = []
    events_name_file = []
    events_time_file = []
    #anasig = seg.analogsignals[2]
    for seg_i, anasig in enumerate(seg.analogsignals):

        chan_list_whole_file.append(anasig.array_annotations['channel_names'].tolist()) # extract chan
        data_whole_file.append(anasig[:, :].magnitude.transpose()) # extract data
        srate_whole_file.append(int(anasig.sampling_rate.rescale('Hz').magnitude.tolist())) # extract srate

    if srate_whole_file != [srate_whole_file[i] for i in range(len(srate_whole_file))] :
        print('srate different in segments')
        exit()
    else :
        srate_file = srate_whole_file[0]

    #### concatenate data
    for seg_i in range(len(data_whole_file)):
        if seg_i == 0 :
            data_file = data_whole_file[seg_i]
            chan_list_file = chan_list_whole_file[seg_i]
        else :
            data_file = np.concatenate((data_file,data_whole_file[seg_i]), axis=0)
            [chan_list_file.append(chan_list_whole_file[seg_i][i]) for i in range(np.size(chan_list_whole_file[seg_i]))]


    #### event
    if len(seg.events[0].magnitude) == 0 : # when its VS recorded
        events_name_file = ['CV_start', 'CV_stop']
        events_time_file = [0, len(data_file[0,:])]
    else : # for other sessions
        #event_i = 0
        for event_i in range(len(seg.events[0])):
            events_name_file.append(seg.events[0].labels[event_i])
            events_time_file.append(int(seg.events[0].times[event_i].magnitude * srate_file))

    #### fill containers
    data_whole.append(data_file)
    chan_list_whole.append(chan_list_file)
    srate_whole.append(srate_file)
    events_name_whole.append(events_name_file)
    events_time_whole.append(events_time_file)

#### concatenate 
print('#### CONCATENATE ####')
data = data_whole[0]
chan_list = chan_list_whole[0]
events_name = events_name_whole[0]
events_time = events_time_whole[0]
srate = srate_whole[0]

if len(trc_file_names) > 1 :
    #trc_i = 0
    for trc_i in range(len(trc_file_names)): 

        if trc_i == 0 :
            len_trc = np.size(data_whole[trc_i],1)
            continue
        else:

            data = np.concatenate((data,data_whole[trc_i]), axis=1)

            [events_name.append(events_name_whole[trc_i][i]) for i in range(len(events_name_whole[trc_i]))]
            [events_time.append(events_time_whole[trc_i][i] + len_trc) for i in range(len(events_time_whole[trc_i]))]

            if chan_list != chan_list_whole[trc_i]:
                print('not the same chan list')
                exit()

            if srate != srate_whole[trc_i]:
                print('not the same srate')
                exit()

            len_trc += np.size(data_whole[trc_i],1)


#### events in df
event_dict = {'name' : events_name, 'time' : events_time}
columns = ['name', 'time']
trig = pd.DataFrame(event_dict, columns=columns)

#### select chan
print('#### REMOVE CHAN ####')
os.chdir(os.path.join(path_anatomy, sujet))

#### first removing
chan_list_first_clean_file = open(sujet + "_trcplot_in_csv.txt", "r")
chan_list_first_clean = chan_list_first_clean_file.read()
chan_list_first_clean = chan_list_first_clean.split("\n")[:-1]
chan_list_first_clean_file.close()

    #### remove chan
if debug:
    data_rmv_first = data.copy() 
else:
    data_rmv_first = data
chan_list_rmv_first = chan_list.copy()
chan_list_nchan_rmv_first = []
for nchan in chan_list:
    if nchan in chan_list_first_clean:
        continue
    else :
        chan_list_nchan_rmv_first.append(nchan)
        chan_i = chan_list_rmv_first.index(nchan)
        data_rmv_first = np.delete(data_rmv_first, chan_i, 0)
        chan_list_rmv_first.remove(nchan)

#### second removing
electrode_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    #### change notation
chan_list_rmv_first_modified, trash = modify_name(chan_list_rmv_first)
chan_list_rmv_first_modified_rmv = chan_list_rmv_first_modified.copy()

    #### remove chan
if debug:
    data_rmv_second = data_rmv_first.copy() 
else:
    data_rmv_second = data_rmv_first
chan_list_rmv_second = chan_list_rmv_first.copy()
chan_list_nchan_rmv_second = []
for nchan in chan_list_rmv_first_modified:
    if electrode_select['select'][electrode_select['plot'] == nchan].values[0] == 1:
        continue
    else :   
        chan_i = chan_list_rmv_first_modified_rmv.index(nchan)
        nchan_trc = chan_list_rmv_second[chan_i]
        chan_list_nchan_rmv_second.append(nchan_trc)

        data_rmv_second = np.delete(data_rmv_second, chan_i, 0)
        chan_list_rmv_second.remove(nchan_trc)
        chan_list_rmv_first_modified_rmv.remove(nchan)

#### identify chan in csv that are not in trc
chan_list_csv = electrode_select['plot'][electrode_select['select'] == 1].values.tolist()
chan_list_add_in_csv = []
for nchan in chan_list_csv:
    if nchan in chan_list_rmv_first_modified_rmv:
        continue
    else:
        chan_list_add_in_csv.append(nchan)

#### idicate removed chan
print('verification nchan out first:')
print(chan_list_nchan_rmv_first)
print('')
print('verification nchan out second:')
print(chan_list_nchan_rmv_second)


#### chan list all rmw
chan_list_all_rmw = chan_list_nchan_rmv_first + chan_list_nchan_rmv_second

#### identify iEEG / respi / ECG

print('#### AUX IDENTIFICATION ####')
nasal_i = chan_list.index(aux_chan.get(sujet).get('nasal'))
ventral_i = chan_list.index(aux_chan.get(sujet).get('ventral'))
ecg_i = chan_list.index(aux_chan.get(sujet).get('ECG'))
data_aux = np.stack((data[nasal_i, :], data[ventral_i, :], data[ecg_i, :]), axis = 0)
chan_list_aux = ['nasal', 'ventral', 'ECG']

data = data_rmv_second.copy()
chan_list = chan_list_rmv_second.copy()

del data_rmv_first
del data_rmv_second

print('#### ECG DETECTION ####')
    
#### adjust ECG
if sujet_ecg_adjust.get(sujet) == 'inverse':
    data_aux[-1,:] = data_aux[-1,:] * -1

#### notch ECG
ch_types = ['misc'] * (np.size(data_aux,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

info_aux = mne.create_info(chan_list_aux, srate, ch_types=ch_types)
raw_aux = mne.io.RawArray(data_aux, info_aux)

raw_aux.notch_filter(50, picks='misc', verbose='critical')

# ECG
event_id = 999
ch_name = 'ECG'
qrs_threshold = .5 #between o and 1
ecg_events = mne.preprocessing.find_ecg_events(raw_aux, event_id=event_id, ch_name=ch_name, qrs_threshold=qrs_threshold, verbose='critical')
ecg_events_time = list(ecg_events[0][:,0])

data_aux_final = raw_aux.get_data()
data_aux = data_aux_final.copy()

del data_aux_final

print('#### ADJUST TRIG ####')


trig_name = ['CV_start', 'CV_stop', '31',   '32',   '11',   '12',   '71',   '72',   '11',   '12',   '51',    '52',    '11',    '12',    '51',    '52',    '31',    '32']
trig_time = [0,          153600,    463472, 555599, 614615, 706758, 745894, 838034, 879833, 971959, 1009299, 1101429, 1141452, 1233580, 1285621, 1377760, 1551821, 1643948]

trig_load = {'name' : trig_name, 'time' : trig_time}
trig = pd.DataFrame(trig_load)  


#### add
ecg_events_corrected = [339111, 347393, 358767, 360242, 363559, 460709, 554965, 870178, 871428, 873406, 1142520, 1298203, 1297285, 1297760]
ecg_events_time += ecg_events_corrected
ecg_events_time.sort()
#### remove
ecg_events_to_remove = []
[ecg_events_time.remove(i) for i in ecg_events_to_remove]    


print('#### DONE ####')






print(data.shape)
print(chan_list)
print(data_aux.shape)
print(chan_list_aux)
print(trig)









print('#### SELECT COND ####')

cond = 'FR_CV'

trig_cond = conditions_trig[cond]
trig_time = [trig['time'][trig['name'] == trig_i].values[0] for trig_i in trig_cond]

data_select = data[:, trig_time[0]:trig_time[1]]
data_aux_select = data_aux[:, trig_time[0]:trig_time[1]]




print(data_select.shape)
print(chan_list)
print(data_aux_select.shape)
print(chan_list_aux)
print(trig)















data = data_select.copy()
data_aux = data_aux_select.copy()












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

prep_step = prep_step_lf

















print('#### PREPROCESSING ####')

# 1. Generate raw structures

ch_types = ['seeg'] * (np.size(data,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

info = mne.create_info(chan_list, srate, ch_types=ch_types)
raw_init = mne.io.RawArray(data, info)

if debug == True :
    for nchan in range(np.size(raw_init.get_data(),0)):

        nwind = int( 10*srate ) 
        nfft = nwind 
        noverlap = np.round(nwind/2) 
        hannw = scipy.signal.windows.hann(nwind)

        hzPxx, Pxx = scipy.signal.welch(raw_init.get_data()[nchan,:], fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

        plt.semilogy(hzPxx, Pxx)
        plt.title(chan_list[nchan])
        #plt.show()
    #plt.show()
    plt.close()

# 2. Initiate preprocessing step


def mean_centered_detrend(raw):

    data = raw.get_data()

    # mean centered
    data_mc = np.zeros((np.size(data,0),np.size(data,1)))
    for chan in range(np.size(data,0)):
        data_mc[chan,:] = data[chan,:] - np.mean(data[chan,:])
        data_mc[chan,:] = scipy.signal.detrend(data_mc[chan,:])

    # fill raw
    for chan in range(np.size(data,0)):
        raw[chan,:] = data_mc[chan,:]

    del data_mc    

    # verif
    if debug == True :
        # all data
        duration = .5
        n_chan = 10
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

        # compare before after
        compare_pre_post(raw.get_data(), raw.get_data(), 4)


    return raw








def line_noise_removing(raw):

    linenoise_freq = [50, 100, 150]

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    raw_post.notch_filter(50, verbose='critical')

    return raw_post





def high_pass(raw, h_freq, l_freq):

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    #filter_length = int(srate*10) # give sec
    filter_length = 'auto'

    if debug == True :
        h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
        flim = (0.1, srate / 2.)
        mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

    raw_eeg_mc_hp = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2', verbose='critical')

    if debug == True :
        duration = 60.
        n_chan = 20
        raw_eeg_mc_hp.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

    return raw_post






def low_pass(raw, h_freq, l_freq):

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    filter_length = int(srate*10) # in samples

    if debug == True :
        h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
        flim = (0.1, srate / 2.)
        mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

    raw_post = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hann', fir_design='firwin2', verbose='critical')

    if debug == True :
        duration = .5
        n_chan = 10
        raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


    return raw_post





def average_reref(raw):

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    raw_post.set_eeg_reference('average')

    if debug == True :
        duration = .5
        n_chan = 10
        raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


    return raw_post



# 3. Execute preprocessing 

if debug:
    raw = raw_init.copy() # first data
else:
    raw = raw_init

if prep_step.get('mean_centered_detrend').get('execute') :
    print('mean_centered_detrend')
    raw_post = mean_centered_detrend(raw)
    #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
    raw = raw_post


if prep_step.get('line_noise_removing').get('execute') :
    print('line_noise_removing')
    raw_post = line_noise_removing(raw)
    #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
    raw = raw_post


if prep_step.get('high_pass').get('execute') :
    print('high_pass')
    h_freq = prep_step.get('high_pass').get('params').get('h_freq')
    l_freq = prep_step.get('high_pass').get('params').get('l_freq')
    raw_post = high_pass(raw, h_freq, l_freq)
    #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
    raw = raw_post

if prep_step.get('low_pass').get('execute') :
    print('low_pass')
    h_freq = prep_step.get('low_pass').get('params').get('h_freq')
    l_freq = prep_step.get('low_pass').get('params').get('l_freq')
    raw_post = low_pass(raw, h_freq, l_freq)
    #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
    raw = raw_post

if prep_step.get('average_reref').get('execute') :
    print('average_reref')
    raw_post = average_reref(raw)
    #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
    raw = raw_post

data_preproc = raw.get_data()



print("DONE")




















data_tmp = data_preproc.copy()
band_prep = 'lf'
cond = 'FR_CV'
session_i = 1
n_surrogates_coh = 10

nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

conditions = ['FR_CV']

os.chdir(os.path.join(path_respfeatures, sujet, 'RESPI'))
respfeatures_listdir = os.listdir()

#### remove fig0 and fig1 file
respfeatures_listdir_clean = []
for file in respfeatures_listdir :
    if file.find('fig') == -1 :
        respfeatures_listdir_clean.append(file)

#### get respi features
respfeatures_allcond = {}

for cond in conditions:

    load_i = []
    for session_i, session_name in enumerate(respfeatures_listdir_clean):
        if session_name.find(cond) > 0:
            load_i.append(session_i)
        else:
            continue

    load_list = [respfeatures_listdir_clean[i] for i in load_i]

    data = []
    for load_name in load_list:
        data.append(pd.read_excel(load_name))

    respfeatures_allcond[cond] = data




def get_shuffle(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift



respi = data_aux[0,:]

hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
hzCxy = hzCxy[mask_hzCxy]

Cxy_surrogates = np.zeros((np.size(data_tmp,0),len(hzCxy)))


def compute_surrogates_coh_n_chan(n_chan):

    if n_chan/np.size(data_tmp,0) % .2 <= .01:
        print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

    x = data_tmp[n_chan,:]
    y = respi

    surrogates_val_tmp = np.zeros((n_surrogates_coh,len(hzCxy)))
    for surr_i in range(n_surrogates_coh):
        
        #if surr_i%100 == 0:
        #    print(surr_i) 

        x_shift = get_shuffle(x)
        #y_shift = get_shuffle(y)
        hzCxy_tmp, Cxy = scipy.signal.coherence(x_shift, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

        surrogates_val_tmp[surr_i,:] = Cxy[mask_hzCxy]

    surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
    percentile_i = int(np.floor(n_surrogates_coh*percentile_coh))
    compute_surrogates_coh_tmp = surrogates_val_tmp_sorted[percentile_i,:]

    return compute_surrogates_coh_tmp

compute_surrogates_coh_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_coh_n_chan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

for n_chan in range(np.size(data_tmp,0)):

    Cxy_surrogates[n_chan,:] = compute_surrogates_coh_results[n_chan]

#Cxy_surrogates.shape





CycleFreq_surrogates = np.zeros((3,np.size(data_tmp,0), stretch_point_surrogates))

respfeatures_i = respfeatures_allcond[cond][0]

def compute_surrogates_cyclefreq_nchan(n_chan):

    if n_chan/np.size(data_tmp,0) % .2 <= .01:
        print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

    x = data_tmp[n_chan,:]

    surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq,stretch_point_surrogates))
    for surr_i in range(n_surrogates_cyclefreq):
        
        #if surr_i%100 == 0:
        #    print(surr_i)

        x_shift = get_shuffle(x)
        #y_shift = get_shuffle(y)

        x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_shift, srate)

        x_stretch_mean = np.mean(x_stretch, axis=0)

        surrogates_val_tmp[surr_i,:] = x_stretch_mean

    mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
    surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
    percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
    percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

    up_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_up,:]
    dw_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_dw,:]

    return mean_surrogate_tmp, up_percentile_values_tmp, dw_percentile_values_tmp

compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

for n_chan in range(np.size(data_tmp,0)):

    CycleFreq_surrogates[0,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][0]
    CycleFreq_surrogates[1,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][1]
    CycleFreq_surrogates[2,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][2]

#CycleFreq_surrogates



os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy', Cxy_surrogates)
np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy', CycleFreq_surrogates)
















band_prep_i, band_prep = 0, 'hf'
freq_band = freq_band_list[band_prep_i]
session_i = 0

#band, freq = list(freq_band.items())[0]
for band, freq in freq_band.items():

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    
    print(band, ' : ', freq)
    print('COMPUTE')

    #### select wavelet parameters
    if band_prep == 'lf':
        wavetime = np.arange(-2,2,1/srate)
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

    if band_prep == 'hf':
        wavetime = np.arange(-.5,.5,1/srate)
        nfrex = nfrex_hf
        ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

    #### compute wavelets
    frex  = np.linspace(freq[0],freq[1],nfrex)
    wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

    # create Morlet wavelet family
    for fi in range(0,nfrex):
        
        s = ncycle_list[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw
        
    # plot all the wavelets
    if debug == True:
        plt.pcolormesh(wavetime,frex,np.real(wavelets))
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Real part of wavelets')
        plt.show()

    tf_allchan = np.zeros((np.size(data_tmp,0), nfrex, np.size(data_tmp,1)))

    def compute_tf_convolution_nchan(n_chan):

        if n_chan/np.size(data_tmp,0) % .2 <= .01:
            print("{:.2f}".format(n_chan/np.size(data_tmp,0)))
        x = data_tmp[n_chan,:]

        tf = np.zeros((nfrex,np.size(x)))

        for fi in range(nfrex):
            
            tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

        tf_allchan[n_chan,:,:] = tf

        return

    for n_chan in range(np.size(data,0)):
        compute_tf_convolution_nchan(n_chan)

    #### stretch
    print('STRETCH')

    tf = tf_allchan

    baseline = np.mean(tf, axis=2)
    baseline = np.transpose(baseline)

    for n_chan in range(np.size(tf,0)):
        
        for fi in range(np.size(tf,1)):

            activity = tf[n_chan,fi,:]
            baseline_fi = baseline[fi,n_chan]

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        if n_chan/np.size(tf,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(tf,0)))

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x, srate)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        return tf_mean

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    for n_chan in range(np.size(tf,0)):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_n_chan(n_chan)    

    #### save
    print('SAVE')
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    np.save(sujet + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy', tf_mean_allchan)









################################
######## PRECOMPUTE ITPC ########
################################



print('ITPC PRECOMPUTE')
    

freq_band = freq_band_list[band_prep_i]

#band, freq = list(freq_band.items())[0]
for band, freq in freq_band.items():

    os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

    print(band, ' : ', freq)

    #### select wavelet parameters
    if band_prep == 'lf':
        wavetime = np.arange(-2,2,1/srate)
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

    if band_prep == 'hf':
        wavetime = np.arange(-.5,.5,1/srate)
        nfrex = nfrex_hf
        ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

    #### compute wavelets
    frex  = np.linspace(freq[0],freq[1],nfrex)
    wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

    # create Morlet wavelet family
    for fi in range(0,nfrex):
        
        s = ncycle_list[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw
        
    # plot all the wavelets
    if debug == True:
        plt.pcolormesh(wavetime,frex,np.real(wavelets))
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Real part of wavelets')
        plt.show()

    #### compute itpc
    print('COMPUTE, STRETCH & ITPC')
    def compute_itpc_n_chan(n_chan):

        if n_chan/np.size(data_tmp,0) % .2 <= .01:
            print("{:.2f}".format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]

        tf = np.zeros((nfrex,np.size(x)), dtype='complex')

        for fi in range(nfrex):
            
            tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        #### stretch
        
        def compute_stretch_tf_itpc(tf, cond, session_i, respfeatures_allcond, stretch_point_TF):
    
            #### identify number stretch
            x = tf[0,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x, srate)
            nb_cycle = np.size(x_stretch, 0)
            
            #### compute tf
            tf_stretch = np.zeros((nb_cycle, np.size(tf,0), int(stretch_point_TF)), dtype='complex')

            for fi in range(np.size(tf,0)):

                x = tf[fi,:]
                x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x, srate)
                tf_stretch[:,fi,:] = x_stretch

            return tf_stretch
        
        tf_stretch = compute_stretch_tf_itpc(tf, cond, session_i, respfeatures_allcond, stretch_point_TF)

        #### ITPC
        tf_angle = np.angle(tf_stretch)
        tf_cangle = np.exp(1j*tf_angle) 
        itpc = np.abs(np.mean(tf_cangle,0))

        if debug == True:
            time = range(stretch_point_TF)
            frex = range(nfrex)
            plt.pcolormesh(time,frex,itpc,vmin=np.min(itpc),vmax=np.max(itpc))
            plt.show()

        return itpc 
    
    itpc_allchan = np.zeros((np.size(data_tmp,0),nfrex,stretch_point_TF))

    for n_chan in range(np.size(data_tmp,0)):

        itpc_allchan[n_chan,:,:] = compute_itpc_n_chan(n_chan)

    #### save
    print('SAVE')
    os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
    np.save(sujet + '_itpc_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy', itpc_allchan)





























