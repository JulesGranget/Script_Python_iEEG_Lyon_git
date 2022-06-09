
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import joblib
import glob
import neo

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






########################################
######## DATA EXTRACTION ######## 
########################################

#path_data, sujet = 'D:\LPPR_CMO_PROJECT\Lyon\Data\iEEG', 'LYONNEURO_2019_CAPp'
def extract_data_trc(sujet):

    os.chdir(os.path.join(path_data,sujet))

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

    #### verif
    #file_i = 1
    #data = data_whole[file_i]
    #chan_list = chan_list_whole[file_i]
    #events_time = events_time_whole[file_i]
    #srate = srate_whole[file_i]

    #chan_name = 'p19+'
    #chan_i = chan_list.index(chan_name)
    #file_stop = (np.size(data,1)/srate)/60
    #start = 0 *60*srate 
    #stop = int( file_stop *60*srate )
    #plt.plot(data[chan_i,start:stop])
    #plt.vlines( np.array(events_time)[(np.array(events_time) > start) & (np.array(events_time) < stop)], ymin=np.min(data[chan_i,start:stop]), ymax=np.max(data[chan_i,start:stop]))
    #plt.show()

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

    
    #### no more use
    del data_whole
    del data_whole_file
    del data_file
    
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

    os.chdir(os.path.join(path_anatomy, sujet))

    #### identify iEEG / respi / ECG

    print('#### AUX IDENTIFICATION ####')
    nasal_i = chan_list.index(aux_chan.get(sujet).get('nasal'))
    ecg_i = chan_list.index(aux_chan.get(sujet).get('ECG'))
    
    if aux_chan.get(sujet).get('ventral') == None:
        _data_ventral = np.zeros((data[nasal_i, :].shape[0]))
        data_aux = np.stack((data[nasal_i, :], _data_ventral, data[ecg_i, :]), axis = 0)
    else:
        ventral_i = chan_list.index(aux_chan.get(sujet).get('ventral'))
        data_aux = np.stack((data[nasal_i, :], data[ventral_i, :], data[ecg_i, :]), axis = 0)

    chan_list_aux = ['nasal', 'ventral', 'ECG']

    data = data_rmv_second.copy()
    chan_list = chan_list_rmv_second.copy()

    del data_rmv_first
    del data_rmv_second

    return data, chan_list, srate













########################################
######## COMPUTE BASELINE ######## 
########################################

#sujet, band_prep = 'pat_03083_1527', 'lf'
def compute_and_save_baseline(sujet, band_prep):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    verif_band_compute = []
    for band in list(freq_band_dict[band_prep].keys()):
        if os.path.exists(os.path.join(path_precompute, sujet, 'baselines', f'{sujet}_{band}_baselines.npy')):
            verif_band_compute.append(True)

    if np.sum(verif_band_compute) > 0:
        print(f'{sujet} : BASELINES ALREADY COMPUTED')
        return
            

    #### open raw and section if sujet from paris
    if sujet in sujet_list_paris_only_FR_CV:
        data = load_data_sujet(sujet, band_prep, 'FR_CV', 0)
        srate = get_srate(sujet)
    elif sujet[:3] == 'pat' and sujet not in sujet_list_paris_only_FR_CV:
        os.chdir(os.path.join(path_data, sujet, 'raw_data', sujet))
        raw = mne.io.read_raw_eeglab(f'{sujet}_allchan.set', preload=True)
        data, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw(raw)
    else:
        data, chan_list, srate = extract_data_trc(sujet)

    #### generate all wavelets to conv
    wavelets_to_conv = {}
        
    #### select wavelet parameters
    if band_prep == 'lf' or band_prep == 'wb':
        wavetime = np.arange(-2,2,1/srate)
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

    if band_prep == 'hf':
        wavetime = np.arange(-.5,.5,1/srate)
        nfrex = nfrex_hf
        ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

    if band_prep == 'wb':
        wavetime = np.arange(-2,2,1/srate)
        nfrex = nfrex_hf
        ncycle_list = np.linspace(ncycle_list_wb[0], ncycle_list_wb[1], nfrex)

    #band, freq = 'theta', [2, 10]
    for band, freq in freq_band_dict[band_prep].items():

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

        wavelets_to_conv[band] = wavelets

    # plot all the wavelets
    if debug == True:
        for band in list(wavelets_to_conv.keys()):
            wavelets2plot = wavelets_to_conv[band]
            plt.pcolormesh(np.arange(wavelets2plot.shape[1]),np.arange(wavelets2plot.shape[0]),np.real(wavelets2plot))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(band)
            plt.show()

    #### compute convolutions
    n_band_to_compute = len(list(freq_band_dict[band_prep].keys()))

    os.chdir(path_memmap)
    baseline_allchan = np.memmap(f'{sujet}_baseline_convolutions_{band_prep}.dat', dtype=np.float64, mode='w+', shape=(n_band_to_compute, data.shape[0], nfrex))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        if n_chan/np.size(data,0) % .2 <= .01:
            print("{:.2f}".format(n_chan/np.size(data,0)))

        x = data[n_chan,:]

        for band_i, band in enumerate(list(wavelets_to_conv.keys())):

            baseline_coeff_band = np.array(())

            for fi in range(nfrex):
                
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2
                baseline_coeff_band = np.append(baseline_coeff_band, np.median(fi_conv))
        
            baseline_allchan[band_i, n_chan,:] = baseline_coeff_band

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data,0)))

    #### save baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

    for band_i, band in enumerate(list(freq_band_dict[band_prep].keys())):
    
        np.save(f'{sujet}_{band}_baselines.npy', baseline_allchan[band_i, :, :])

    #### remove memmap
    os.chdir(path_memmap)
    os.remove(f'{sujet}_baseline_convolutions_{band_prep}.dat')

    print('done')
    print(sujet)




################################
######## EXECUTE ########
################################


if __name__== '__main__':


    #### compute
    #compute_and_save_baseline(sujet, band_prep)
    
    #### slurm execution
    #### possibility to launch several subjects simultaneously
    for band_prep in band_prep_list:
        execute_function_in_slurm_bash('n5_precompute_baselines', 'compute_and_save_baseline', [sujet, band_prep])

