
import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import joblib
import physio

import mne
import scipy.fftpack
import scipy.signal
from sklearn.utils import resample

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n2bis_prep_adjust_values import *


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
    if debug:
        file_i = 2
        data = data_whole[file_i]
        chan_list = chan_list_whole[file_i]
        events_time = events_time_whole[file_i]
        srate = srate_whole[file_i]

        chan_name = 'p17+'
        chan_i = chan_list.index(chan_name)
        plt.plot(data[chan_i,:])
        plt.vlines( np.array(events_time), ymin=np.min(data[chan_i,start:stop]), ymax=np.max(data[chan_i,start:stop]), color='r')
        plt.show()

        chan_name = 'p17+'
        chan_i = chan_list.index(chan_name)
        file_stop = (np.size(data,1)/srate)/60
        start = 0 *60*srate 
        stop = int( file_stop *60*srate )
        plt.plot(data[chan_i,start:stop])
        plt.vlines( np.array(events_time)[(np.array(events_time) > start) & (np.array(events_time) < stop)], ymin=np.min(data[chan_i,start:stop]), ymax=np.max(data[chan_i,start:stop]), color='r')
        plt.show()

    #### concatenate 
    print('#### CONCATENATE ####')
    data = data_whole[0]
    chan_list = chan_list_whole[0]
    events_name = events_name_whole[0]
    events_time = events_time_whole[0]
    srate = srate_whole[0]

    if len(trc_file_names) > 1 :
        #trc_i = 0
        for trc_i, trc_name in enumerate(trc_file_names_ordered): 

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

    #### verif
    if debug:

        chan_name = 'p17+'
        chan_i = chan_list.index(chan_name)
        plt.plot(data[chan_i,:])
        plt.vlines( np.array(events_time), ymin=np.min(data[chan_i,start:stop]), ymax=np.max(data[chan_i,start:stop]), color='r')
        plt.show()

    
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

    add_in_csv_textfile = open(sujet + "_add_in_csv.txt", "w")
    for element in chan_list_add_in_csv:
        add_in_csv_textfile.write(element + "\n")
    add_in_csv_textfile.close()

    #### indicate removed chan
    print('verification nchan out first:')
    print(chan_list_nchan_rmv_first)
    print('')
    print('verification nchan out second:')
    print(chan_list_nchan_rmv_second)

    chan_list_nchan_rmv_first_textfile = open(sujet + "_first_rmv.txt", "w")
    for element in chan_list_nchan_rmv_first:
        chan_list_nchan_rmv_first_textfile.write(element + "\n")
    chan_list_nchan_rmv_first_textfile.close()

    chan_list_nchan_rmv_second_textfile = open(sujet + "_second_rmv.txt", "w")
    for element in chan_list_nchan_rmv_second:
        chan_list_nchan_rmv_second_textfile.write(element + "\n")
    chan_list_nchan_rmv_second_textfile.close()

    #### chan list all rmw
    chan_list_all_rmw = chan_list_nchan_rmv_first + chan_list_nchan_rmv_second

    #### identify iEEG / respi / ECG
    print('#### AUX IDENTIFICATION ####')
    nasal_i = chan_list.index(aux_chan[sujet]['nasal'])
    ecg_i = chan_list.index(aux_chan.get(sujet).get('ECG'))
    
    if aux_chan.get(sujet).get('ventral') == None:
        _data_ventral = np.zeros((data[nasal_i, :].shape[0]))
        data_aux = np.stack((data[nasal_i, :], _data_ventral, data[ecg_i, :]), axis = 0)
    else:
        ventral_i = chan_list.index(aux_chan.get(sujet).get('ventral'))
        data_aux = np.stack((data[nasal_i, :], data[ventral_i, :], data[ecg_i, :]), axis = 0)

    chan_list_aux = ['nasal', 'ventral', 'ECG']

    if sujet_respi_adjust[sujet] == 'inverse':
        data_aux[0, :] *= -1

    data = data_rmv_second.copy()
    chan_list = chan_list_rmv_second.copy()

    del data_rmv_first
    del data_rmv_second

    return data, chan_list, data_aux, chan_list_aux, chan_list_all_rmw, trig, srate






#path_data, sujet = 'D:\LPPR_CMO_PROJECT\Lyon\Data\iEEG', 'LYONNEURO_2019_CAPp'
def extract_data_trc_bi(sujet):

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
        for trc_i in range(len(trc_file_names_ordered)): 

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

    #### identify iEEG / respi / ECG
    print('#### AUX IDENTIFICATION ####')
    nasal_i = chan_list.index(aux_chan[sujet]['nasal'])
    ecg_i = chan_list.index(aux_chan.get(sujet).get('ECG'))
    
    if aux_chan.get(sujet).get('ventral') == None:
        _data_ventral = np.zeros((data[nasal_i, :].shape[0]))
        data_aux = np.stack((data[nasal_i, :], _data_ventral, data[ecg_i, :]), axis = 0)
    else:
        ventral_i = chan_list.index(aux_chan.get(sujet).get('ventral'))
        data_aux = np.stack((data[nasal_i, :], data[ventral_i, :], data[ecg_i, :]), axis = 0)

    chan_list_aux = ['nasal', 'ventral', 'ECG']

    if sujet_respi_adjust[sujet] == 'inverse':
        data_aux[0, :] *= -1

    data = data_rmv_first.copy()
    chan_list = chan_list_rmv_first.copy()

    #### identify pair available for bipol channels
    os.chdir(os.path.join(path_anatomy, sujet))

    chan_list = np.array(modify_name(chan_list)[0])
    
    plot_loca_df_bi = pd.read_excel(sujet + '_plot_loca_bi.xlsx')

    plot_select_bi = plot_loca_df_bi['plot'][plot_loca_df_bi['select'] == 1]

    pairs_bi_available_i = np.array([], dtype='int64')
    pairs_bi_removed = np.array([])

    for row_i, bi_plot_selected_i in enumerate(plot_loca_df_bi['plot']): 
        plot_A, plot_B = bi_plot_selected_i.split('-')[0], bi_plot_selected_i.split('-')[-1]
        if np.sum(chan_list == plot_A) == 0 or np.sum(chan_list == plot_B) == 0:
            pairs_bi_removed = np.append(pairs_bi_removed, bi_plot_selected_i)
            plot_loca_df_bi['select'][row_i] = 0
            continue
        else:
            pairs_bi_available_i = np.append(pairs_bi_available_i, row_i)

    print('#### PAIRS REMOVED : ####')
    print(pairs_bi_removed)

    #### save df updated with removed pairs
    plot_loca_df_bi.to_excel(f'{sujet}_plot_loca_bi.xlsx')

    #### generate data bi
    data_ieeg_bi = np.zeros((pairs_bi_available_i.shape[0], data.shape[-1]))

    for row_i, bi_plot_selected_i in enumerate(pairs_bi_available_i): 
        plot_A, plot_B = plot_loca_df_bi['plot'][bi_plot_selected_i].split('-')[0], plot_loca_df_bi['plot'][bi_plot_selected_i].split('-')[-1]
        plot_A_i, plot_B_i = np.where(chan_list == plot_A)[0][0], np.where(chan_list == plot_B)[0][0]
        data_ieeg_bi[row_i, :] = data[plot_A_i, :] - data[plot_B_i, :]

        #### verify
        if debug:

            plt.plot(data[plot_A_i, :] - data[plot_B_i, :])
            plt.show()

    #### 2nd removing
    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca_bi.xlsx')

    chan_list_ieeg_keep = file_plot_select['plot'][file_plot_select['select'] == 1].values.tolist()

    remove_second = []

    chan_list_bi = plot_loca_df_bi['plot'][pairs_bi_available_i].values

    #nchan = chan_list_bi[0]
    for nchan in chan_list_bi:

        if nchan not in chan_list_ieeg_keep:
            remove_second.append(nchan)
            rmv_i = np.where(chan_list_bi == nchan)[0][0]
            chan_list_bi = np.delete(chan_list_bi, rmv_i)
            data_ieeg_bi = np.delete(data_ieeg_bi, rmv_i, 0)

    print('#### SECOND REMOVE ####')
    print(remove_second)    

    return data_ieeg_bi, chan_list_bi.tolist(), data_aux, chan_list_aux, trig, srate





def import_raw_data(sujet):

    os.chdir(os.path.join(path_data, sujet))

    if sujet == 'pat_03146_1608':

        raw_1 = mne.io.read_raw_eeglab(f'{sujet}_1_allchan.set')
        raw_2 = mne.io.read_raw_eeglab(f'{sujet}_2_allchan.set')

        mne.rename_channels(raw_2.info, {ch_B : ch_A for ch_B, ch_A in zip(raw_2.info['ch_names'], raw_1.info['ch_names'])})

        raw = mne.concatenate_raws([raw_1, raw_2], preload=True)

        del raw_1, raw_2

    else:

        raw = mne.io.read_raw_eeglab(f'{sujet}_allchan.set', preload=True)

    return raw






def organize_raw(sujet, raw):

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

    #### inspi down
    if sujet_respi_adjust[sujet] == 'inverse':
        data_aux[0, :] *= -1

    #### remove from data
    data_ieeg = data.copy()

    del data

    #### remove other aux
    for aux_name in aux_chan[sujet].keys():

        aux_i = chan_list_clean.index(aux_chan[sujet][aux_name])
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_clean.remove(aux_chan[sujet][aux_name])

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']
    chan_list_ieeg = chan_list_clean

    #### remove chan that are not in parcelisation
    os.chdir(os.path.join(path_anatomy, sujet))
    
    plot_loca_df = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_not_in_parcel = []
    chan_i_rmv = 0
    #chan_i, chan_name = 0, chan_list_ieeg[0]
    for chan_i, chan_name in enumerate(chan_list_ieeg):

        if chan_name not in plot_loca_df['plot'].values:

            chan_not_in_parcel.append(chan_name)
            data_ieeg = np.delete(data_ieeg, chan_i-chan_i_rmv, axis=0)

            chan_i_rmv += 1

    [chan_list_ieeg.remove(chan_name) for chan_name in chan_not_in_parcel]

    #### filter channel from plot_loca
    chan_to_reject = plot_loca_df['plot'][plot_loca_df['select'] == 0].values

    chan_i_to_remove = [chan_list_ieeg.index(nchan) for nchan in chan_list_ieeg if nchan in chan_to_reject]

    data_ieeg_rmv = data_ieeg.copy()
    chan_list_ieeg_rmv = chan_list_ieeg.copy()
    rmv_count = 0
    for remove_i in chan_i_to_remove:
        remove_i_adj = remove_i - rmv_count
        data_ieeg_rmv = np.delete(data_ieeg_rmv, remove_i_adj, 0)
        rmv_count += 1

        chan_list_ieeg_rmv.remove(chan_list_ieeg[remove_i])

    if data_ieeg.shape[0] - len(chan_i_to_remove) != data_ieeg_rmv.shape[0]:
        raise ValueError('problem on chan selection from plot_loca')

    if data_ieeg.shape[1] != data_ieeg_rmv.shape[1]:
        raise ValueError('problem on chan selection from plot_loca')

    data_ieeg = data_ieeg_rmv.copy()
    chan_list_ieeg = chan_list_ieeg_rmv.copy()

    del data_ieeg_rmv

    return data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate




#plot = plot_A
def adjust_plot_name(plot):

    if len(plot.split('_')[-1]) == 1:
        plot_adjusted = f"{plot.split('_')[0]}_0{plot.split('_')[-1]}"

        return plot_adjusted

    else:

        return plot


def organize_raw_bi(sujet, raw):

    #### extract chan_list
    chan_list_clean = []
    chan_list = raw.info['ch_names']
    srate = int(raw.info['sfreq'])
    [chan_list_clean.append(nchan[23:]) for nchan in chan_list]

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']

    #### extract data
    data = raw.get_data()

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

    del data

    #### bipol channels
    os.chdir(os.path.join(path_anatomy, sujet))
    
    plot_loca_df_bi = pd.read_excel(sujet + '_plot_loca_bi.xlsx')
    plot_loca_df_mono = pd.read_excel(sujet + '_plot_loca.xlsx')
    plot_loca_df = plot_loca_df_mono.copy()

    for row_i in plot_loca_df.index:
        plot_name, plot_i = plot_loca_df['plot'][row_i].split('_')[0], plot_loca_df['plot'][row_i].split('_')[-1]
        if len(plot_i) == 1:
            plot_loca_df['plot'][row_i] = f'{plot_name}_0{plot_i}'

    data_ieeg_bi = np.zeros((plot_loca_df_bi.index.shape[0], data_ieeg.shape[-1]))

    for row_i in plot_loca_df_bi.index: 
        plot_A, plot_B = plot_loca_df_bi['plot'][row_i].split('-')[0], plot_loca_df_bi['plot'][row_i].split('-')[-1]
        plot_A, plot_B = adjust_plot_name(plot_A), adjust_plot_name(plot_B)
        plot_A_i, plot_B_i = np.where(plot_loca_df['plot'] == plot_A)[0][0], np.where(plot_loca_df['plot'] == plot_B)[0][0]
        data_ieeg_bi[row_i, :] = data_ieeg[plot_A_i, :] - data_ieeg[plot_B_i, :]

        #### verify
        if debug:

            plt.plot(data_ieeg[plot_A_i, :] - data_ieeg[plot_B_i, :])
            plt.show()
    
    #### filter channel from plot_loca
    chan_list_ieeg = plot_loca_df_bi['plot'].values.tolist()
    chan_to_reject = plot_loca_df_bi['plot'][plot_loca_df_bi['select'] == 0].values

    chan_i_to_remove = [chan_list_ieeg.index(nchan) for nchan in chan_list_ieeg if nchan in chan_to_reject]

    data_ieeg_rmv = data_ieeg_bi.copy()
    chan_list_ieeg_rmv = chan_list_ieeg.copy()
    rmv_count = 0
    for remove_i in chan_i_to_remove:
        remove_i_adj = remove_i - rmv_count
        data_ieeg_rmv = np.delete(data_ieeg_rmv, remove_i_adj, 0)
        rmv_count += 1

        chan_list_ieeg_rmv.remove(chan_list_ieeg[remove_i])

    if data_ieeg_bi.shape[0] - len(chan_i_to_remove) != data_ieeg_rmv.shape[0]:
        raise ValueError('problem on chan selection from plot_loca')

    if data_ieeg_bi.shape[1] != data_ieeg_rmv.shape[1]:
        raise ValueError('problem on chan selection from plot_loca')

    data_ieeg_bi = data_ieeg_rmv.copy()
    chan_list_ieeg = chan_list_ieeg_rmv.copy()

    return data_ieeg_bi, chan_list_ieeg, data_aux, chan_list_aux, srate








################################
######## COMPARISON ########
################################


# to compare during preprocessing
def compare_pre_post(pre, post, nchan):

    # compare before after
    x_pre = pre[nchan,:]
    x_post = post[nchan,:]

    nwind = int(10*srate)
    nfft = nwind
    noverlap = np.round(nwind/2)
    hannw = scipy.signal.windows.hann(nwind)

    hzPxx, Pxx_pre = scipy.signal.welch(x_pre,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
    hzPxx, Pxx_post = scipy.signal.welch(x_post,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

    plt.plot(x_pre, label='x_pre')
    plt.plot(x_post, label='x_post')
    plt.legend()
    plt.title(chan_list[nchan])
    plt.show()

    plt.semilogy(hzPxx, Pxx_pre, label='Pxx_pre')
    plt.semilogy(hzPxx, Pxx_post, label='Pxx_post')
    plt.legend()
    plt.xlim(0,150)
    plt.title(chan_list[nchan])
    plt.show()












################################
######## PREPROCESSING ########
################################

#data, chan_list, srate = data, chan_list, srate
def preprocessing_ieeg(data, chan_list, srate, prep_step):


    print('#### PREPROCESSING ####')
    
    # 1. Generate raw structures

    ch_types = ['seeg'] * (np.size(data,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

    info = mne.create_info(chan_list, srate, ch_types=ch_types, verbose='critical')
    raw_init = mne.io.RawArray(data, info, verbose='critical')

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

    del data

    # 2. Initiate preprocessing step


    def mean_centered(raw):
        
        data = raw.get_data()
        
        # mean centered
        data_mc = np.zeros((np.size(data,0),np.size(data,1)))
        for chan in range(np.size(data,0)):
            data_mc[chan,:] = data[chan,:] - np.mean(data[chan,:])
            #### no detrend to keep low derivation
            #data_mc[chan,:] = scipy.signal.detrend(data_mc[chan,:]) 

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

        
        if debug == True :

            # compare before after
            compare_pre_post(raw.get_data(), raw_post.get_data(), 4)

    
            #test

        #data_lnr = np.zeros((np.size(raw.get_data(), 0), np.size(raw.get_data(), 0)))       
        #for nchan in range(len(chan_list)):

        #    print(chan_list[nchan]+' : {:.2f}'.format(nchan/len(chan_list)))

        #    x = raw.get_data()[nchan,:]
        #    #x_notch = mne.filter.notch_filter(x, Fs=srate, freqs=linenoise_freq, method='spectrum_fit', mt_bandwidth=2, p_value=0.01, filter_length='10s', n_jobs=-1)
        #    x_notch = mne.filter.notch_filter(x, Fs=srate, freqs=50)
        #    data_lnr[nchan,:] = x_notch

        #raw_post.get_data()[:,:] = data_lnr

        #nchan = 4
        #x = raw.get_data()[nchan,:]
        #start = 0 * 60 * srate # give min 
        #stop = 3 * 60 * srate # give min
        #x = x[start:stop]
        
        #linenoise_freq = [50, 100]

        #x_notch = mne.filter.notch_filter(x, Fs=srate, freqs=50)
        #x_notch_mt = mne.filter.notch_filter(x, Fs=srate, freqs=linenoise_freq, method='spectrum_fit', mt_bandwidth=2, p_value=0.01, filter_length='10s', n_jobs=-1)
        #x_notch_mt_none = mne.filter.notch_filter(x, Fs=srate, freqs=None, method='spectrum_fit', mt_bandwidth=2, p_value=0.05, filter_length='10s', n_jobs=-1)

        #nwind = int(10*srate)
        #nfft = nwind
        #noverlap = np.round(nwind/2)
        #hannw = scipy.signal.windows.hann(nwind)

        #hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        #hzPxx, Pxx_notch = scipy.signal.welch(x_notch,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        #hzPxx, Pxx_notch_mt = scipy.signal.welch(x_notch_mt,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        #hzPxx, Pxx_notch_mt_none = scipy.signal.welch(x_notch_mt_none,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

        #plt.semilogy(hzPxx, Pxx, label='Pxx')
        #plt.semilogy(hzPxx, Pxx_notch, label='Pxx_notch')
        #plt.semilogy(hzPxx, Pxx_notch_mt, label='Pxx_notch_mt')
        #plt.semilogy(hzPxx, Pxx_notch_mt_none, label='Pxx_notch_mt_none')
        #plt.legend()
        #plt.xlim(0,150)
        #plt.show()

    
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

    if prep_step.get('mean_centered').get('execute') :
        print('mean_centered')
        raw_post = mean_centered(raw)
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

    del raw, raw_init, raw_post

    return data_preproc







################################
######## AUX PREPROC ########
################################

def ecg_detection(data_aux, chan_list_aux, srate):

    print('#### ECG DETECTION ####')
    
    #### adjust ECG
    if sujet_ecg_adjust.get(sujet) == 'inverse':
        data_aux[1,:] = data_aux[1,:] * -1

    #### filtre
    ecg_clean = physio.preprocess(data_aux[1,:], srate, band=[5., 45.], ftype='bessel', order=5, normalize=True)

    ecg_events_time = physio.detect_peak(ecg_clean, srate, thresh=10, exclude_sweep_ms=4.0) # thresh = n MAD

    if debug:
        plt.plot(data_aux[1,:])
        plt.plot(ecg_clean)
        plt.vlines(ecg_events_time, ymin=ecg_clean.min(), ymax=ecg_clean.max(), color='r')
        plt.show()
    
    #### replace
    data_aux[1,:] = ecg_clean.copy()

    ch_types = ['misc'] * (np.size(data_aux,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

    info_aux = mne.create_info(chan_list_aux, srate, ch_types=ch_types)
    raw_aux = mne.io.RawArray(data_aux, info_aux)

    # raw_aux.notch_filter(50, picks='misc', verbose='critical')

    # ECG
    # event_id = 999
    # ch_name = 'ECG'
    # qrs_threshold = .5 #between o and 1
    # ecg_events = mne.preprocessing.find_ecg_events(raw_aux, event_id=event_id, ch_name=ch_name, qrs_threshold=qrs_threshold, verbose='critical')
    # ecg_events_time = list(ecg_events[0][:,0])

    return raw_aux, ecg_events_time




def respi_preproc(raw_aux):

    raw_aux.info['ch_names']
    srate = raw_aux.info['sfreq']
    respi = raw_aux.get_data()[0,:]

    #### inspect Pxx
    if debug:
        plt.plot(np.arange(respi.shape[0])/srate, respi)
        plt.show()

        srate = raw_aux.info['sfreq']
        nwind = int(10*srate)
        nfft = nwind
        noverlap = np.round(nwind/2)
        hannw = scipy.signal.windows.hann(nwind)
        hzPxx, Pxx = scipy.signal.welch(respi,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        plt.semilogy(hzPxx, Pxx, label='respi')
        plt.legend()
        plt.xlim(0,60)
        plt.show()

    #### filter respi   
    # fcutoff = 1.5
    # transw  = .2
    # order   = np.round( 7*srate/fcutoff )
    # if order%2==0:
    #     order += 1

    # shape   = [ 1,1,0,0 ]
    # frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]

    # filtkern = scipy.signal.firls(order,frex,shape,fs=srate)

    # respi_filt = scipy.signal.filtfilt(filtkern,1,respi)

    #### filter respi physio
    respi_filt = physio.preprocess(respi, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    respi_filt = physio.smooth_signal(respi_filt, srate, win_shape='gaussian', sigma_ms=40.0)

    if debug:
        plt.plot(respi, label='respi')
        plt.plot(respi_filt, label='respi_filtered')
        plt.legend()
        plt.show()

        hzPxx, Pxx_pre = scipy.signal.welch(respi,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        hzPxx, Pxx_post = scipy.signal.welch(respi_filt,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        plt.semilogy(hzPxx, Pxx_pre, label='pre')
        plt.semilogy(hzPxx, Pxx_post, label='post')
        plt.legend()
        plt.xlim(0,60)
        plt.show()


    #### replace respi 
    data = raw_aux.get_data()
    data[0,:] = respi_filt
    raw_aux._data = data

    #### verif
    #plt.plot(raw_aux.get_data()[0,:]),plt.show()

    return raw_aux








################################
######## CHOP & SAVE ########
################################

#data, band_preproc, export_info = data_preproc_lf, 'lf', True
def chop_save_trc(sujet, data, chan_list, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, monopol, band_preproc, export_info):

    print('#### SAVE ####')
    
    #### agregate data
    data_all = np.vstack(( data, data_aux, np.zeros(( len(data[0,:]) )) ))
    chan_list_all = chan_list + chan_list_aux + ['ECG_cR']

    #### resample if needed

    if srate != 500:

        ratio_resampling = dw_srate/srate

        dw_npnts = int( data_all.shape[1]*dw_srate/srate )

        dw_data = np.zeros(( data_all.shape[0], dw_npnts ))

        #chan_i = 0
        for chan_i in range(data_all.shape[0]):
            
            dw_data[chan_i,:] = scipy.signal.resample(data_all[chan_i,:], dw_npnts)

            if debug:
                up_times = np.arange(0,data_all.shape[1])/srate
                dw_times = np.arange(0,dw_data.shape[1])/dw_srate
                plt.plot(dw_times, dw_data[chan_i,:], label='resample')
                plt.plot(up_times, data_all[chan_i,:], label='Original')
                plt.legend()
                plt.show()

        data_all = dw_data.copy()
        srate = 500

        del dw_data

        #### resample trig
        pre_trig_resample = trig.time.values 
        post_trig = []
        for trig_i in pre_trig_resample:
            post_trig.append(int(trig_i*ratio_resampling))
        post_trig = np.array(post_trig)

        trig['time'] = post_trig

        ecg_events_time = [int(i*ratio_resampling) for i in ecg_events_time]

    ch_types = ['seeg'] * (len(chan_list_all)-4) + ['misc'] * 4
    info = mne.create_info(chan_list_all, srate, ch_types=ch_types, verbose='critical')
    raw_all = mne.io.RawArray(data_all, info, verbose='critical')

    if debug:
        for i in range(data_all.shape[0]):
            plt.plot(data_all[i,:]+i)
        plt.show()

    del data
    del data_all

    #### save chan_list
    os.chdir(os.path.join(path_anatomy, sujet))
    if monopol:
        keep_plot_textfile = open(sujet + "_chanlist_ieeg.txt", "w")
    else:
        keep_plot_textfile = open(sujet + "_chanlist_ieeg_bi.txt", "w")
    for element in chan_list_all[:-4]:
        keep_plot_textfile.write(element + "\n")
    keep_plot_textfile.close()

    #### add cR events
    event_cR = np.zeros((len(ecg_events_time),3))
    for cR in range(len(ecg_events_time)):
        event_cR[cR, 0] = ecg_events_time[cR]
        event_cR[cR, 2] = 10

    raw_all.add_events(event_cR, stim_channel='ECG_cR', replace=True)
    raw_all.info['ch_names']

    #### save chunk
    count_session = {
    'RD_CV' : 0,
    'RD_FV' : 0,
    'RD_SV' : 0,
    'RD_AV' : 0,
    'FR_CV' : 0,
    'FR_MV' : 0,
    }

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    #condition, trig_cond = list(conditions_trig.items())[0]
    for condition, trig_cond in conditions_trig.items():

        if sujet not in sujet_list and condition != 'FR_CV':
            continue

        cond_i = np.where(trig.name.values == trig_cond[0])[0]
        #export_i, trig_i = 0, cond_i[0]
        for export_i, trig_i in enumerate(cond_i):

            count_session[condition] = count_session[condition] + 1 

            raw_chunk = raw_all.copy()
            raw_chunk.crop( tmin = (trig.iloc[trig_i,:].time)/srate , tmax= (trig.iloc[trig_i+1,:].time/srate)-0.2 )
            if monopol:
                raw_chunk.save(f'{sujet}_{condition}_{str(export_i+1)}_{band_preproc}.fif')
            else:
                raw_chunk.save(f'{sujet}_{condition}_{str(export_i+1)}_{band_preproc}_bi.fif')
            
            del raw_chunk

    if monopol:
        raw_all.save(f'{sujet}_allcond_{band_preproc}.fif')
    else:
        raw_all.save(f'{sujet}_allcond_{band_preproc}_bi.fif')

    df = {'condition' : list(count_session.keys()), 'count' : list(count_session.values())}
    count_session = pd.DataFrame(df, columns=['condition', 'count'])

    if export_info == True :
    
        #### export trig, count_session, cR
        os.chdir(os.path.join(path_prep, sujet, 'info'))
        
        trig.to_excel(sujet + '_trig.xlsx')

        count_session.to_excel(sujet + '_count_session.xlsx')

        cR = pd.DataFrame(ecg_events_time, columns=['cR_time'])
        cR.to_excel(sujet +'_cR_time.xlsx')

    del raw_all

    return 




























################################
######## EXECUTE ########
################################


if __name__== '__main__':

    #### whole protocole
    # sujet = 'CHEe'
    # sujet = 'GOBc' 
    # sujet = 'MAZm' 
    # sujet = 'TREt' 
    # sujet = 'POTm'

    #### FR_CV only
    # sujet = 'KOFs'
    # sujet = 'MUGa'
    # sujet = 'BANc'
    # sujet = 'LEMl'

    # sujet = 'pat_02459_0912'
    # sujet = 'pat_02476_0929'
    # sujet = 'pat_02495_0949'

    # sujet = 'pat_03083_1527'
    # sujet = 'pat_03105_1551'
    # sujet = 'pat_03128_1591'
    # sujet = 'pat_03138_1601'
    # sujet = 'pat_03146_1608'
    # sujet = 'pat_03174_1634'

    #monopol = True
    for monopol in [True, False]:

        #sujet = sujet_list_FR_CV[4]
        for sujet in sujet_list_FR_CV:

            print(f'######## {sujet}, monopol : {monopol} ########')
            
            #### verify computing

            os.chdir(os.path.join(path_prep, sujet, 'sections'))
    
            if monopol:
                if os.path.exists(os.path.join(path_prep, sujet, 'sections', f'{sujet}_allcond_wb.fif')):
                    print('ALREADY COMPUTED')
                    continue
            else:
                if os.path.exists(os.path.join(path_prep, sujet, 'sections', f'{sujet}_allcond_wb_bi.fif')):
                    print('ALREADY COMPUTED')
                    continue


            ################################
            ######## EXTRACT DATA ########
            ################################

            if sujet.find('pat') == -1:
                if monopol:
                    data, chan_list, data_aux, chan_list_aux, chan_list_all_rmw, trig, srate = extract_data_trc(sujet)
                else:
                    data, chan_list, data_aux, chan_list_aux, trig, srate = extract_data_trc_bi(sujet)

            else:
                raw = import_raw_data(sujet)
                if monopol:
                    data, chan_list, data_aux, chan_list_aux, srate = organize_raw(sujet, raw)
                else:
                    data, chan_list, data_aux, chan_list_aux, srate = organize_raw_bi(sujet, raw)

                del raw



            ################################
            ######## ADJUST TRIG ######## 
            ################################

            #### verif and adjust trig for some patients
            if debug == True:

                trig_clean = trig.query("name in ['CV_start', 'CV_stop', '11', '12', '31', '32', '51', '52', '61', '62', '71', '72']")

                chan_name = 'nasal'
                chan_i = chan_list_aux.index(chan_name)
                plt.plot(data_aux[chan_i,:])
                plt.vlines(trig_clean.time.values, ymin=data_aux[chan_i,:].min(), ymax=data_aux[chan_i,:].max(), colors='k')
                plt.show()

                chan_name = 'nasal'
                data_plot = data_aux
                chan_list_plot = chan_list_aux
                start = 0 *60*srate # give min
                stop =  50 *60*srate  # give min
                stop =  data_plot.shape[-1]

                chan_i = chan_list_plot.index(chan_name)
                times = np.arange(np.size(data_plot,1))
                trig_keep = (trig.time.values >= start) & (trig.time.values <= stop)
                x = data_plot[chan_i,start:stop]
                time = times[start:stop]

                plt.plot(time, x)
                plt.vlines(trig.time.values[trig_keep], ymin=np.min(x), ymax=np.max(x), colors='k')
                plt.vlines(trigger_allsujet[sujet]['trig_time'], ymin=np.min(x), ymax=np.max(x), colors='k')
                plt.show()

            #### adjust                
            trig_load = {'name' : trigger_allsujet[sujet]['trig_name'], 'time' : trigger_allsujet[sujet]['trig_time']}
            trig = pd.DataFrame(trig_load)
   
            # verif trig
            if debug == True:
                x = data_aux[0, 1551821:1643948]

                nwind = int(10*srate)
                nfft = nwind
                noverlap = np.round(nwind/2)
                hannw = scipy.signal.windows.hann(nwind)

                hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

                plt.plot(x)
                plt.show()

                plt.plot(hzPxx, Pxx)
                plt.xlim(0,2)
                plt.show()




            ################################
            ######## AUX PROCESSING ########
            ################################

            raw_aux, ecg_events_time = ecg_detection(data_aux, chan_list_aux, srate)

            raw_aux = respi_preproc(raw_aux)

            if debug == True:
                #### verif ECG
                chan_list_aux = raw_aux.info['ch_names']
                ecg_i = chan_list_aux.index('ECG')
                ecg = raw_aux.get_data()[ecg_i,:]
                plt.plot(ecg)
                plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
                trig_values = []
                for trig_i in trig.values():
                    [trig_values.append(i) for i in trig_i]
                plt.vlines(trig_values, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)

                plt.legend()
                plt.show()

                #### add events if necessary
                corrected = []
                cR_init = trig['time'].values
                ecg_events_corrected = np.hstack([cR_init, np.array(corrected)])

                #### find an event to remove
                around_to_find = 1000
                value_to_find = 3265670    
                ecg_cR_array = np.array(ecg_events_time) 
                ecg_cR_array[ ( np.array(ecg_events_time) >= (value_to_find - around_to_find) ) & ( np.array(ecg_events_time) <= (value_to_find + around_to_find) ) ] 

                #### verify add events
                plt.plot(ecg)
                plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
                plt.vlines(ecg_events_corrected, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)
                plt.legend()
                plt.show()






            ################################
            ######## ADJUST ECG ######## 
            ################################


            #### add
            # ecg_events_corrected = ecg_adjust_allsujet[sujet]['ecg_events_corrected']
            # ecg_events_time = np.append(ecg_events_time, np.array(ecg_events_corrected))
            # ecg_events_time.sort()
            # #### remove
            # ecg_events_to_remove = ecg_adjust_allsujet[sujet]['ecg_events_to_remove']
            # [ecg_events_time.remove(i) for i in ecg_events_to_remove]    






            ################################################
            ######## PREPROCESSING, CHOP AND SAVE ########
            ################################################

            data_preproc_wb  = preprocessing_ieeg(data, chan_list, srate, prep_step_wb)
            chop_save_trc(sujet, data_preproc_wb, chan_list, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, monopol, band_preproc='wb', export_info=True)

            del data_preproc_wb

            # data_preproc_lf  = preprocessing_ieeg(data, chan_list, srate, prep_step_lf)
            # chop_save_trc(sujet, data_preproc_lf, chan_list, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, monopol, band_preproc='lf', export_info=True)

            # del data_preproc_lf

            # data_preproc_hf = preprocessing_ieeg(data, chan_list, srate, prep_step_hf)
            # chop_save_trc(sujet, data_preproc_hf, chan_list, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, monopol, band_preproc='hf', export_info=False)

            #### verif
            if debug == True:
                compare_pre_post(data, data_preproc_wb, 0)













