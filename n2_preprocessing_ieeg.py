
import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

import mne
import scipy.fftpack
import scipy.signal

from n0_config import *
from n1_generate_electrode_selection import modify_name


debug = False


########################################
######## DATA EXTRACTION ######## 
########################################

#path_data, sujet = 'D:\LPPR_CMO_PROJECT\Lyon\Data\iEEG', 'LYONNEURO_2019_CAPp'
def extract_data_trc():

    os.chdir(os.path.join(path_data,sujet))

    #### identify number of trc file
    trc_file_names = glob.glob('*.TRC')

    #### extract file one by one
    print('#### EXTRACT TRC ####')
    data_whole = []
    chan_list_whole = []
    srate_whole = []
    events_name_whole = []
    events_time_whole = []
    #file_i, file_name = 0, trc_file_names[0]
    for file_i, file_name in enumerate(trc_file_names):

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

    #### events in df
    event_dict = {'name' : events_name, 'time' : events_time}
    columns = ['name', 'time']
    trig = pd.DataFrame(event_dict, columns=columns)

    #### select chan
    print('#### REMOVE CHAN ####')
    os.chdir(os.path.join(path_anatomy, sujet))

    #### first removing
    chan_list_first_clean_file = open(sujet + "_keep_plot.txt", "r")
    chan_list_first_clean = chan_list_first_clean_file.read()
    chan_list_first_clean = chan_list_first_clean.split("\n")[:-1]
    chan_list_first_clean_file.close()

        #### remove chan
    data_rmv_first = data.copy() 
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
    data_rmv_second = data_rmv_first.copy() 
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

    #### verif chan number
    chan2suppr = len(np.where(electrode_select['select'] == 0)[0]) - len(chan_list_add_in_csv)
    chan_final = len(chan_list_rmv_first) - chan2suppr

    if len(chan_list_rmv_second) != chan_final and np.size(data_rmv_second,0) != chan_final:
        print('chan remove incorrect')
        exit()

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

    return data, chan_list, data_aux, chan_list_aux, chan_list_all_rmw, trig, srate














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

        raw_post = raw.copy()

        raw_post.notch_filter(50)

        
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

        raw_post = raw.copy()

        #filter_length = int(srate*10) # give sec
        filter_length = 'auto'

        if debug == True :
            h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
            flim = (0.1, srate / 2.)
            mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

        raw_eeg_mc_hp = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')

        if debug == True :
            duration = 60.
            n_chan = 20
            raw_eeg_mc_hp.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

        return raw_post


    



    def low_pass(raw, h_freq, l_freq):

        raw_post = raw.copy()

        filter_length = int(srate*10) # in samples

        if debug == True :
            h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
            flim = (0.1, srate / 2.)
            mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

        raw_post = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hann', fir_design='firwin2')

        if debug == True :
            duration = .5
            n_chan = 10
            raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


        return raw_post





    def average_reref(raw):

        raw_post = raw.copy()

        raw_post.set_eeg_reference('average')

        if debug == True :
            duration = .5
            n_chan = 10
            raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


        return raw_post



    # 3. Execute preprocessing 

    raw = raw_init.copy() # first data

    if prep_step.get('mean_centered_detrend').get('execute') :
        print('mean_centered_detrend')
        raw_post = mean_centered_detrend(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()

    

    if prep_step.get('line_noise_removing').get('execute') :
        print('line_noise_removing')
        raw_post = line_noise_removing(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()


    if prep_step.get('high_pass').get('execute') :
        print('high_pass')
        h_freq = prep_step.get('high_pass').get('params').get('h_freq')
        l_freq = prep_step.get('high_pass').get('params').get('l_freq')
        raw_post = high_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()


    if prep_step.get('low_pass').get('execute') :
        print('low_pass')
        h_freq = prep_step.get('low_pass').get('params').get('h_freq')
        l_freq = prep_step.get('low_pass').get('params').get('l_freq')
        raw_post = low_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()


    if prep_step.get('average_reref').get('execute') :
        print('average_reref')
        raw_post = average_reref(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()


    data_preproc = raw.get_data()

    return data_preproc







################################
######## ECG DETECTION ########
################################

def ecg_detection(data_aux, chan_list_aux, srate):

    #### adjust ECG
    if sujet_ecg_adjust.get(sujet) == 'inverse':
        data_aux[-1,:] = data_aux[-1,:] * -1
    
    #### notch ECG
    ch_types = ['misc'] * (np.size(data_aux,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

    info_aux = mne.create_info(chan_list_aux, srate, ch_types=ch_types)
    raw_aux = mne.io.RawArray(data_aux, info_aux)

    raw_aux.notch_filter(50, picks='misc')

    # ECG
    event_id = 999
    ch_name = 'ECG'
    qrs_threshold = .5 #between o and 1
    ecg_events = mne.preprocessing.find_ecg_events(raw_aux, event_id=event_id, ch_name=ch_name, qrs_threshold=qrs_threshold)
    ecg_events_time = list(ecg_events[0][:,0])

    data_aux_final = raw_aux.get_data()

    return data_aux_final, chan_list_aux, ecg_events_time




################################
######## CHOP & SAVE ########
################################

def chop_save_trc(data, chan_list, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, band_preproc, export_info):

    #### save alldata + stim chan
    data_all = np.vstack(( data, data_aux, np.zeros(( len(data[0,:]) )) ))
    chan_list_all = chan_list + chan_list_aux + ['ECG_cR']

    ch_types = ['seeg'] * (len(chan_list_all)-4) + ['misc'] * 4
    info = mne.create_info(chan_list_all, srate, ch_types=ch_types)
    raw_all = mne.io.RawArray(data_all, info)

    #### save chan_list
    os.chdir(os.path.join(path_anatomy, sujet))
    keep_plot_textfile = open(sujet + "_chanlist_ieeg.txt", "w")
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

    # condition, trig_cond = list(conditions_trig.items())[0]
    for condition, trig_cond in conditions_trig.items():

        cond_i = np.where(trig.name.values == trig_cond[0])[0]

        for i, trig_i in enumerate(cond_i):

            count_session[condition] = count_session[condition] + 1 

            raw_chunk = raw_all.copy()
            raw_chunk.crop( tmin = (trig.iloc[trig_i,:].time)/srate , tmax= (trig.iloc[trig_i+1,:].time/srate)-0.2 )
            
            raw_chunk.save(sujet + '_' + condition + '_' + str(i+1) + '_' + band_preproc + '.fif')


    df = {'condition' : list(count_session.keys()), 'count' : list(count_session.values())}
    count_session = pd.DataFrame(df, columns=['condition', 'count'])

    if export_info == True :
    
        #### export trig, count_session, cR
        os.chdir(os.path.join(path_prep, sujet, 'info'))
        
        trig.to_excel(sujet + '_trig.xlsx')

        count_session.to_excel(sujet + '_count_session.xlsx')

        cR = pd.DataFrame(ecg_events_time, columns=['cR_time'])
        cR.to_excel(sujet +'_cR_time.xlsx')


    return 




























################################
######## EXECUTE ########
################################


if __name__== '__main__':



    ################################
    ######## EXTRACT DATA ########
    ################################

    data, chan_list, data_aux, chan_list_aux, chan_list_all_rmw, trig, srate = extract_data_trc()


    #### verif and adjust trig for some patients
    if debug == True:
        chan_name = 'nasal'
        data_plot = data_aux
        chan_list_plot = chan_list_aux
        start = 0 *60*srate # give min
        stop =  57 *60*srate  # give min

        chan_i = chan_list_plot.index(chan_name)
        times = np.arange(np.size(data_plot,1))
        trig_keep = (trig.time.values >= start) & (trig.time.values <= stop)
        x = data_plot[chan_i,start:stop]
        time = times[start:stop]

        plt.plot(time, x)
        plt.vlines(trig.time.values[trig_keep], ymin=np.min(x), ymax=np.max(x), colors='k')
        plt.show()

    #### adjust
    if sujet == 'CHEe':
        
        trig_name = ['CV_start', 'CV_stop', '31',   '32',   '11',   '12',   '71',   '72',   '11',   '12',   '51',    '52',    '11',    '12',    '51',    '52',    '31',    '32']
        trig_time = [0,          153600,    463472, 555599, 614615, 706758, 745894, 838034, 879833, 971959, 1009299, 1101429, 1141452, 1233580, 1285621, 1377760, 1551821, 1643948]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)    

    if sujet == 'GOBc':

        trig_name = ['MV_start', 'MV_stop']
        trig_time = [2947600,    3219072]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        index_append = [len(trig.name), len(trig.name)+1]
        trig_append = pd.DataFrame(trig_load, index=index_append)
        trig = trig.append(trig_append)
    
    if sujet == 'MAZm':
        
        trig_name = ['CV_start','CV_stop',  '31',   '32',   '11',   '12',   '31',   '32',   '11',   '12',   '51',   '52',   '11',   '12',   '51',   '52',   '61',    '62',    '61',    '62',    'MV_start', 'MV_stop']
        trig_time = [0,         164608,     164609, 240951, 275808, 367946, 396690, 488808, 529429, 621558, 646959, 739078, 763014, 855141, 877518, 969651, 1102256, 1194377, 1218039, 1310170, 1391000,    1558000]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)

    if sujet == 'MUGa':
        
        trig_name = ['CV_start', 'CV_stop']
        trig_time = [17400,       98700]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)    

    if sujet == 'TREt':

        trig_name = ['61',    '62',    'MV_start', 'MV_stop']
        trig_time = [1492770, 1584894, 1679260,    1751039]
        
        trig = trig.drop(labels=range(51,59), axis=0, inplace=False)

        trig_load = {'name' : trig_name, 'time' : trig_time}
        index_append = [i for i in range(len(trig.name), (len(trig.name)+len(trig_name)))]
        trig_append = pd.DataFrame(trig_load, index=index_append)
        trig = trig.append(trig_append)


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

    data_aux, chan_list_aux, ecg_events_time = ecg_detection(data_aux, chan_list_aux, srate)

    if debug == True:
        #### verif ECG
        ecg_i = chan_list_aux.index('ECG')
        ecg = data_aux[ecg_i,:]
        plt.plot(ecg)
        plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
        plt.vlines(trig['time'].values, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)
        plt.legend()
        plt.show()

        #### add events if necessary
        corrected = []
        cR_init = trig['time'].values
        ecg_events_corrected = cR_init + corrected

        #### find an event to remove
        around_to_find = 1000
        value_to_find = 1297748    
        ecg_cR_array = np.array(ecg_events_time) 
        ecg_cR_array[ ( np.array(ecg_events_time) >= (value_to_find - around_to_find) ) & ( np.array(ecg_events_time) <= (value_to_find + around_to_find) ) ] 

        #### verify add events
        plt.plot(ecg)
        plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
        plt.vlines(ecg_events_corrected, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)
        plt.legend()
        plt.show()

    #### adjust trig for some patients
    if sujet == 'CHEe':
        #### add
        ecg_events_corrected = [339111, 347393, 358767, 360242, 363559, 460709, 554965, 870178, 871428, 873406, 1142520, 1298203, 1297285, 1297760]
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()
        #### remove
        ecg_events_to_remove = [555198, 1298638, 1297486, 1297749]
        [ecg_events_time.remove(i) for i in ecg_events_to_remove]    

    if sujet == 'GOBc':
        #### add
        ecg_events_corrected = [914419, 1206975, 1.70231e6, 1721770, 1730034, 1730871, 1731349, 1732781, 1.78158e6, 1.78227e6, 1.78493e6, 2199475, 2321365, 2322851, 2473316, 2797246, 2800339, 2.88987e6, 2.89661e6, 2968938]
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()
        #### remove
        ecg_events_to_remove = []
        [ecg_events_time.remove(i) for i in ecg_events_to_remove]  

    if sujet == 'MAZm':
        #### add
        ecg_events_corrected = [7.9927e5, 8.6539e5, 9.9984e5, 1.01111e6, 1.16265e6, 1.28444e6, 1.35341e6]
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()   
        #### remove
        ecg_events_to_remove = [1353229]
        [ecg_events_time.remove(i) for i in ecg_events_to_remove]  

    if sujet == 'MUGa':
        #### add
        ecg_events_corrected = []
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()       
        #### remove
        ecg_events_to_remove = []
        [ecg_events_time.remove(i) for i in ecg_events_to_remove]   

    if sujet == 'TREt':
        #### add
        ecg_events_corrected = [21506, 142918, 289897, 1308016, 1.36292e6, 1.36483e6, 1.36523e6, 1.36563e6, 1.36647e6, 1.36690e6, 1.36730e6, 1.36968e6, 1.37006e6, 1.60849e6, 1626322, 1629380, 1630109, 1636351, 1641558, 1642374, 1645133]
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()
        #### remove
        ecg_events_to_remove = []
        [ecg_events_time.remove(i) for i in ecg_events_to_remove]  





    ################################
    ######## PREPROCESSING ########
    ################################

    data_preproc_lf  = preprocessing_ieeg(data, chan_list, srate, prep_step_lf)
    data_preproc_hf = preprocessing_ieeg(data, chan_list, srate, prep_step_hf)
 
    #### verif
    if debug == True:
        compare_pre_post(data, data_preproc_lf, 0)



    ################################
    ######## CHOP AND SAVE ########
    ################################

    chop_save_trc(data_preproc_lf, chan_list, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, band_preproc='lf', export_info=True)
    chop_save_trc(data_preproc_hf, chan_list, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, band_preproc='hf', export_info=False)











