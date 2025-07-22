



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd

from n00_config_params import *
from n00bis_config_analysis_functions import *



########################
######## VIEWER ########
########################


def viewer(sujet, cond, session_i, chan_selection, color_sig, monopol=monopol, filter=False):

    #### params
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, chan_list_keep = modify_name(chan_list_ieeg)
    chan_list_mod_list_i = [chan_list_mod.index(chan) for chan in chan_selection]
    chan_list_mod_list_i = [chan_list_mod[chan_i] for chan_i in chan_list_mod_list_i]
    chan_list_i = [chan_i for chan_i, chan in enumerate(chan_list_mod) if chan in chan_selection]
    chan_list_i.append(-4)

    #### load data
    data = load_data_sujet(sujet, 'wb', cond, session_i, monopol)[chan_list_i,:]
    trig = pd.read_excel(os.path.join(path_prep, sujet, 'info', f"{sujet}_trig.xlsx"))
    df_loca = get_loca_df(sujet, monopol)
    loca_list = [df_loca.query(f"name == '{chan_i}'")['ROI'].values for chan_i in chan_list_mod_list_i]

    chan_labels = [f"{chan} : {loca_list[chan_i]}" for chan_i, chan in enumerate(chan_selection)]
    chan_labels.extend(['respi'])

    if debug:

        plt.plot(data[-1,:])
        plt.show()

    #### downsample
    srate_downsample = 10

    time_vec = np.linspace(0,data.shape[-1],data.shape[-1])/srate
    time_vec_resample = np.linspace(0,data.shape[-1],int(data.shape[-1] * (srate_downsample / srate)))/srate

    data_resampled = np.zeros((data.shape[0], time_vec_resample.shape[0]))

    for chan_i in range(data.shape[0]):
        f = scipy.interpolate.interp1d(time_vec, data[chan_i,:], kind='quadratic', fill_value="extrapolate")
        data_resampled[chan_i,:] = f(time_vec_resample)

    trig = pd.DataFrame({'start' : trig[::2]['time'].values/srate, 'stop' : trig[1::2]['time'].values/srate})

    if debug:

        plt.plot(time_vec, data[chan_i,:], label='raw')
        plt.plot(time_vec_resample, data_resampled[chan_i,:], label='resampled')
        plt.legend()
        plt.show()

    #### for one chan
    if len(chan_selection) == 1:

        respi = data_resampled[-1,:]

        if filter:

            fcutoff = 40
            transw  = .2
            order   = np.round( 7*srate/fcutoff )
            shape   = [ 0,0,1,1 ]
            frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
            filtkern = scipy.signal.firls(order+1,frex,shape,fs=srate)
            x = scipy.signal.filtfilt(filtkern,1,x)


            fcutoff = 100
            transw  = .2
            order   = np.round( 7*srate/fcutoff )
            shape   = [ 1,1,0,0 ]
            frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
            filtkern = scipy.signal.firls(order,frex,shape,fs=srate)
            x = scipy.signal.filtfilt(filtkern,1,x)

        chan_i = 0

        fig, ax = plt.subplots()
    
        x = data_resampled[chan_i,:]
        ax.plot(time_vec_resample, zscore(x), label=chan_labels[chan_i], color=color_sig, alpha=0.7)
        ax.plot(time_vec_resample, zscore(respi), label=chan_labels[-1])

        if cond == 'allcond':

            ax.vlines(trig['start'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+2)).max(), colors='g', label='start')
            ax.vlines(trig['stop'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+2)).max(), colors='r', label='stop')
        
        ax.set_title(f"{sujet} {cond} {session_i+1}")
        plt.legend()

        plt.show()

    #### for several chan
    else:

        respi = data_resampled[-1,:]

        if filter:

            fcutoff = 40
            transw  = .2
            order   = np.round( 7*srate/fcutoff )
            shape   = [ 0,0,1,1 ]
            frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
            filtkern = scipy.signal.firls(order+1,frex,shape,fs=srate)
            x = scipy.signal.filtfilt(filtkern,1,x)


            fcutoff = 100
            transw  = .2
            order   = np.round( 7*srate/fcutoff )
            shape   = [ 1,1,0,0 ]
            frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
            filtkern = scipy.signal.firls(order,frex,shape,fs=srate)
            x = scipy.signal.filtfilt(filtkern,1,x)

        fig, ax = plt.subplots()

        ax.plot(time_vec_resample, zscore(respi), label=chan_labels[0])

        for chan_i, _ in enumerate(chan_list_i[:-1]):
        
            x = data_resampled[chan_i,:]
            ax.plot(time_vec_resample, zscore(x)+3*(chan_i+1), label=chan_labels[chan_i+1])

        if cond == 'allcond':

            ax.vlines(trig['start'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+1)).max(), colors='g', label='start')
            ax.vlines(trig['stop'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+1)).max(), colors='r', label='stop')
        
        ax.set_title(f"{sujet} {cond} {session_i+1}")
        plt.legend()

        plt.show()




################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    ### Good plot for plotting : MAZm : B01, DUCa : H04 / Y10

    ################################
    ######## FR_CV ########
    ################################

    sujet = 'CARv'

    monopol = True

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, _ = modify_name(chan_list_ieeg)

    chan_selection = ['P03']
    chan_selection = [chan for chan_i, chan in enumerate(chan_list_mod) if chan_i in np.random.randint(low=0, high=len(chan_list_ieeg), size=10)]

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, chan_list_keep = modify_name(chan_list_ieeg)
    chan_list_mod_list_i = [chan_list_mod.index(chan) for chan in chan_selection]
    chan_list_mod_list_i = [chan_list_mod[chan_i] for chan_i in chan_list_mod_list_i]
    chan_list_i = [chan_i for chan_i, chan in enumerate(chan_list_mod) if chan in chan_selection]
    chan_list_i.append(-4)
    chan_sel_i = chan_list_i[0]

    #### load data
    trig = pd.read_excel(os.path.join(path_prep, sujet, 'info', f"{sujet}_trig.xlsx"))
    df_loca = get_loca_df(sujet, monopol)
    loca_list = [df_loca.query(f"name == '{chan_i}'")['ROI'].values for chan_i in chan_list_mod_list_i]

    chan_labels = [f"{chan} : {loca_list[chan_i]}" for chan_i, chan in enumerate(chan_selection)]
    chan_labels.extend(['respi'])

    #### compute spectra
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzPxx = (hzPxx>=0.05) & (hzPxx<0.55)
    hzPxx = hzPxx[mask_hzPxx]

    spectra = {}

    for cond in ['FR_CV']:
    
        data = load_data_sujet(sujet, 'wb', cond, 0, monopol)[chan_list_i,:]
        x = data[0,:]
        y = data[-1,:]

        hzPxx_tmp, Pxx_spectrum = scipy.signal.welch(x, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)
        hzPxx_tmp, Cxy_spectrum = scipy.signal.coherence(x, y, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

        spectra[cond] = {'Pxx' : Pxx_spectrum[mask_hzPxx], 'Cxy' : Cxy_spectrum[mask_hzPxx]}

    #### get respfeatures
    df_respfeatures_compact = pd.DataFrame()

    respfeatures = load_respfeatures(sujet)

    for cond in conditions:
    
        for cond_i in range(session_count[cond]):

            _respfeatures = respfeatures[cond][cond_i].query(f"select == 1")
            _respfeatures['cond'] = [f"{cond}"] * _respfeatures.shape[0]
            _respfeatures['sujet'] = [f"{sujet}"] * _respfeatures.shape[0]
            df_respfeatures_compact = pd.concat([df_respfeatures_compact, _respfeatures]).drop(columns=['Unnamed: 0'])

    resofeatures_metric_list = ['cycle_duration', 'cycle_freq', 'total_amplitude', 'total_volume', 'select', 'cond', 'sujet']

    resp = {}

    for cond in ['FR_CV']:

        resp[cond] = df_respfeatures_compact.query(f"cond == '{cond}'")['cycle_freq'].median()

    #### plot
    vlim = {}

    for spectrum_type_i, spectrum_type in enumerate(['Pxx', 'Cxy']):

        _data = []

        for cond in ['FR_CV']:

            _data.append(spectra[cond][spectrum_type].max())

        vlim[spectrum_type] = np.array(_data).max()

    colors = {'FR_CV' : 'tab:orange'}

    fig, axs = plt.subplots(ncols=2, figsize=(10,4))

    for spectrum_type_i, spectrum_type in enumerate(['Pxx', 'Cxy']):

        ax = axs[spectrum_type_i]

        if spectrum_type == 'Pxx':

            ax.semilogy()

        for cond in ['FR_CV']:

            ax.plot(hzPxx, spectra[cond][spectrum_type], label=cond, color=colors[cond])
            ax.vlines(resp[cond], ymin=0, ymax=vlim[spectrum_type], color=colors[cond], linestyle='--')

        ax.set_title(spectrum_type)

    plt.legend()
    # plt.show()

    os.chdir(os.path.join(path_results, 'allplot', 'cosmetics'))
    fig.savefig(f"{sujet}_example_FR_CV_Pxx_Cxy.png")


    ################################
    ######## ALLCOND ########
    ################################

    
    sujet_list = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'POTm', 'VERj', 'DUCa', 'CARv', 'BOUt', 'FLAb'
                    'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949',
                    'pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601',
                    'pat_03146_1608', 'pat_03174_1634'
                    ]

    sujet = 'MAZm'

    session_i = 0
    session_i = 1
    session_i = 2

    monopol = True
    monopol = False

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, _ = modify_name(chan_list_ieeg)

    chan_selection = ['B01']

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, chan_list_keep = modify_name(chan_list_ieeg)
    chan_list_mod_list_i = [chan_list_mod.index(chan) for chan in chan_selection]
    chan_list_mod_list_i = [chan_list_mod[chan_i] for chan_i in chan_list_mod_list_i]
    chan_list_i = [chan_i for chan_i, chan in enumerate(chan_list_mod) if chan in chan_selection]
    chan_list_i.append(-4)
    chan_sel_i = chan_list_i[0]

    #### load data
    data = load_data_sujet(sujet, 'wb', cond, session_i, monopol)[chan_list_i,:]
    trig = pd.read_excel(os.path.join(path_prep, sujet, 'info', f"{sujet}_trig.xlsx"))
    df_loca = get_loca_df(sujet, monopol)
    loca_list = [df_loca.query(f"name == '{chan_i}'")['ROI'].values for chan_i in chan_list_mod_list_i]

    chan_labels = [f"{chan} : {loca_list[chan_i]}" for chan_i, chan in enumerate(chan_selection)]
    chan_labels.extend(['respi'])

    #### compute spectra
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzPxx = (hzPxx>=0.05) & (hzPxx<0.55)
    hzPxx = hzPxx[mask_hzPxx]

    spectra = {}

    for cond in ['RD_CV' ,'RD_SV', 'RD_FV']:
    
        data = load_data_sujet(sujet, 'wb', cond, session_i, monopol)[chan_list_i,:]
        x = data[0,:]
        y = data[-1,:]

        hzPxx_tmp, Pxx_spectrum = scipy.signal.welch(x, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)
        hzPxx_tmp, Cxy_spectrum = scipy.signal.coherence(x, y, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

        spectra[cond] = {'Pxx' : Pxx_spectrum[mask_hzPxx], 'Cxy' : Cxy_spectrum[mask_hzPxx]}

    #### get respfeatures
    df_respfeatures_compact = pd.DataFrame()

    respfeatures = load_respfeatures(sujet)

    for cond in conditions:
    
        for cond_i in range(session_count[cond]):

            _respfeatures = respfeatures[cond][cond_i].query(f"select == 1")
            _respfeatures['cond'] = [f"{cond}"] * _respfeatures.shape[0]
            _respfeatures['sujet'] = [f"{sujet}"] * _respfeatures.shape[0]
            df_respfeatures_compact = pd.concat([df_respfeatures_compact, _respfeatures]).drop(columns=['Unnamed: 0'])

    respfeatures_metric_list = ['cycle_duration', 'cycle_freq', 'total_amplitude', 'total_volume', 'select', 'cond', 'sujet']

    resp = {}

    for cond in ['RD_CV' ,'RD_SV', 'RD_FV']:

        resp[cond] = df_respfeatures_compact.query(f"cond == '{cond}'")['cycle_freq'].median()

    #### plot
    vlim = {}

    for spectrum_type_i, spectrum_type in enumerate(['Pxx', 'Cxy']):

        _data = []

        for cond in ['RD_CV' ,'RD_SV', 'RD_FV']:

            _data.append(spectra[cond][spectrum_type].max())

        vlim[spectrum_type] = np.array(_data).max()

    colors = {'RD_CV' : 'tab:blue', 'RD_SV' : 'tab:red', 'RD_FV' : 'tab:green'}

    fig, axs = plt.subplots(ncols=2, figsize=(10,4))

    for spectrum_type_i, spectrum_type in enumerate(['Pxx', 'Cxy']):

        ax = axs[spectrum_type_i]

        if spectrum_type == 'Pxx':

            ax.semilogy()

        for cond in ['RD_CV' ,'RD_SV', 'RD_FV']:

            ax.plot(hzPxx, spectra[cond][spectrum_type], label=cond, color=colors[cond])
            ax.vlines(resp[cond], ymin=0, ymax=vlim[spectrum_type], color=colors[cond], linestyle='--')

        ax.set_title(spectrum_type)

    plt.legend()
    # plt.show()

    os.chdir(os.path.join(path_results, 'allplot', 'cosmetics'))
    fig.savefig(f"{sujet}_example_ALLCOND_Pxx_Cxy.png")

    #### for each conditions

    for cond in ['RD_CV' ,'RD_SV', 'RD_FV']:

        fig, axs = plt.subplots(ncols=2, figsize=(10,4))

        for spectrum_type_i, spectrum_type in enumerate(['Pxx', 'Cxy']):

            ax = axs[spectrum_type_i]

            # if spectrum_type == 'Pxx':

            #     ax.semilogy()

            if spectrum_type == 'Cxy':

                ax.set_ylim(0,1)

                ax.plot(hzPxx, spectra[cond][spectrum_type], label=cond, color=colors[cond])
                ax.vlines(resp[cond], ymin=0, ymax=1, color=colors[cond], linestyle='--')

            else:

                ax.plot(hzPxx, spectra[cond][spectrum_type], label=cond, color=colors[cond])
                ax.vlines(resp[cond], ymin=0, ymax=spectra[cond][spectrum_type].max(), color=colors[cond], linestyle='--')

            ax.set_title(spectrum_type)

        # plt.show()
        plt.suptitle(cond)

        os.chdir(os.path.join(path_results, 'allplot', 'cosmetics'))
        fig.savefig(f"{sujet}_example_{cond}_Pxx_Cxy.png")


    ################################
    ######## MI construction ########
    ################################

    
    sujet_list = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'POTm', 'VERj', 'DUCa', 'CARv', 'BOUt', 'FLAb'
                    'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949',
                    'pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601',
                    'pat_03146_1608', 'pat_03174_1634'
                    ]

    sujet = 'CARv'

    session_i = 0
    session_i = 1
    session_i = 2

    cond = 'FR_CV'

    monopol = True
    monopol = False

    chan_selection = ['B01']

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, chan_list_keep = modify_name(chan_list_ieeg)
    chan_list_mod_list_i = [chan_list_mod.index(chan) for chan in chan_selection]
    chan_list_mod_list_i = [chan_list_mod[chan_i] for chan_i in chan_list_mod_list_i]
    chan_list_i = [chan_i for chan_i, chan in enumerate(chan_list_mod) if chan in chan_selection]
    chan_list_i.append(-4)
    chan_sel_i = chan_list_i[0]

    #### load data
    data = load_data_sujet(sujet, 'wb', cond, session_i, monopol)[chan_list_i,:]
    trig = pd.read_excel(os.path.join(path_prep, sujet, 'info', f"{sujet}_trig.xlsx"))
    df_loca = get_loca_df(sujet, monopol)
    loca_list = [df_loca.query(f"name == '{chan_i}'")['ROI'].values for chan_i in chan_list_mod_list_i]

    chan_labels = [f"{chan} : {loca_list[chan_i]}" for chan_i, chan in enumerate(chan_selection)]
    chan_labels.extend(['respi'])

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    if sujet[:3] != 'pat':
        if monopol:
            chan_list_ieeg, chan_list_keep = modify_name(chan_list_ieeg)

    if monopol:
        tf = np.load(f'{sujet}_tf_conv_{cond}.npy')[chan_list_ieeg.index(chan_selection[0]), :, :, :]
    else:
        tf = np.load(f'{sujet}_tf_conv_{cond}_bi.npy')[chan_list_ieeg.index(chan_selection[0]), :, :, :]

    band = 'theta'
    freq = [4, 8]

    mask_band = (frex > freq[0]) & (frex < freq[1])

    tf_band = tf[:,mask_band]

    plt.plot(np.median(np.median(tf_band, axis=0), axis=0))
    plt.show()



    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    if sujet[:3] != 'pat':
        if monopol:
            chan_list_sel, chan_list_keep = modify_name(chan_list_ieeg)
        else:
            chan_list_sel = chan_list_ieeg
    else:
        chan_list_sel = chan_list_ieeg

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    plot_data = {'obs' : {}, 'surr' : {}}

    #chan_i = 66
    for chan_i, _ in enumerate(chan_list_sel):

        #cond = conditions[-1]
        for cond_i, cond in enumerate(conditions):

            print(cond, flush=True)
                
            #### Pxx
            if monopol:
                tf = np.load(f'{sujet}_tf_conv_{cond}.npy')
            else:
                tf = np.load(f'{sujet}_tf_conv_{cond}_bi.npy')

            #### fill df
            #chan_i, chan_name = 0, chan_list_sel[0]
            # for chan_i, chan_name in enumerate(chan_list_sel):

            def get_surr_MI_for_chan(chan_i):

                data_chan_xr = np.zeros((3, len(freq_band_dict_FC_lmm['wb']), stretch_point_TF)) 

                tf_chan = tf[chan_i,:,:]

                #band, freq = 'theta', [4,8]
                for band_i, (band, freq) in enumerate(freq_band_dict_FC_lmm['wb'].items()):

                    frex_sel = (frex >= freq[0]) & (frex <= freq[-1])
                    tf_chan_frex = np.median(tf_chan[:,frex_sel], axis=1)
                    flat_suffle_sig = tf_chan_frex.reshape(-1)
                    n_cycle = tf_chan_frex.shape[0]

                    _surr = np.zeros((n_surrogates_tf, tf_chan_frex.shape[-1]))

                    for surr_i in np.arange(n_surrogates_tf):

                        shuffle_win_start = np.random.choice(np.arange(tf_chan_frex.shape[-1], flat_suffle_sig.size-tf_chan_frex.shape[-1]), n_cycle, replace=False)
                        shuffle_win_stop = shuffle_win_start + tf_chan_frex.shape[-1]

                        _shuffle_win = np.zeros((n_cycle, tf_chan_frex.shape[-1]))
                        for win_i, (_win_start, _win_stop) in enumerate(zip(shuffle_win_start, shuffle_win_stop)):
                            _shuffle_win[win_i] = flat_suffle_sig[_win_start:_win_stop]

                        _surr[surr_i] = np.median(_shuffle_win, axis=0)

                        if debug:

                            plt.plot(np.median(tf_chan_frex, axis=0))
                            plt.plot(np.median(_shuffle_win, axis=0), color='r')
                            plt.legend()
                            plt.show()
                        
                    if debug:

                        count, _, _ = plt.hist(_surr, bins=50)
                        plt.vlines([np.median(tf_chan_frex, axis=0).max() - np.median(tf_chan_frex, axis=0).min()], ymin=0, ymax=count.max(), color='r', label='obs')
                        plt.vlines([np.percentile(_surr, 99)], ymin=0, ymax=count.max(), color='g', label='99th')
                        plt.legend()
                        plt.show()

                        plt.plot(np.median(tf_chan_frex, axis=0))
                        plt.plot(np.percentile(_surr, 99, axis=0), color='r')
                        plt.plot(np.percentile(_surr, 1, axis=0), color='r')
                        plt.legend()
                        plt.show()

                    data_chan_xr[0, band_i, :], data_chan_xr[1, band_i, :] = np.percentile(_surr, percentile_MI[0], axis=0), np.percentile(_surr, percentile_MI[1], axis=0)
                    data_chan_xr[2, band_i, :] = np.median(tf_chan_frex, axis=0)

                return data_chan_xr

            data_chan_xr_res = get_surr_MI_for_chan(chan_i)
            plot_data['obs'][cond] = data_chan_xr_res[2]
            plot_data['surr'][cond] = {'dw' : data_chan_xr_res[0], 'up' : data_chan_xr_res[1]}

        #### ONE COND
        for band_i, band in enumerate(band_list_global):

            for cond in conditions:
        
                fig, ax = plt.subplots()
                ax.plot(plot_data['obs'][cond][band_i])
                ax.plot(plot_data['surr'][cond]['up'][band_i], color='r', linestyle='--')
                ax.plot(plot_data['surr'][cond]['dw'][band_i], color='r', linestyle='--')
                plt.title(f"{cond} {band}")
                plt.show()
        
        
        #### ALLCOND
        for band_i, band in enumerate(band_list_global):
        
            fig, ax = plt.subplots()

            for cond in ['RD_CV', 'RD_SV', 'RD_FV']:

                ax.plot(plot_data['obs'][cond][band_i], label=cond)

            plt.title(f"{sujet} {chan_i} {band}")
            plt.legend()
            plt.show()




    ################################
    ######## PLOT SIG ########
    ################################

    
    sujet_list = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'POTm', 'VERj', 'DUCa', 'CARv', 'BOUt', 'FLAb'
                    'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949',
                    'pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601',
                    'pat_03146_1608', 'pat_03174_1634'
                    ]

    sujet = 'MAZm'

    session_i = 0
    session_i = 1
    session_i = 2

    cond = 'RD_SV'

    monopol = True
    monopol = False

    chan_selection = ['B01']

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, chan_list_keep = modify_name(chan_list_ieeg)
    chan_list_mod_list_i = [chan_list_mod.index(chan) for chan in chan_selection]
    chan_list_mod_list_i = [chan_list_mod[chan_i] for chan_i in chan_list_mod_list_i]
    chan_list_i = [chan_i for chan_i, chan in enumerate(chan_list_mod) if chan in chan_selection]
    chan_list_i.append(-4)
    chan_sel_i = chan_list_i[0]

    viewer(sujet, cond, session_i, chan_selection, 'tab:red', monopol=monopol, filter=False)


    ################################
    ######## ENVELOPPE ########
    ################################

    
    sujet_list = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'POTm', 'VERj', 'DUCa', 'CARv', 'BOUt', 'FLAb'
                    'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949',
                    'pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601',
                    'pat_03146_1608', 'pat_03174_1634'
                    ]

    sujet = 'KOFs'

    session_i = 0
    session_i = 1
    session_i = 2

    cond = 'FR_CV'

    monopol = True
    monopol = False

    df_loca = get_loca_df(sujet, monopol)
    df_loca.query(f"ROI != 'WM'")

    chan_selection = 'Sp02'
    chan_selection_notmod = "S'2"

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    loca = df_loca.query(f"name == '{chan_selection}'")['ROI'].iloc[0]

    band_sel = 'alpha'

    frex_mask = (frex > freq_band_dict_FC['wb'][band_sel][0]) & (frex < freq_band_dict_FC['wb'][band_sel][1])
    tf_band = tf[:,frex_mask,:]

    #### select wavelet parameters
    wavelets = get_wavelets()

    #session_i = 0
    data = load_data_sujet(sujet, 'wb', cond, session_i, monopol)[chan_list_ieeg.index(chan_selection_notmod),:]
    respi = load_data_sujet(sujet, 'wb', cond, session_i, monopol)[chan_list.index('nasal'),:]
    
    tf_conv = np.zeros((nfrex, data.shape[0]), dtype=np.float32)

    for fi in range(nfrex):
        
        tf_conv[fi,:] = abs(scipy.signal.fftconvolve(data, wavelets[fi,:], 'same'))**2

    tf_band = np.median(tf_conv[frex_mask,:], axis=0)
    # tf_band = np.max(tf_conv[frex_mask,:], axis=0)

    plt.plot(zscore(tf_band))
    plt.plot(zscore(respi))
    plt.title(f"{sujet} {cond} {chan_selection} {loca}")
    plt.show()



    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

    ROI_sel = 'supramarginal'

    if monopol:
        tf_ROI = xr.open_dataarray(f'allsujet_{cond}_ROI.nc').loc[ROI_sel,:,:].values
    else:
        tf_ROI = xr.open_dataarray(f'allsujet_{cond}_ROI_bi.nc').loc[ROI_sel,:,:].values

    tf_ROI_band = np.median(tf_ROI[frex_mask,:], axis=0)



    data = load_data_sujet(sujet, 'wb', 'allcond', session_i, monopol)[chan_list.index('ventral'),:]
    plt.plot(data)
    plt.show()