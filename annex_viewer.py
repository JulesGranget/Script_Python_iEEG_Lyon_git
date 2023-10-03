



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd

from n0_config_params import *
from n0bis_config_analysis_functions import *



########################
######## VIEWER ########
########################


def viewer(sujet, cond, session_i, chan_selection, monopol, filter=False):

    #### params
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    chan_list_mod, chan_list_keep = modify_name(chan_list_ieeg)
    chan_list_mod_list_i = [chan_list_keep.index(chan) for chan in chan_selection]
    chan_list_mod_list_i = [chan_list_mod[chan_i] for chan_i in chan_list_mod_list_i]
    chan_list_i = [chan_i for chan_i, chan in enumerate(chan_list) if chan in chan_selection]
    chan_list_i.append(-4)
    chan_list_i.append(-2)

    #### load data
    data = load_data_sujet(sujet, 'wb', cond, session_i, monopol)[chan_list_i,:]
    trig = pd.read_excel(os.path.join(path_prep, sujet, 'info', f"{sujet}_trig.xlsx"))
    df_loca = get_loca_df(sujet, monopol)
    loca_list = [df_loca.query(f"name == '{chan_i}'")['ROI'].values for chan_i in chan_list_mod_list_i]

    chan_labels = ['respi', 'ecg']
    chan_labels.extend([f"{chan} : {loca_list[chan_i]}" for chan_i, chan in enumerate(chan_selection)])

    if debug:

        plt.plot(data[0,:])
        plt.show()

    #### downsample
    srate_downsample = 50

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

        respi = data_resampled[-2,:]
        ecg = data_resampled[-1,:]

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

        ax.plot(time_vec_resample, zscore(respi), label=chan_labels[0])
        ax.plot(time_vec_resample, zscore(ecg)+3, label=chan_labels[1])
    
        x = data_resampled[chan_i,:]
        ax.plot(time_vec_resample, zscore(x)+3*(chan_i+2), label=chan_labels[chan_i+2])

        ax.vlines(trig['start'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+2)).max(), colors='g', label='start')
        ax.vlines(trig['stop'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+2)).max(), colors='r', label='stop')
        
        ax.set_title(f"{sujet} {cond} {session_i+1}")
        plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper left')  # reverse to keep order consistent

        plt.show()

    #### for several chan
    else:

        respi = data_resampled[-2,:]
        ecg = data_resampled[-1,:]

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
        ax.plot(time_vec_resample, zscore(ecg)+3, label=chan_labels[1])

        for chan_i, _ in enumerate(chan_list_i[:-2]):
        
            x = data_resampled[chan_i,:]
            ax.plot(time_vec_resample, zscore(x)+3*(chan_i+2), label=chan_labels[chan_i+2])

        ax.vlines(trig['start'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+2)).max(), colors='g', label='start')
        ax.vlines(trig['stop'].values, ymin=zscore(respi).min(), ymax=(zscore(x)+3*(chan_i+2)).max(), colors='r', label='stop')
        
        ax.set_title(f"{sujet} {cond} {session_i+1}")
        plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper left')  # reverse to keep order consistent

        plt.show()







################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    
    sujet_list = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'POTm', 'BANc', 'KOFs', 'LEMl', 'MUGa',
                    'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949',
                    'pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601',
                    'pat_03146_1608', 'pat_03174_1634'
                    ]

    sujet = 'CHEe'

    cond = 'allcond'
    
    cond = 'FR_CV'
    cond = 'RD_CV'
    cond = 'RD_SV'
    cond = 'RD_FV'

    session_i = 0
    session_i = 1
    session_i = 2

    monopol = True
    monopol = False

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    chan_selection = ["J' 1", "J' 2", "J' 3", "J' 4", "J' 5", "A' 1", "A' 2", "A' 3"]

    viewer(sujet, cond, session_i, chan_selection, monopol)

    nchan_to_plot = 10
    chan_selection_list_i = np.arange(len(chan_list_ieeg))[::nchan_to_plot]

    for chan_selection_i in chan_selection_list_i:

        if chan_selection_i + nchan_to_plot > len(chan_list_ieeg):

            chan_selection = chan_list[chan_selection_i:len(chan_list_ieeg)]

        else:

            chan_selection = chan_list[chan_selection_i:(chan_selection_i+nchan_to_plot)]

        viewer(sujet, cond, session_i, chan_selection, monopol)


