
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd

from n00_config_params import *
from n00bis_config_analysis_functions import *

import joblib

debug = False






################################################
######## CXY CYCLE FREQ SURROGATES ########
################################################



#sujet, band_prep, cond, monopol = sujet_list[0], 'wb', 'RD_SV', True
def precompute_surrogates_coh(sujet, band_prep, cond, monopol):
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    
    print(cond, flush=True)

    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    if monopol:
        if os.path.exists(f'{sujet}_{cond}_Cxy.npy') == True :
            print('ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(f'{sujet}_{cond}_Cxy_bi.npy') == True :
            print('ALREADY COMPUTED', flush=True)
            return

    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    Cxy_vec = np.zeros((len(chan_list_ieeg), hzCxy.size))

    #### generate whole sig vec
    time_chunk = []

    for session_i in range(session_count[cond]):

        if session_i == 0:

            data_allsession = load_data_sujet(sujet, band_prep, cond, session_i, monopol)
            time_chunk.append(data_allsession.shape[-1])

        else:

            _data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)
            data_allsession = np.append(data_allsession, _data, axis=1)
            time_chunk.append(_data.shape[-1])

    time_chunk = np.cumsum(np.array(time_chunk))

    if debug:

        chan_i = 0
        plt.plot(data_allsession[chan_i,:])
        plt.vlines(time_chunk, ymin=data_allsession[chan_i,:].min(), ymax=data_allsession[chan_i,:].max(), color='r')
        plt.show()

    #### correct session suite
    #sig = data_allsession[0,:]
    def correct_sig_cut(sig, time_chunk, fade_duration_sec=1):

        fade_len = int(fade_duration_sec * srate)
        corrected_sig = sig.copy()

        for cut_i in time_chunk:

            # Skip if the window goes out of bounds
            if cut_i - fade_len < 0 or cut_i + fade_len > len(sig):
                continue

            # Extract fade-out and fade-in regions
            fade_out = corrected_sig[cut_i - fade_len : cut_i]
            fade_in  = corrected_sig[cut_i : cut_i + fade_len]

            # pad reflect to match whole sig size
            fade_out = np.concatenate([fade_out, fade_out[::-1]])
            fade_in = np.concatenate([fade_in, fade_in[::-1]])

            if debug:
                
                plt.plot(np.concatenate([fade_out, fade_out[::-1]]))
                plt.show()

            # Create ramps
            ramp = np.linspace(0, 1, fade_len*2)
            fade_out_weighted = fade_out * (1 - ramp)
            fade_in_weighted  = fade_in * ramp

            if debug:

                plt.plot(fade_out_weighted)
                plt.plot(fade_in_weighted)
                plt.plot(fade_out_weighted + fade_in_weighted)
                plt.show()

            # Blended region
            blended = fade_out_weighted + fade_in_weighted

            # Insert the blend into the signal
            corrected_sig[cut_i - fade_len : cut_i + fade_len] = blended

        if debug:

            plt.plot(sig, label='sig')
            plt.plot(corrected_sig, label='corrected_sig')
            plt.vlines(time_chunk, ymin=data_allsession[chan_i,:].min(), ymax=data_allsession[chan_i,:].max(), color='r')
            plt.legend()
            plt.show()

        return corrected_sig

    for chan_i, _ in enumerate(chan_list):

        data_allsession[chan_i,:] = correct_sig_cut(data_allsession[chan_i,:], time_chunk, fade_duration_sec = 1)

    if debug:

        for chunk_i in time_chunk:

            pre_i, post_i = chunk_i - srate*1, chunk_i + srate*1    
            plt.plot(data_allsession[:,pre_i:post_i], label='corrected_sig')
            plt.vlines(chunk_i, ymin=data_allsession[:,pre_i:post_i].min(), ymax=data_allsession[:,pre_i:post_i].max(), color='r')
            plt.legend()
            plt.show()

    y = data_allsession[chan_list.index('nasal'),:]

    if debug:

        plt.plot(y)
        plt.vlines(time_chunk, ymin=y.min(), ymax=y.max(), color='r')
        plt.show()

    for chan_i, _ in enumerate(chan_list_ieeg):

        x = data_allsession[chan_i,:]
        x_rscore = rscore(x)
        hzCxy_tmp, Cxy = scipy.signal.coherence(x_rscore, y, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)
        Cxy_vec[chan_i,:] = Cxy[mask_hzCxy]

    surrogates_n_chan = np.zeros((len(chan_list_ieeg),len(hzCxy)))

    def compute_surrogates_coh_n_chan(n_chan):

        print_advancement(n_chan, len(chan_list_ieeg), steps=[25, 50, 75])

        x = data_allsession[n_chan,:]
        x_rscore = rscore(x)

        surrogates_val_tmp = np.zeros((n_surrogates_coh,len(hzCxy)))

        for surr_i in range(n_surrogates_coh):

            x_shift = shuffle_Cxy(x_rscore)
            #y_shift = shuffle_Cxy(y)
            hzCxy_tmp, Cxy = scipy.signal.coherence(x_shift, y, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

            surrogates_val_tmp[surr_i,:] = Cxy[mask_hzCxy]

        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i = int(np.floor(n_surrogates_coh*percentile_coh))
        compute_surrogates_coh_tmp = surrogates_val_tmp_sorted[percentile_i,:]

        return compute_surrogates_coh_tmp
    
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        compute_surrogates_coh_results = joblib.Parallel(n_jobs=n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_coh_n_chan)(n_chan) for n_chan, _ in enumerate(chan_list_ieeg))

    for chan_i, _ in enumerate(chan_list_ieeg):

        surrogates_n_chan[chan_i,:] = compute_surrogates_coh_results[chan_i]

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    if monopol:
        np.save(f'{sujet}_{cond}_Cxy_mask.npy', surrogates_n_chan)
        np.save(f'{sujet}_{cond}_Cxy.npy', Cxy_vec)
    else:
        np.save(f'{sujet}_{cond}_Cxy_mask_bi.npy', surrogates_n_chan)
        np.save(f'{sujet}_{cond}_Cxy_bi.npy', Cxy_vec)

    print('done', flush=True)



def precompute_surrogates_cyclefreq(sujet, band_prep, cond, monopol):

    for session_i in range(session_count[cond]):
    
        print(f'{cond} {session_i+1}', flush=True)

        #### load params
        respfeatures_allcond = load_respfeatures(sujet)
        chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

        #### load data
        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
        data_tmp = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[:len(chan_list_ieeg),:]

        if monopol:
            if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}.npy') == True :
                print('ALREADY COMPUTED', flush=True)
                continue
        else:
            if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}_bi.npy') == True :
                print('ALREADY COMPUTED', flush=True)
                continue

        #### compute surrogates
        surrogates_n_chan = np.zeros((3, data_tmp.shape[0], stretch_point_surrogates_MVL_Cxy))

        respfeatures_i = respfeatures_allcond[cond][session_i]

        #n_chan = 0
        def compute_surrogates_cyclefreq_nchan(n_chan):

            print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

            x = data_tmp[n_chan,:]

            surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates_MVL_Cxy))

            #surr_i = 0
            for surr_i in range(n_surrogates_cyclefreq):

                # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

                x_shift = shuffle_Cxy(x)
                #y_shift = shuffle_Cxy(y)

                x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates_MVL_Cxy, x_shift, srate)

                x_stretch_mean = np.mean(x_stretch, axis=0)

                surrogates_val_tmp[surr_i,:] = x_stretch_mean

            mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
            surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
            percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
            percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

            up_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_up,:]
            dw_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_dw,:]

            return mean_surrogate_tmp, up_percentile_values_tmp, dw_percentile_values_tmp

        compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan, _ in enumerate(chan_list_ieeg))

        #### fill results
        for n_chan in range(np.size(data_tmp,0)):

            surrogates_n_chan[0, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][0]
            surrogates_n_chan[1, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][1]
            surrogates_n_chan[2, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][2]
        
        #### save
        if monopol:
            np.save(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}.npy', surrogates_n_chan)
        else:
            np.save(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}_bi.npy', surrogates_n_chan)

        print('done', flush=True)






################################
######## MI / MVL ########
################################




#x = x_stretch_linear
def shuffle_windows(x):

    n_cycles_stretch = int( x.shape[0]/stretch_point_surrogates_MVL_Cxy )

    shuffle_win = np.zeros(( n_cycles_stretch, stretch_point_surrogates_MVL_Cxy ))

    for cycle_i in range(n_cycles_stretch):

        cut_i = np.random.randint(0, x.shape[0]-stretch_point_surrogates_MVL_Cxy)
        shuffle_win[cycle_i,:] = x[int(cut_i):int(cut_i+stretch_point_surrogates_MVL_Cxy)]

    x_shuffled = np.mean(shuffle_win, axis=0)

    if debug:
        plt.plot(x_shuffled)
        plt.show()

    return x_shuffled



def precompute_MI(sujet, band_prep, cond, session_i, monopol):

    print(cond, flush=True)

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)
    respfeatures_i = respfeatures_allcond[cond][session_i]
    chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i, monopol)

    if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_MI_{band_prep}.npy') == True :
        print('ALREADY COMPUTED', flush=True)
        return

    #### compute surrogates
    #n_chan = 95
    def compute_surrogates_cyclefreq_nchan(n_chan):

        print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

        x = data_tmp[n_chan,:]
        x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates_MVL_Cxy, x, srate)
        x_stretch_linear = x_stretch.reshape(-1) 

        surrogates_stretch_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates_MVL_Cxy))

        for surr_i in range(n_surrogates_cyclefreq):

            # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

            surrogates_stretch_tmp[surr_i,:] = shuffle_windows(x_stretch_linear)

        #### compute MI
        MI_surrogates_i = np.array([])
        MI_bin_i = int(stretch_point_surrogates_MVL_Cxy / MI_n_bin)
        x_bin_surr = np.zeros(( stretch_point_surrogates_MVL_Cxy, MI_n_bin ))
        for surr_i in range(n_surrogates_cyclefreq):

            x = surrogates_stretch_tmp[surr_i,:]

            x_bin = np.zeros(( MI_n_bin ))

            for bin_i in range(MI_n_bin):
                x_bin[bin_i] = np.mean(x[MI_bin_i*bin_i:MI_bin_i*(bin_i+1)])

            # x += np.abs(x.min())*2
            # x = x/np.sum(x)
            x_bin += np.abs(x_bin.min())*2
            x_bin = x_bin/np.sum(x_bin)

            x_bin_surr[surr_i, :] = x_bin
            
            MI_surrogates_i = np.append(MI_surrogates_i, Shannon_MI(x_bin))

        if debug:
            times_binned = np.arange(int(stretch_point_surrogates_MVL_Cxy/MI_n_bin), stretch_point_surrogates_MVL_Cxy, int(stretch_point_surrogates_MVL_Cxy/MI_n_bin))
            _99th = np.percentile(MI_surrogates_i, 99) 
            plot_i = np.where(MI_surrogates_i > _99th)[0]
            for i in plot_i:
                plt.plot(np.mean(x_stretch,axis=0), label='original')
                plt.plot(x_bin_surr[i,:], label='shuffle')
                plt.title(f'MI : {MI_surrogates_i[i]}')
                plt.legend()
                plt.show()

            for i in range(n_surrogates_cyclefreq):
                plt.plot(surrogates_stretch_tmp[i,:])
            plt.plot(np.mean(x_stretch,axis=0), linewidth=5)
            plt.show()

        return MI_surrogates_i

    compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(data_tmp.shape[0]))

    #### fill results
    MI_surrogates = np.zeros(( data_tmp.shape[0], n_surrogates_cyclefreq ))

    for n_chan in range(data_tmp.shape[0]):

        MI_surrogates[n_chan,:] = compute_surrogates_cyclefreq_results[n_chan]

    #### verif
    if debug:
        count, values, fig = plt.hist(MI_surrogates[95,:])
        plt.vlines(np.percentile(MI_surrogates[0,:], 99), ymin=0, ymax=count.max())
        plt.vlines(np.percentile(MI_surrogates[0,:], 95), ymin=0, ymax=count.max())
        plt.show()
    
    #### save
    np.save(f'{sujet}_{cond}_{str(session_i+1)}_MI_{band_prep}.npy', MI_surrogates)

    print('done', flush=True)








def precompute_MVL(sujet, band_prep, cond, monopol):

    #session_i = 0
    for session_i in range(session_count[cond]):
    
        print(f'{cond} {session_i+1}', flush=True)

        #### load params
        respfeatures_allcond = load_respfeatures(sujet)
        respfeatures_i = respfeatures_allcond[cond][session_i]
        chan_list, chan_list_ieeg = get_chanlist(sujet, monopol)

        #### load data
        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
        data_tmp = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[:len(chan_list_ieeg),:]

        if monopol:
            if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_MVL_{band_prep}.npy'):
                print('ALREADY COMPUTED', flush=True)
                continue
        else:
            if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_MVL_{band_prep}_bi.npy'):
                print('ALREADY COMPUTED', flush=True)
                continue

        #### compute surrogates
        #n_chan = 0
        def compute_surrogates_cyclefreq_nchan(n_chan):

            print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

            #### stretch
            os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
            x = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[n_chan,:]
            x_rscore = rscore(x)
            x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates_MVL_Cxy, x_rscore, srate)

            MVL_nchan = get_MVL(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())

            x_stretch_linear = x_stretch.reshape(-1) 

            #### surrogates
            surrogates_stretch_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates_MVL_Cxy))

            for surr_i in range(n_surrogates_cyclefreq):

                # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

                surrogates_stretch_tmp[surr_i,:] = shuffle_windows(x_stretch_linear)

            #### compute MVL
            MVL_surrogates_i = np.array([])
            for surr_i in range(n_surrogates_cyclefreq):

                x = surrogates_stretch_tmp[surr_i,:]
                
                MVL_surrogates_i = np.append(MVL_surrogates_i, get_MVL(x-x.min()))

            return MVL_nchan, MVL_surrogates_i

        compute_surrogates_MVL = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan, _ in enumerate(chan_list_ieeg))

        #### fill results
        MVL_surrogates = np.zeros(( data_tmp.shape[0], n_surrogates_cyclefreq ))
        MVL_val = np.zeros(( data_tmp.shape[0] ))

        for n_chan in range(data_tmp.shape[0]):

            MVL_surrogates[n_chan,:] = compute_surrogates_MVL[n_chan][1]
            MVL_val[n_chan] = compute_surrogates_MVL[n_chan][0]

        #### verif
        if debug:
            n_chan = 95
            count, values, fig = plt.hist(MVL_surrogates[n_chan,:])
            plt.vlines(np.percentile(MVL_surrogates[n_chan,:], 99), ymin=0, ymax=count.max())
            plt.vlines(np.percentile(MVL_surrogates[n_chan,:], 95), ymin=0, ymax=count.max())
            plt.vlines(MVL_val[n_chan], ymin=0, ymax=count.max(), color='r')
            plt.show()
        
        #### save
        if monopol:
            np.save(f'{sujet}_{cond}_{str(session_i+1)}_MVL_{band_prep}.npy', MVL_surrogates)
        else:
            np.save(f'{sujet}_{cond}_{str(session_i+1)}_MVL_{band_prep}_bi.npy', MVL_surrogates)

        print('done', flush=True)









################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    list_params = []
    for sujet in sujet_list_FR_CV:    
        if sujet in sujet_list:
            conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']
        else:
            conditions = ['FR_CV']
        for monopol in [True, False]:
            band_prep = 'wb'
            for cond in conditions:
                list_params.append([sujet, band_prep, cond, monopol])

    execute_function_in_slurm_bash('n06_precompute_Cxy_MVL', 'precompute_surrogates_coh', list_params)
    #sync_folders__push_to_crnldata()

    #sujet = sujet_list_FR_CV[10]
    for sujet in sujet_list_FR_CV:    
        if sujet in sujet_list:
            conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']
        else:
            conditions = ['FR_CV']
        for monopol in [True, False]:
            band_prep = 'wb'
            for cond in conditions:
                precompute_surrogates_coh(sujet, band_prep, cond, monopol)



    # list_params = []
    # for sujet in sujet_list_FR_CV:    
    #     if sujet in sujet_list:
    #         conditions = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']
    #     else:
    #         conditions = ['FR_CV']
    #     for monopol in [True, False]:
    #         band_prep = 'wb'
    #         for cond in conditions:
    #             list_params.append([sujet, band_prep, cond, monopol])

    # execute_function_in_slurm_bash('n06_precompute_Cxy_MVL', 'precompute_MVL', list_params)
    # #sync_folders__push_to_crnldata()
      







