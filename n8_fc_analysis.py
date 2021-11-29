

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n3_respi_analysis import analyse_resp

from n0_config import *
from n0bis_analysis_functions import *


debug = False



################################
######## LOAD DATA ########
################################

#### data params
conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(conditions_allsubjects)

#### respfeatures
respfeatures_allcond = load_respfeatures(conditions)
respi_ratio_allcond = get_all_respi_ratio(conditions, respfeatures_allcond)

#### localization
dict_loca = get_electrode_loca()
df_loca = get_loca_df()





#######################################
############# ISPC & PLI #############
#######################################

#data = data_tmp
def compute_fc_metrics(band_prep, data, freq, band, cond, session_i):
    
    #### check if already computed
    pli_mat = np.array([0])
    ispc_mat = np.array([0])

    os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'matrix'))
    if os.path.exists(sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1) + '.npy'):
        print('ALREADY COMPUTED : ' + sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1))
        ispc_mat = np.load(sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1) + '.npy')

    os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'matrix'))
    if os.path.exists(sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1) + '.npy'):
        print('ALREADY COMPUTED : ' + sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1))
        pli_mat = np.load(sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1) + '.npy')

    if len(ispc_mat) != 1 and len(pli_mat) != 1:
        return pli_mat, ispc_mat 
    

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

    data = data[:len(chan_list_ieeg),:]

    #### compute all convolution
    os.chdir(path_memmap)
    convolutions = np.memmap(sujet + '_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(chan_list_ieeg), nfrex, data.shape[1]))

    print('CONV')

    def convolution_x_wavelets_nchan(nchan):

        if nchan/np.size(data,0) % .25 <= .01:
            print("{:.2f}".format(nchan/len(chan_list_ieeg)))
        
        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

        x = data[nchan,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan,:,:] = nchan_conv

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan) for nchan in range(np.size(data,0)))

    #### compute metrics
    pli_mat = np.zeros((np.size(data,0),np.size(data,0)))
    ispc_mat = np.zeros((np.size(data,0),np.size(data,0)))

    print('COMPUTE')

    for seed in range(np.size(data,0)) :

        if seed/len(chan_list_ieeg) % .25 <= .01:
            print("{:.2f}".format(seed/len(chan_list_ieeg)))

        def compute_ispc_pli(nchan):

            if nchan == seed : 
                return
                
            else :

                # initialize output time-frequency data
                ispc = np.zeros((nfrex))
                pli  = np.zeros((nfrex))

                # compute metrics
                for fi in range(nfrex):
                    
                    as1 = convolutions[seed][fi,:]
                    as2 = convolutions[nchan][fi,:]

                    # collect "eulerized" phase angle differences
                    cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                    
                    # compute ISPC and PLI (and average over trials!)
                    ispc[fi] = np.abs(np.mean(cdd))
                    pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

            # compute mean
            mean_ispc = np.mean(ispc,0)
            mean_pli = np.mean(pli,0)

            return mean_ispc, mean_pli

        compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(nchan) for nchan in range(np.size(data,0)))
        
        for nchan in range(np.size(data,0)) :
                
            if nchan == seed:

                continue

            else:
                    
                ispc_mat[seed,nchan] = compute_ispc_pli_res[nchan][0]
                pli_mat[seed,nchan] = compute_ispc_pli_res[nchan][1]

    #### save matrix
    os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'matrix'))
    np.save(sujet + '_ISPC_' + band + '_' + cond + '_' + str(session_i+1) + '.npy', ispc_mat)

    os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'matrix'))
    np.save(sujet + '_PLI_' + band + '_' + cond + '_' + str(session_i+1) + '.npy', pli_mat)

    #### supress mmap
    os.chdir(path_memmap)
    os.remove(sujet + '_fc_convolutions.dat')
    
    return pli_mat, ispc_mat


################################
######## EXECUTE ########
################################

pli_allband = {}
ispc_allband = {}

#band_prep_i, band_prep = 0, 'lf'
for band_prep_i, band_prep in enumerate(band_prep_list):

    #band = 'theta'
    for band in freq_band_list[band_prep_i].keys():

        if band == 'whole' :

            continue

        else: 

            freq = freq_band_fc_analysis.get(band)

            pli_allcond = {}
            ispc_allcond = {}

            #cond_i, cond = 0, conditions[0]
            #session_i = 0
            for cond_i, cond in enumerate(conditions) :

                print(band, cond)

                if len(respfeatures_allcond.get(cond)) == 1:

                    data_tmp = load_data(band_prep, cond, 0)
                    pli_mat, ispc_mat = compute_fc_metrics(band_prep, data_tmp, freq, band, cond, 0)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]

                elif len(respfeatures_allcond.get(cond)) > 1:

                    load_ispc = []
                    load_pli = []

                    for session_i in range(len(respfeatures_allcond.get(cond))):
                        
                        data_tmp = load_data(band_prep, cond, session_i)
                        pli_mat, ispc_mat = compute_fc_metrics(band_prep, data_tmp, freq, band, cond, session_i)
                        load_ispc.append(ispc_mat)
                        load_pli.append(pli_mat)

                    pli_allcond[cond] = load_pli
                    ispc_allcond[cond] = load_ispc

            pli_allband[band] = pli_allcond
            ispc_allband[band] = ispc_allcond

#### verif

if debug == True:

    for band, freq in freq_band_fc_analysis.items():

        for cond_i, cond in enumerate(conditions) :

            print(band, cond, len(pli_allband.get(band).get(cond)))
            print(band, cond, len(ispc_allband.get(band).get(cond)))



#### reduce to one cond

#### generate dict to fill
ispc_allband_reduced = {}
pli_allband_reduced = {}

for band, freq in freq_band_fc_analysis.items():

    ispc_allband_reduced[band] = {}
    pli_allband_reduced[band] = {}

    for cond_i, cond in enumerate(conditions) :

        ispc_allband_reduced.get(band)[cond] = []
        pli_allband_reduced.get(band)[cond] = []

    #### fill
for band_prep_i, band_prep in enumerate(band_prep_list):

    for band, freq in freq_band_list[band_prep_i].items():

        if band == 'whole' :

            continue

        else:

            for cond_i, cond in enumerate(conditions) :

                if len(respfeatures_allcond.get(cond)) == 1:

                    ispc_allband_reduced.get(band)[cond] = ispc_allband.get(band).get(cond)[0]
                    pli_allband_reduced.get(band)[cond] = pli_allband.get(band).get(cond)[0]

                elif len(respfeatures_allcond.get(cond)) > 1:

                    load_ispc = []
                    load_pli = []

                    for session_i in range(len(respfeatures_allcond.get(cond))):

                        if session_i == 0 :

                            load_ispc.append(ispc_allband.get(band).get(cond)[session_i])
                            load_pli.append(pli_allband.get(band).get(cond)[session_i])

                        else :
                        
                            load_ispc = (load_ispc[0] + ispc_allband.get(band).get(cond)[session_i]) / 2
                            load_pli = (load_pli[0] + pli_allband.get(band).get(cond)[session_i]) / 2

                    pli_allband_reduced.get(band)[cond] = load_pli
                    ispc_allband_reduced.get(band)[cond] = load_ispc




################################
######## SAVE FIG ########
################################

print('######## SAVEFIG FC ########')

#### sort matrix

def sort_mat(mat):

    mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
    for i_before_sort_r, i_sort_r in enumerate(df_sorted.index.values):
        for i_before_sort_c, i_sort_c in enumerate(df_sorted.index.values):
            mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

    return mat_sorted

#### verify sorting
#mat = pli_allband_reduced.get(band).get(cond)
#mat_sorted = sort_mat(mat)
#plt.matshow(mat_sorted)
#plt.show()

#### prepare sort
os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'figures'))

df_sorted = df_loca.sort_values(['lobes', 'ROI'])
chan_name_sorted = df_sorted['ROI'].values.tolist()


chan_name_sorted_mat = []
rep_count = 0
for i, name_i in enumerate(chan_name_sorted):
    if i == 0:
        chan_name_sorted_mat.append(name_i)
        continue
    else:
        if name_i == chan_name_sorted[i-(rep_count+1)]:
            chan_name_sorted_mat.append('')
            rep_count += 1
            continue
        if name_i != chan_name_sorted[i-(rep_count+1)]:
            chan_name_sorted_mat.append(name_i)
            rep_count = 0
            continue
            

#### count for cond subpolt
if len(conditions) == 1:
    nrows, ncols = 1, 0
elif len(conditions) == 2:
    nrows, ncols = 1, 1
elif len(conditions) == 3:
    nrows, ncols = 2, 1
elif len(conditions) == 4:
    nrows, ncols = 2, 2
elif len(conditions) == 5:
    nrows, ncols = 3, 2
elif len(conditions) == 6:
    nrows, ncols = 3, 3


#### identify scale
scale = {'ispc' : {'min' : {}, 'max' : {}}, 'pli' : {'min' : {}, 'max' : {}}}
for band, freq in freq_band_fc_analysis.items():

    band_ispc = {'min' : [], 'max' : []}
    band_pli = {'min' : [], 'max' : []}

    for cond_i, cond in enumerate(conditions):
        band_ispc['max'].append(np.max(ispc_allband_reduced[band][cond]))
        band_ispc['min'].append(np.min(ispc_allband_reduced[band][cond]))
        
        band_pli['max'].append(np.max(pli_allband_reduced[band][cond]))
        band_pli['min'].append(np.min(pli_allband_reduced[band][cond]))

    scale['ispc']['max'][band] = np.max(band_ispc['max'])
    scale['ispc']['min'][band] = np.min(band_ispc['min'])
    scale['pli']['max'][band] = np.max(band_pli['max'])
    scale['pli']['min'][band] = np.min(band_pli['min'])


#### ISPC
#band_prep_i, band_prep, nchan, band, freq = 0, 'lf', 0, 'theta', [2, 10]
for band, freq in freq_band_fc_analysis.items():

    #### graph
    fig = plt.figure(facecolor='black')
    if ncols == 0:
        for cond_i, cond in enumerate(conditions):
            mne.viz.plot_connectivity_circle(sort_mat(ispc_allband_reduced.get(band).get(cond)), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig)
    else:
        for cond_i, cond in enumerate(conditions):
            mne.viz.plot_connectivity_circle(sort_mat(ispc_allband_reduced.get(band).get(cond)), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
    plt.suptitle('ISPC_' + band, color='w')
    fig.set_figheight(10)
    fig.set_figwidth(12)
    #fig.show()

    fig.savefig(sujet + '_ISPC_' + band + '_graph', dpi = 100)

    #### matrix
    if ncols == 0:
        fig = plt.figure(figsize=(20,10)) 
    else:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

    for row_i in range(nrows):
        for col_i in range(ncols):
            cond = conditions[col_i + row_i]
            if ncols != 0:
                ax = axs[row_i, col_i]
                ax.matshow(sort_mat(ispc_allband_reduced[band][cond]), vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
                ax.set_title(cond)
            else:
                plt.matshow(sort_mat(ispc_allband_reduced[band][cond]))
            ax.set_yticks(range(len(df_loca['ROI'].values)))
            ax.set_yticklabels(chan_name_sorted_mat)
                
    plt.suptitle('ISPC_' + band)
    #plt.show()
                
    fig.savefig(sujet + '_ISPC_' + band + '_mat', dpi = 100)


#### PLI

os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'figures'))

#band_prep_i, band_prep, nchan, band, freq = 0, 'lf', 0, 'theta', [2, 10]
for band, freq in freq_band_fc_analysis.items():

    fig = plt.figure(facecolor='black')
    if ncols == 0:
        for cond_i, cond in enumerate(conditions):
            mne.viz.plot_connectivity_circle(sort_mat(pli_allband_reduced.get(band).get(cond)), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig)
    else:
        for cond_i, cond in enumerate(conditions):
            mne.viz.plot_connectivity_circle(sort_mat(pli_allband_reduced.get(band).get(cond)), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
    plt.suptitle('PLI_' + band, color='w')
    fig.set_figheight(10)
    fig.set_figwidth(12)
    #fig.show()

    fig.savefig(sujet + '_PLI_' + band + '_graph', dpi = 100)

    #### matrix
    if ncols == 0:
        fig = plt.figure(figsize=(20,10)) 
    else:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

    for row_i in range(nrows):
        for col_i in range(ncols):
            cond = conditions[col_i + row_i]
            if ncols != 0:
                ax = axs[row_i, col_i]
                ax.matshow(pli_allband_reduced.get(band).get(cond), vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
                ax.set_title(cond)
            else:
                plt.matshow(pli_allband_reduced.get(band).get(cond))
            ax.set_yticks(range(len(df_loca['ROI'].values)))
            ax.set_yticklabels(chan_name_sorted_mat)
                
    plt.suptitle('PLI_' + band)
    #plt.show()
                
    fig.savefig(sujet + '_PLI_' + band + '_mat', dpi = 100)


