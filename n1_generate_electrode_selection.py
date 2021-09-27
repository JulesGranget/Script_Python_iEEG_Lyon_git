
import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

from n0_config import *

debug = False

#### to change chan name
def modify_name(chan_list):
    
    chan_list_modified = []
    chan_list_keep = []

    for nchan in chan_list:

        #### what we remove
        if nchan.find("+") != -1:
            continue

        if np.sum([str.isalpha(str_i) for str_i in nchan]) >= 2 and nchan.find('p') == -1:
            continue

        if nchan.find('ECG') != -1:
            continue

        if nchan.find('.') != -1:
            continue

        if nchan.find('*') != -1:
            continue

        #### what we do to chan we keep
        else:

            nchan_mod = nchan.replace(' ', '')
            nchan_mod = nchan_mod.replace("'", 'p')

            if nchan_mod.find('p') != -1:
                split = nchan_mod.split('p')
                letter_chan = split[0]

                if len(split[1]) == 1:
                    num_chan = '0' + split[1] 
                else:
                    num_chan = split[1]

                chan_list_modified.append(letter_chan + 'p' + num_chan)
                chan_list_keep.append(nchan)
                continue

            if nchan_mod.find('p') == -1:
                letter_chan = nchan_mod[0]

                split = nchan_mod[1:]

                if len(split) == 1:
                    num_chan = '0' + split
                else:
                    num_chan = split

                chan_list_modified.append(letter_chan + num_chan)
                chan_list_keep.append(nchan)
                continue


    return chan_list_modified, chan_list_keep

#trc_filename = 'LYONNEURO_2021_GOBc_RESPI.TRC'
def extract_chanlist():

    print('#### EXTRACT ####')

    os.chdir(os.path.join(path_data,sujet))

    # identify number of trc file
    trc_file_names = glob.glob('*.TRC')

    # extract chanlist one by one
    chan_list_whole = []
    #file_i, file_name = 0, trc_file_names[1]
    for file_i, file_name in enumerate(trc_file_names):

        # current file
        print(file_name)

        # extract segment with neo
        reader = neo.MicromedIO(filename=file_name)
        seg = reader.read_segment()
        print('len seg : ' + str(len(seg.analogsignals)))
        
        # extract data
        chan_list_whole_file = []
        #anasig = seg.analogsignals[2]
        for seg_i, anasig in enumerate(seg.analogsignals):
            
            chan_list_whole_file.append(anasig.array_annotations['channel_names'].tolist()) # extract chan

        # concatenate data
        for seg_i in range(len(chan_list_whole_file)):
            if seg_i == 0 :
                chan_list_file = chan_list_whole_file[seg_i]
            else :
                [chan_list_file.append(chan_list_whole_file[seg_i][i]) for i in range(np.size(chan_list_whole_file[seg_i]))]


        # fill containers
        chan_list_whole.append(chan_list_file)

    # verif
    #chan_list
    #srate
    #(len(data[0,:])/srate)/60
    #x = data[10,:]
    #plt.plot(x)
    #plt.vlines([events[i][1] for i in range(len(events))], ymax=np.max(x), ymin=np.min(x))
    #plt.show()

    # concatenate
    chan_list = chan_list_whole[0]

    if len(trc_file_names) > 1 :
        for trc_i in range(len(trc_file_names)-1): 

            trc_i += 1

            if chan_list != chan_list_whole[trc_i]:
                print('not the same chan list')
                exit()


    #### modify chan_name
    chan_list_modified, chan_list_keep = modify_name(chan_list)

    #### sort
    chan_list_trc = list(np.sort(chan_list_keep))

    return chan_list_trc



def generate_plot_loca(chan_list_trc):

    #### get the patient date
    os.chdir(os.path.join(path_data, sujet))
    trc_file_names = glob.glob('*.TRC')
    header_trc = trc_file_names[0][:14]

    #### open loca file
    os.chdir(os.path.join(path_data, sujet, 'anatomy'))

    parcellisation = pd.read_excel(header_trc + '_' + sujet + '.xlsx')

    new_cols = parcellisation.iloc[1,:].values.tolist()
    parcellisation.columns = new_cols

    parcellisation = parcellisation.drop(labels=[0, 1], axis=0)
    new_indexs = range(len(parcellisation['contact'].values))
    parcellisation.index = new_indexs

    #### open nomenclature
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    #### identify if parcellisation miss plots
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    plot_csv_extract = []
    for i in range(len(parcellisation.index)):
        if len(str(parcellisation.iloc[i,0])) <= 4 and str(parcellisation.iloc[i,0]) != 'nan':
            plot_csv_extract.append(parcellisation.iloc[i,0])

    plot_csv, test2 = modify_name(plot_csv_extract)

    chan_list_modified, chan_list_keep = modify_name(chan_list_trc)
    
    miss_plot = []
    chan_list_trc_rmw = []
    for nchan_i, nchan in enumerate(chan_list_modified):
        if nchan in plot_csv :
            chan_list_trc_rmw.append(chan_list_keep[nchan_i])
            continue
        else:
            miss_plot.append(nchan)
            

    #### export missed plots
    os.chdir(os.path.join(path_anatomy, sujet))
    miss_plot_textfile = open(sujet + "_miss_in_csv.txt", "w")
    for element in miss_plot:
        miss_plot_textfile.write(element + "\n")
    miss_plot_textfile.close()

    keep_plot_textfile = open(sujet + "_trcplot_in_csv.txt", "w")
    for element in chan_list_trc_rmw:
        keep_plot_textfile.write(element + "\n")
    keep_plot_textfile.close()

    #### categories to fill
    columns = ['subject', 'plot', 'MNI', 'freesurfer_destrieux', 'correspondance_ROI', 'correspondance_lobes', 'comparison', 'abscent',	
                'noisy_signal', 'inSOZ', 'not_in_atlas', 'select', 'localisation_corrected', 'lobes_corrected']

    #### find correspondances
    freesurfer_destrieux = [ parcellisation['Freesurfer'][parcellisation['contact'] == nchan].values.tolist()[0] for nchan in plot_csv_extract]
    if debug:
        for nchan in plot_csv_extract:
            print(nchan)
            parcellisation['Freesurfer'][parcellisation['contact'] == nchan].values.tolist()[0]

    correspondance_ROI = []
    correspondance_lobes = []
    for parcel_i in freesurfer_destrieux:
        if parcel_i[0] == 'c':
            parcel_i_chunk = parcel_i[7:]
        elif parcel_i[0] == 'L':
            parcel_i_chunk = parcel_i[5:]
        elif parcel_i[0] == 'R':
            parcel_i_chunk = parcel_i[6:]
        elif parcel_i[0] == 'W':
            parcel_i_chunk = 'Cerebral-White-Matter'
        else:
            parcel_i_chunk = parcel_i
        
        correspondance_ROI.append(nomenclature['Our correspondances'][nomenclature['Labels'] == parcel_i_chunk].values[0])
        correspondance_lobes.append(nomenclature['Lobes'][nomenclature['Labels'] == parcel_i_chunk].values[0])

    #### generate df
    electrode_select_dict = {}
    for ncol in columns:
        if ncol == 'subject':
            electrode_select_dict[ncol] = [sujet] * len(plot_csv_extract)
        elif ncol == 'plot':
            electrode_select_dict[ncol] = plot_csv
        elif ncol == 'MNI':
            electrode_select_dict[ncol] = [ parcellisation['MNI'][parcellisation['contact'] == nchan].values.tolist()[0] for nchan in plot_csv_extract]
        elif ncol == 'freesurfer_destrieux':
            electrode_select_dict[ncol] = freesurfer_destrieux
        elif ncol == 'localisation_corrected' or ncol == 'lobes_corrected':
            electrode_select_dict[ncol] = [0] * len(plot_csv_extract)
        elif ncol == 'correspondance_ROI':
            electrode_select_dict[ncol] = correspondance_ROI
        elif ncol == 'correspondance_lobes':
            electrode_select_dict[ncol] = correspondance_lobes
        else :
            electrode_select_dict[ncol] = [1] * len(plot_csv_extract)


    #### generate df and save
    electrode_select_df = pd.DataFrame(electrode_select_dict, columns=columns)

    os.chdir(os.path.join(path_anatomy, sujet))
    
    electrode_select_df.to_excel(sujet + '_plot_loca.xlsx')

    return
        


if __name__== '__main__':

    #### verif if file exist
    os.chdir(os.path.join(path_anatomy, sujet))
    if os.path.exists(sujet + '_plot_loca.xlsx'):
        print('#### ALREADY COMPUTED ####')
        exit()

    #### execute
    chan_list_trc = extract_chanlist()
    generate_plot_loca(chan_list_trc)







