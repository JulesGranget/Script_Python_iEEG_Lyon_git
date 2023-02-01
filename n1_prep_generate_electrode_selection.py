
import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False

#trc_filename = 'LYONNEURO_2021_GOBc_RESPI.TRC'
def extract_chanlist(sujet):

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



def generate_plot_loca(sujet, chan_list_trc):

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
        



################################
######## BIPOLARIZATION ########
################################


def bipolarize_anatomy_name(plot_name_sel):

    #### bipolarize anatomy localization
    correspondance_bipol = []

    for plot_i, plot_name in enumerate(plot_name_sel):

        if plot_i == len(plot_name_sel)-1:

            continue

        correspondance_bipol.append(f'{plot_name_sel[plot_i]}-{plot_name_sel[plot_i+1]}')

    return correspondance_bipol






def bipolarize_anatomy_localization(anat_selection):

    #### bipolarize anatomy localization
    correspondance_ROI_bipol = []
    #plot_i, plot_name = 0, correspondance_ROI[0]
    for plot_i, plot_name in enumerate(anat_selection):

        if plot_i == len(anat_selection)-1:
            
            continue

        if anat_selection[plot_i+1] == plot_name:
            
            correspondance_ROI_bipol.append(plot_name)
            continue

        if anat_selection[plot_i] not in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] in ['WM', 'unknown', 'Unknown', 'ventricule']:
            
            correspondance_ROI_bipol.append(plot_name)
            continue

        if anat_selection[plot_i] in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] in ['WM', 'unknown', 'Unknown', 'ventricule']:
            
            correspondance_ROI_bipol.append(plot_name)
            continue

        if anat_selection[plot_i] in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] not in ['WM', 'unknown', 'Unknown', 'ventricule']:

            correspondance_ROI_bipol.append(anat_selection[plot_i+1])
            continue

        if anat_selection[plot_i] not in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] not in ['WM', 'unknown', 'Unknown', 'ventricule']:

            correspondance_ROI_bipol.append(anat_selection[plot_i])
            continue

    return correspondance_ROI_bipol

        



#sujet = sujet_list_FR_CV[8]
def generate_plot_loca_bipolaire(sujet):

    #### open loca file
    os.chdir(os.path.join(path_anatomy, sujet))

    df = pd.read_excel(f'{sujet}_plot_loca.xlsx')
    df = df.drop(['Unnamed: 0'], axis=1)

    df = df.sort_values('plot')
    df.index = range(df.index.shape[0])

    #### separate electrodes
    plot_name_bip = []
    plot_ROI_bip = []
    plot_Lobes_bip = []

    verif_count = 0
    verif_count_name_bip = 0
    verif_count_anat_ROI_bip = 0
    verif_count_anat_Lobes_bip = 0

    #### discriminate electrodes  
    if sujet.find('pat') == -1:
        plot_list_unique = np.unique(np.array([plot_i.split('0')[0] for plot_i in df['plot'].values]))
        plot_list_unique = np.unique(np.array([plot_i.split('1')[0] for plot_i in plot_list_unique]))
    else:
        plot_list_unique = np.unique(np.array([plot_i.split('_')[0] for plot_i in df['plot'].values]))

    #plot_unique_i = plot_list_unique[1]
    for plot_unique_i in plot_list_unique:
        
        plot_selection_i = np.array([plot_i for plot_i, plot_name in enumerate(df['plot'].values) if plot_name.find(plot_unique_i) != -1 and plot_name.find('p') == plot_unique_i.find('p')])
        
        verif_count += plot_selection_i.shape[0]-1
        
        plot_name_sel = df['plot'][plot_selection_i].values
        plot_name_sel_bipol = bipolarize_anatomy_name(plot_name_sel)

        anat_selection_ROI = df['correspondance_ROI'][plot_selection_i].values
        anat_selection_Lobes = df['correspondance_lobes'][plot_selection_i].values

        anat_selection_ROI_bi = bipolarize_anatomy_localization(anat_selection_ROI)
        anat_selection_Lobes_bi = bipolarize_anatomy_localization(anat_selection_Lobes)

        verif_count_name_bip += len(plot_name_sel_bipol)
        verif_count_anat_ROI_bip += len(anat_selection_ROI_bi)
        verif_count_anat_Lobes_bip += len(anat_selection_Lobes_bi)

        plot_name_bip.extend(plot_name_sel_bipol)
        plot_ROI_bip.extend(anat_selection_ROI_bi)
        plot_Lobes_bip.extend(anat_selection_Lobes_bi)

    #### verif bipol
    if verif_count != verif_count_anat_ROI_bip or verif_count != verif_count_anat_Lobes_bip or verif_count != verif_count_name_bip:
        raise ValueError('!! WARNING !! bipolarization issue')

    #### update df for bipol
    df = df.iloc[:verif_count, :]
    df = df.drop('MNI', axis=1)
    df = df.drop('freesurfer_destrieux', axis=1)
    df['plot'] = plot_name_bip
    df['correspondance_ROI'] = plot_ROI_bip
    df['correspondance_lobes'] = plot_Lobes_bip

    #### export
    os.chdir(os.path.join(path_anatomy, sujet))
    
    df.to_excel(sujet + '_plot_loca_bi.xlsx')

    return
        


        




################################
######## EXECUTE ########
################################


if __name__== '__main__':

    #sujet = sujet_list_FR_CV[5]
    for sujet in sujet_list_FR_CV:

        print(f'#### {sujet}')

        construct_token = generate_folder_structure(sujet)

        if construct_token != 0 :
            
            print('Folder structure has been generated')
            print('Lauch the script again for electrode selection')

        else:

            #### execute
            os.chdir(os.path.join(path_anatomy, sujet))
            if os.path.exists(sujet + '_plot_loca.xlsx'):
                print('#### MONOPOL LREADY COMPUTED ####')
            else:
                chan_list_trc = extract_chanlist(sujet)
                generate_plot_loca(sujet, chan_list_trc)

            if os.path.exists(sujet + '_plot_loca_bi.xlsx'):
                print('#### BIPOL LREADY COMPUTED ####')
            else:
                generate_plot_loca_bipolaire(sujet)







