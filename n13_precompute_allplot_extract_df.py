

import os
import pandas as pd

from n00_config_params import *
from n00bis_config_analysis_functions import *


debug = False




########################################
######## COMPILATION FUNCTION ########
########################################




# def compilation_export_df_allplot(sujet_list, monopol):

#     #### generate df
#     df_export_Cxy_MVL = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'Cxy', 'Cxy_surr', 'MVL', 'MVL_surr'])
#     df_export_Pxx = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'Pxx', 'phase'])
#     # df_export_graph_DFC = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'CPL', 'GE', 'SWN'])
#     # df_export_HRV = pd.DataFrame(columns=['sujet', 'cond', 'session', 'RDorFR', 'compute_type', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_S'])
#     # df_export_FC = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'pair', 'value'])

#     #### fill
#     for sujet in sujet_list:

#         print(sujet)

#         os.chdir(os.path.join(path_results, sujet, 'df'))

#         if monopol:
#             df_export_Cxy_MVL_i = pd.read_excel(f'{sujet}_df_Cxy_MVL.xlsx')
#             df_export_Pxx_i = pd.read_excel(f'{sujet}_df_Pxx.xlsx')
#             # df_export_graph_DFC_i = pd.read_excel(f'{sujet}_df_graph_DFC.xlsx')
#             # df_export_HRV_i = pd.read_excel(f'{sujet}_df_HRV.xlsx')
#             # df_export_FC_i = pd.read_excel(f'{sujet}_df_FC.xlsx')

#         else:
#             df_export_Cxy_MVL_i = pd.read_excel(f'{sujet}_df_Cxy_MVL_bi.xlsx')
#             df_export_Pxx_i = pd.read_excel(f'{sujet}_df_Pxx_bi.xlsx')
#             # df_export_graph_DFC_i = pd.read_excel(f'{sujet}_df_graph_DFC_bi.xlsx')
#             # df_export_HRV_i = pd.read_excel(f'{sujet}_df_HRV_bi.xlsx')
#             # df_export_FC_i = pd.read_excel(f'{sujet}_df_FC_bi.xlsx')

#         df_export_Cxy_MVL = pd.concat([df_export_Cxy_MVL, df_export_Cxy_MVL_i])
#         df_export_Pxx = pd.concat([df_export_Pxx, df_export_Pxx_i])
#         # df_export_graph_DFC = pd.concat([df_export_graph_DFC, df_export_graph_DFC_i])
#         # df_export_HRV = pd.concat([df_export_HRV, df_export_HRV_i])
#         # df_export_FC = pd.concat([df_export_FC, df_export_FC_i])

#     #### save
#     os.chdir(os.path.join(path_results, 'allplot', 'df'))

#     if monopol:
        
#         if os.path.exists('allplot_df_Cxy_MVL.xlsx'):
#             print('Cxy_MVL : ALREADY COMPUTED')
#         else:
#             df_export_Cxy_MVL.to_excel('allplot_df_Cxy_MVL.xlsx')

#         if os.path.exists('allplot_df_Pxx.xlsx'):
#             print('Pxx : ALREADY COMPUTED')
#         else:
#             df_export_Pxx.to_excel('allplot_df_Pxx.xlsx')
        
#         # if os.path.exists('allplot_df_graph_DFC.xlsx'):
#         #     print('graph DFC : ALREADY COMPUTED')
#         # else:
#         #     df_export_graph_DFC.to_excel('allplot_df_graph_DFC.xlsx')

#         # if os.path.exists('allplot_df_HRV.xlsx'):
#         #     print('HRV : ALREADY COMPUTED')
#         # else:
#         #     df_export_HRV.to_excel('allplot_df_HRV.xlsx')

#         # if os.path.exists('allplot_df_FC.xlsx'):
#         #     print('FC : ALREADY COMPUTED')
#         # else:
#         #     df_export_FC.to_excel('allplot_df_FC.xlsx')

#     else:

#         if os.path.exists('allplot_df_Cxy_MVL_bi.xlsx'):
#             print('Cxy_MVL : ALREADY COMPUTED')
#         else:
#             df_export_Cxy_MVL.to_excel('allplot_df_Cxy_MVL_bi.xlsx')

#         if os.path.exists('allplot_df_Pxx_bi.xlsx'):
#             print('Pxx : ALREADY COMPUTED')
#         else:
#             df_export_Pxx.to_excel('allplot_df_Pxx_bi.xlsx')
        
#         # if os.path.exists('allplot_df_graph_DFC_bi.xlsx'):
#         #     print('graph DFC : ALREADY COMPUTED')
#         # else:
#         #     df_export_graph_DFC.to_excel('allplot_df_graph_DFC_bi.xlsx')

#         # if os.path.exists('allplot_df_HRV_bi.xlsx'):
#         #     print('HRV : ALREADY COMPUTED')
#         # else:
#         #     df_export_HRV.to_excel('allplot_df_HRV_bi.xlsx')

#         # if os.path.exists('allplot_df_FC_bi.xlsx'):
#         #     print('FC : ALREADY COMPUTED')
#         # else:
#         #     df_export_FC.to_excel('allplot_df_FC_bi.xlsx')
    
    


def aggregate_df_Pxx(monopol):

    #### verif computation
    if monopol:

        if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_aggregates', f'df_Pxx.xlsx')):
            print('Pxx : ALREADY COMPUTED', flush=True)
            return

    else:

        if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_aggregates', f'df_Pxx_bi.xlsx')):
            print('Pxx : ALREADY COMPUTED', flush=True)
            return

    os.chdir(os.path.join(path_results, 'allplot', 'df', 'subject_wise'))
    df_Pxx_aggregates = pd.DataFrame()

    for sujet in sujet_list_FR_CV:

        if monopol:
            _df = pd.read_excel(f'{sujet}_df_Pxx.xlsx')
        else:
            _df = pd.read_excel(f'{sujet}_df_Pxx_bi.xlsx')
                
        df_Pxx_aggregates = pd.concat([df_Pxx_aggregates, _df])

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))

    if monopol:
        df_Pxx_aggregates.to_excel(f'df_Pxx.xlsx')
    else:
        df_Pxx_aggregates.to_excel(f'df_Pxx_bi.xlsx')
            
    print('done', flush=True)



#monopol = True
def aggregate_df_MI(monopol):

    #### verif computation
    # if monopol:

    #     if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_aggregates', f'df_MI.xlsx')):
    #         print('MI : ALREADY COMPUTED', flush=True)
    #         return

    # else:

    #     if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_aggregates', f'df_MI_bi.xlsx')):
    #         print('MI : ALREADY COMPUTED', flush=True)
    #         return

    
    df_MI_aggregates = pd.DataFrame()

    for sujet in sujet_list_FR_CV:

        print(sujet)

        os.chdir(os.path.join(path_results, 'allplot', 'df', 'subject_wise'))

        if monopol:
            _df = pd.read_excel(f'{sujet}_df_MI.xlsx')
        else:
            _df = pd.read_excel(f'{sujet}_df_MI_bi.xlsx')

        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

        if monopol:
            _xr = xr.open_dataarray(f'xr_MI_{sujet}.nc')
        else:
            _xr = xr.open_dataarray(f'xr_MI_{sujet}_bi.nc')

        if sujet[:3] != 'pat' and monopol:
            _chan_list_modified, _ = modify_name(_xr['chan'].values)
            _xr['chan'] = _chan_list_modified

        MI_signi = []

        for row_i in _df.iterrows():

            mask_dw = _xr.loc[row_i[1]['cond'], row_i[1]['chan'], 'obs', row_i[1]['band']].values <= _xr.loc[row_i[1]['cond'], row_i[1]['chan'], 'down', row_i[1]['band']].values
            mask_up = _xr.loc[row_i[1]['cond'], row_i[1]['chan'], 'obs', row_i[1]['band']].values >= _xr.loc[row_i[1]['cond'], row_i[1]['chan'], 'up', row_i[1]['band']].values

            if debug: 

                plt.plot(_xr.loc[row_i[1]['cond'], row_i[1]['chan'], 'obs', row_i[1]['band']].values)
                plt.plot(_xr.loc[row_i[1]['cond'], row_i[1]['chan'], 'up', row_i[1]['band']].values, color='r')
                plt.plot(_xr.loc[row_i[1]['cond'], row_i[1]['chan'], 'down', row_i[1]['band']].values, color='r')
                plt.show()

            if mask_dw.sum() != 0 and mask_dw.sum() != 0:
                mask_thresh_dw, mask_thresh_up = mask_dw.astype('uint8'), mask_up.astype('uint8')
                nb_blobs_dw, im_with_separated_blobs_dw, stats_dw, _ = cv2.connectedComponentsWithStats(mask_thresh_dw)
                nb_blobs_up, im_with_separated_blobs_up, stats_up, _ = cv2.connectedComponentsWithStats(mask_thresh_up)
                sizes_dw, sizes_up = stats_dw[1:, -1], stats_up[1:, -1]
                if (sizes_up >= int(stretch_point_TF*0.01)).sum() != 0 and (sizes_dw >= int(stretch_point_TF*0.01)).sum() != 0:
                    MI_signi.append(True)
                else:
                    MI_signi.append(False)
            else:
                MI_signi.append(False)

        _df['MI_signi'] = MI_signi

        df_MI_aggregates = pd.concat([df_MI_aggregates, _df])

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))

    if monopol:
        df_MI_aggregates.to_excel(f'df_MI.xlsx')
    else:
        df_MI_aggregates.to_excel(f'df_MI_bi.xlsx')
            
    print('done', flush=True)




def aggregate_df_Cxy(monopol):

    #### verif computation
    if monopol:

        if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_aggregates', f'df_Cxy.xlsx')):
            print('Cxy : ALREADY COMPUTED', flush=True)
            return

    else:

        if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_aggregates', f'df_Cxy_bi.xlsx')):
            print('Cxy : ALREADY COMPUTED', flush=True)
            return

    os.chdir(os.path.join(path_results, 'allplot', 'df', 'subject_wise'))
    df_Cxy_aggregates = pd.DataFrame()

    for sujet in sujet_list_FR_CV:

        if monopol:
            _df = pd.read_excel(f'{sujet}_df_Cxy.xlsx')
        else:
            _df = pd.read_excel(f'{sujet}_df_Cxy_bi.xlsx')
                
        df_Cxy_aggregates = pd.concat([df_Cxy_aggregates, _df])

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))

    if monopol:
        df_Cxy_aggregates.to_excel(f'df_Cxy.xlsx')
    else:
        df_Cxy_aggregates.to_excel(f'df_Cxy_bi.xlsx')
            
    print('done', flush=True)







########################
######## FC ########
########################

    


def get_df_aggregates_fc(monopol):

    #cf_metric = 'WPLI'
    for cf_metric in ['ISPC', 'WPLI']:

        os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))
        if monopol:
            if os.path.exists(f"df_{cf_metric}_FC_FR_CV.xlsx") and os.path.exists(f"df_{cf_metric}_FC_ALLCOND.xlsx"):
                print(f'ALREADY COMPUTE : {cf_metric} {monopol}')
                continue
        else:
            if os.path.exists(f"df_{cf_metric}_FC_FR_CV_bi.xlsx") and os.path.exists(f"df_{cf_metric}_FC_ALLCOND_bi.xlsx"):
                print(f'ALREADY COMPUTE : {cf_metric} {monopol}')
                continue

        print(cf_metric, monopol)

        phase_i_list = {'EI' : np.arange(stretch_point_FC/4, dtype='int'), 'I' : np.arange(stretch_point_FC/4, stretch_point_FC/2, dtype='int'), 
                        'IE' : np.arange(stretch_point_FC/2, stretch_point_FC*3/4, dtype='int'), 'E' : np.arange(stretch_point_FC*3/4, stretch_point_FC, dtype='int'),
                        'W' : np.arange(stretch_point_FC, dtype='int')}

        ######## FR_CV ########

        #### extract respi
        resp_allsujet = {}

        #sujet = sujet_list_dfc_FR_CV[1]
        for sujet_i, sujet in enumerate(sujet_list_dfc_FR_CV):

            resp_allsujet[sujet] = {}

            if sujet in sujet_list:

                cond_sel = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

            else:

                cond_sel = ['FR_CV']

            respfeatures_allcond = load_respfeatures(sujet)

            for cond in cond_sel:

                respi_median = np.array([])
                for session_i in range(session_count[cond]):
                    respi_median = np.append(respi_median, respfeatures_allcond[cond][session_i]['cycle_freq'].values)
                resp_allsujet[sujet][cond] = np.median(respi_median)
        
        #### extract data
        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))

        print('FR_CV')
        
        xr_list = {}

        #sujet = sujet_list_dfc_FR_CV[1]
        for sujet_i, sujet in enumerate(sujet_list_dfc_FR_CV):

            print(sujet)
                
            if monopol:
                _xr_dfc_FR_CV = xr.open_dataarray(f'{cf_metric}_{sujet}_stretch_rscore.nc')
            else:
                _xr_dfc_FR_CV = xr.open_dataarray(f'{cf_metric}_{sujet}_stretch_rscore_bi.nc')

            _xr_dfc_FR_CV = _xr_dfc_FR_CV.loc[:,'FR_CV']
            _xr_dfc_FR_CV = _xr_dfc_FR_CV.drop_vars('cond')
            normalized_pairs = ['-'.join(sorted(pair.split('-'))) for pair in _xr_dfc_FR_CV['pair'].values]
            _xr_dfc_FR_CV['pair'] = normalized_pairs
            xr_list[sujet] = _xr_dfc_FR_CV

        pair_list = []        
        
        for _sujet in sujet_list_dfc_FR_CV:

            pair_list.extend(np.unique(xr_list[_sujet]['pair']))

        pair_list = np.unique(pair_list)

        params_pairs = {}

        for pair in pair_list:

            params_pairs[pair] = {}
            _sujet_list_sel = [_sujet for _sujet in sujet_list_dfc_FR_CV if any(xr_list[_sujet]['pair'] == pair)]
            params_pairs[pair]['sujet_list'] = _sujet_list_sel
            params_pairs[pair]['min_count'] = np.array([(xr_list[_sujet]['pair'] == pair).sum() for _sujet in _sujet_list_sel]).min()

        #### filter

        df_allpairs = pd.DataFrame()

        for pair in pair_list:

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
                _xr = _xr.expand_dims({'sujet': [_sujet]})
                _xr = _xr.rename({'pair' : 'pair_i'})
                _xr = _xr.median('cycle')
                _xr['pair_i'] = [f'{i}' for i in np.arange(params_pairs[pair]['min_count'])]

                _xr_pair.append(_xr)

            _xr_pair = xr.concat(_xr_pair, dim='sujet')
            _xr_pair = _xr_pair.expand_dims({'pair': [pair]})

            _xr_pair = _xr_pair.roll(time=int(stretch_point_FC/8))

            xr_concat_list = [_xr_pair[:,:,:,:,phase_i_list[phase_respi]].median('time') for phase_respi in phase_i_list]

            xr_phase = xr.concat(xr_concat_list, dim='phase')
            xr_phase = xr_phase.assign_coords(phase=['EI', 'I', 'IE', 'E', 'W'])

            df_allpairs = pd.concat([df_allpairs, xr_phase.to_dataframe(name='fc').reset_index()])

        resp_vec = []
        for row_i, row_df in df_allpairs.iterrows():

            resp_vec.append(resp_allsujet[row_df['sujet']]['FR_CV'])

        df_allpairs['resp'] = resp_vec

        os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))
        if monopol:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_FR_CV.xlsx")
        else:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_FR_CV_bi.xlsx")


        ######## ALLCOND ########

        print('ALLCOND')

        #### extract respi
        resp_allsujet = {}

        #sujet = sujet_list_dfc_FR_CV[1]
        for sujet_i, sujet in enumerate(sujet_list_dfc_allcond):

            resp_allsujet[sujet] = {}

            if sujet in sujet_list:

                cond_sel = ['FR_CV', 'RD_CV', 'RD_FV', 'RD_SV']

            else:

                cond_sel = ['FR_CV']

            respfeatures_allcond = load_respfeatures(sujet)

            for cond in cond_sel:

                respi_median = np.array([])
                for session_i in range(session_count[cond]):
                    respi_median = np.append(respi_median, respfeatures_allcond[cond][session_i]['cycle_freq'].values)
                resp_allsujet[sujet][cond] = np.median(respi_median)

        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))

        xr_list = {}

        #sujet = sujet_list_dfc_FR_CV[1]
        for sujet_i, sujet in enumerate(sujet_list_dfc_allcond):

            print(sujet)
                
            if monopol:
                _xr_dfc_allcond = xr.open_dataarray(f'{cf_metric}_{sujet}_stretch_rscore.nc')
            else:
                _xr_dfc_allcond = xr.open_dataarray(f'{cf_metric}_{sujet}_stretch_rscore_bi.nc')

            normalized_pairs = ['-'.join(sorted(pair.split('-'))) for pair in _xr_dfc_allcond['pair'].values]
            _xr_dfc_allcond['pair'] = normalized_pairs
            xr_list[sujet] = _xr_dfc_allcond

        pair_list = []        
        
        for _sujet in sujet_list_dfc_allcond:

            pair_list.extend(np.unique(xr_list[_sujet]['pair']))

        pair_list = np.unique(pair_list)

        params_pairs = {}

        for pair in pair_list:

            params_pairs[pair] = {}
            _sujet_list_sel = [_sujet for _sujet in sujet_list_dfc_allcond if any(xr_list[_sujet]['pair'] == pair)]
            params_pairs[pair]['sujet_list'] = _sujet_list_sel
            params_pairs[pair]['min_count'] = np.array([(xr_list[_sujet]['pair'] == pair).sum() for _sujet in _sujet_list_sel]).min()

        #### filter
        df_allpairs = pd.DataFrame()

        for pair in pair_list:

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
                _xr = _xr.expand_dims({'sujet': [_sujet]})
                _xr = _xr.rename({'pair' : 'pair_i'})
                _xr = _xr.median('cycle')
                _xr['pair_i'] = [f'{i}' for i in np.arange(params_pairs[pair]['min_count'])]

                _xr_pair.append(_xr)

            _xr_pair = xr.concat(_xr_pair, dim='sujet')
            _xr_pair = _xr_pair.expand_dims({'pair': [pair]})

            _xr_pair = _xr_pair.roll(time=int(stretch_point_FC/8))

            xr_concat_list = [_xr_pair[:,:,:,:,:,phase_i_list[phase_respi]].median('time') for phase_respi in phase_i_list]

            xr_phase = xr.concat(xr_concat_list, dim='phase')
            xr_phase = xr_phase.assign_coords(phase=['EI', 'I', 'IE', 'E', 'W'])

            df_allpairs = pd.concat([df_allpairs, xr_phase.to_dataframe(name='fc').reset_index()])

        resp_vec = []
        for row_i, row_df in df_allpairs.iterrows():

            resp_vec.append(resp_allsujet[row_df['sujet']]['FR_CV'])

        df_allpairs['resp'] = resp_vec

        os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))
        if monopol:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_ALLCOND.xlsx")
        else:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_ALLCOND_bi.xlsx")



def compilation_export_df_allplot_filtered(monopol):

    os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))

    if monopol:
        df_dict = {}
        df_dict['Cxy'] = pd.read_excel(f'df_Cxy.xlsx')
        df_dict['Pxx'] = pd.read_excel(f'df_Pxx.xlsx')
        df_dict['MI'] = pd.read_excel(f'df_MI.xlsx')
    else:
        df_dict = {}
        df_dict['Cxy'] = pd.read_excel(f'df_Cxy_bi.xlsx')
        df_dict['Pxx'] = pd.read_excel(f'df_Pxx_bi.xlsx')
        df_dict['MI'] = pd.read_excel(f'df_MI_bi.xlsx')

    #data_type = "MI"
    for data_type in ['Cxy', 'Pxx', 'MI']:

        #cond = 'ALLCOND'
        for cond in ['FR_CV', 'ALLCOND']:

            print(data_type, cond, monopol)

            if cond == 'FR_CV':
                df_export = df_dict[data_type].query(f"cond == 'FR_CV'")
            elif cond == 'ALLCOND':
                df_export = df_dict[data_type].query(f"sujet in {sujet_list}")

            if data_type == 'Cxy':

                df_export_count = df_export.query(f"cond == 'FR_CV'")

            elif data_type == 'Pxx':

                df_export_count = df_export.query(f"band == 'theta' and phase == 'whole' and cond == 'FR_CV'")

            elif data_type == 'MI':

                df_export_count = df_export.query(f"band == 'theta' and cond == 'FR_CV'")

            else:

                df_export_count = df_export.copy()

            df_export_filt_plot = pd.DataFrame()

            _thresh_plot = lmm_thresh_filt[cond]['plot']
            _thresh_sujet = lmm_thresh_filt[cond]['sujet']

            for sujet in df_export['sujet'].unique():

                _ROI_sel_sujet = df_export_count.query(f"sujet == '{sujet}'").groupby(['sujet', 'ROI']).count().query(f"cond >= {_thresh_plot}").reset_index()['ROI'].values
                _df_filt = df_export.query(f"sujet == '{sujet}' and ROI in {_ROI_sel_sujet.tolist()}")
                df_export_filt_plot = pd.concat([df_export_filt_plot, _df_filt])

            sujet_list_thresh_sujet = df_export_filt_plot[['ROI', 'sujet', data_type]].groupby(['ROI', 'sujet']).count().reset_index().groupby(['ROI']).count().query(f"sujet >= {_thresh_sujet}").reset_index()['ROI'].unique()
            df_export_filt_plot_sujet = df_export_filt_plot.query(f"ROI in {sujet_list_thresh_sujet.tolist()}")
                
            df_export_filt_plot_sujet = df_export_filt_plot_sujet[[col for col in df_export_filt_plot_sujet.columns.values if col.find('Unnamed') == -1]]

            if debug:

                df_export_filt_plot_sujet.query(f"band == 'theta' and cond == 'FR_CV'").groupby(['sujet', 'ROI']).count()

            if monopol:
                df_export_filt_plot_sujet.to_excel(f'df_{data_type}_{cond}_filt.xlsx')
            else:
                df_export_filt_plot_sujet.to_excel(f'df_{data_type}_{cond}_filt_bi.xlsx')

    









################################
######## EXECUTE ########
################################

if __name__ == '__main__':

        
    #### export df
    #monopol = False
    for monopol in [True, False]:

        # sujet_list = sujet_list_FR_CV
        # compilation_export_df_allplot(sujet_list, monopol)

        aggregate_df_Cxy(monopol)
        aggregate_df_Pxx(monopol)
        aggregate_df_MI(monopol)

    for monopol in [True, False]:
        compilation_export_df_allplot_filtered(monopol)

    
    for monopol in [True, False]:
        get_df_aggregates_fc(monopol)
    






    