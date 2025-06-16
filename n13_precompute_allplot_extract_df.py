

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
                        'IE' : np.arange(stretch_point_FC/2, stretch_point_FC*3/4, dtype='int'), 'E' : np.arange(stretch_point_FC*3/4, stretch_point_FC, dtype='int')}

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
            xr_phase = xr_phase.assign_coords(phase=['EI', 'I', 'IE', 'E'])

            df_allpairs = pd.concat([df_allpairs, xr_phase.to_dataframe(name='fc').reset_index()])

        os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))
        if monopol:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_FR_CV.xlsx")
        else:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_FR_CV_bi.xlsx")


        ######## ALLCOND ########

        print('ALLCOND')

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
            xr_phase = xr_phase.assign_coords(phase=['EI', 'I', 'IE', 'E'])

            df_allpairs = pd.concat([df_allpairs, xr_phase.to_dataframe(name='fc').reset_index()])

        os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))
        if monopol:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_ALLCOND.xlsx")
        else:
            df_allpairs.to_excel(f"df_{cf_metric}_FC_ALLCOND_bi.xlsx")



def compilation_export_df_allplot_filtered(monopol):

    os.chdir(os.path.join(path_results, 'allplot', 'df', 'df_aggregates'))

    if monopol:
        df_Cxy = pd.read_excel(f'df_Cxy.xlsx')
        df_Pxx = pd.read_excel(f'df_Pxx.xlsx')
    else:
        df_Cxy = pd.read_excel(f'df_Cxy_bi.xlsx')
        df_Pxx = pd.read_excel(f'df_Pxx_bi.xlsx')

    #### Cxy FR_CV
    df_Cxy_FR_CV = df_Cxy.query(f"cond == 'FR_CV'")
    
    sujet_list_thresh_sujet = df_Cxy_FR_CV[['sujet', 'ROI', 'Cxy']].groupby(['ROI', 'sujet']).count().reset_index().groupby(['ROI']).count().query(f"sujet >= {thresh_sujet_FR_CV}").reset_index()['ROI'].unique()
    df_Cxy_FR_CV_filt_sujet = df_Cxy_FR_CV.query(f"ROI in {sujet_list_thresh_sujet.tolist()}")
    
    df_Cxy_FR_CV_filt_sujet = df_Cxy_FR_CV_filt_sujet[[col for col in df_Cxy_FR_CV_filt_sujet.columns.values if col.find('Unnamed') == -1]]

    if monopol:
        df_Cxy_FR_CV_filt_sujet.to_excel('df_Cxy_FR_CV_filt.xlsx')
    else:
        df_Cxy_FR_CV_filt_sujet.to_excel('df_Cxy_FR_CV_filt_bi.xlsx')

    #### Cxy ALLCOND
    df_Cxy_ALLCOND = df_Cxy.query(f"sujet in {sujet_list}")

    sujet_list_thresh_sujet = df_Cxy_ALLCOND[['ROI', 'sujet', 'Cxy']].groupby(['ROI', 'sujet']).count().reset_index().groupby(['ROI']).count().query(f"sujet >= {thresh_sujet_ALLCOND}").reset_index()['ROI'].unique()
    df_Cxy_ALLCOND_filt_sujet = df_Cxy_ALLCOND.query(f"ROI in {sujet_list_thresh_sujet.tolist()}")

    df_count_plot = df_Cxy_ALLCOND_filt_sujet.groupby(['sujet', 'ROI']).count().query(f"cond >= {thresh_plot_ALLCOND}").query(f"cond >= {thresh_plot_ALLCOND}").reset_index()
    df_Cxy_ALLCOND_filt_plot_sujet = pd.DataFrame()

    for sujet in df_count_plot['sujet'].unique():

        _ROI_sel_sujet = df_count_plot.query(f"sujet == '{sujet}'")['ROI'].values
        _df_Cxy_filt = df_Cxy_ALLCOND_filt_sujet.query(f"sujet == '{sujet}' and ROI in {_ROI_sel_sujet.tolist()}")
        df_Cxy_ALLCOND_filt_plot_sujet = pd.concat([df_Cxy_ALLCOND_filt_plot_sujet, _df_Cxy_filt])
        
    df_Cxy_ALLCOND_filt_plot_sujet = df_Cxy_ALLCOND_filt_plot_sujet[[col for col in df_Cxy_ALLCOND_filt_plot_sujet.columns.values if col.find('Unnamed') == -1]]

    if monopol:
        df_Cxy_ALLCOND_filt_plot_sujet.to_excel('df_Cxy_ALLCOND_filt.xlsx')
    else:
        df_Cxy_ALLCOND_filt_plot_sujet.to_excel('df_Cxy_ALLCOND_filt_bi.xlsx')

    #### Pxx FR_CV
    df_Pxx_FR_CV = df_Pxx.query(f"cond == 'FR_CV'")

    sujet_list_thresh_sujet = df_Pxx_FR_CV[['ROI', 'sujet', 'Pxx']].groupby(['ROI', 'sujet']).count().reset_index().groupby(['ROI']).count().query(f"sujet >= {thresh_sujet_FR_CV}").reset_index()['ROI'].unique()
    df_Pxx_FR_CV_filt_sujet = df_Pxx_FR_CV.query(f"ROI in {sujet_list_thresh_sujet.tolist()}")
    
    df_count_plot = df_Pxx_FR_CV_filt_sujet.groupby(['sujet', 'ROI', 'chan']).count().groupby(['sujet', 'ROI']).count().query(f"cond >= {thresh_plot_ALLCOND}").query(f"cond >= {thresh_plot_ALLCOND}").reset_index()
    df_Pxx_FR_CV_filt_sujet_plot = pd.DataFrame()

    for sujet in df_count_plot['sujet'].unique():

        _ROI_sel_sujet = df_count_plot.query(f"sujet == '{sujet}'")['ROI'].values
        _df_Pxx_FR_CV_filt = df_Pxx_FR_CV.query(f"sujet == '{sujet}' and ROI in {_ROI_sel_sujet.tolist()}")
        df_Pxx_FR_CV_filt_sujet_plot = pd.concat([df_Pxx_FR_CV_filt_sujet_plot, _df_Pxx_FR_CV_filt])

    df_Pxx_FR_CV_filt_sujet_plot = df_Pxx_FR_CV_filt_sujet_plot[[col for col in df_Pxx_FR_CV_filt_sujet_plot.columns.values if col.find('Unnamed') == -1]]

    if monopol:
        df_Pxx_FR_CV_filt_sujet_plot.to_excel('df_Pxx_FR_CV_filt.xlsx')
    else:
        df_Pxx_FR_CV_filt_sujet_plot.to_excel('df_Pxx_FR_CV_filt_bi.xlsx')

    #### Pxx ALLCOND
    df_Pxx_ALLCOND = df_Pxx.query(f"sujet in {sujet_list}")

    sujet_list_thresh_sujet = df_Pxx_ALLCOND[['ROI', 'sujet', 'Pxx']].groupby(['ROI', 'sujet']).count().reset_index().groupby(['ROI']).count().query(f"sujet >= {thresh_sujet_ALLCOND}").reset_index()['ROI'].unique()
    df_Pxx_FR_CV_filt_sujet = df_Pxx_ALLCOND.query(f"ROI in {sujet_list_thresh_sujet.tolist()}")

    df_count_plot = df_Pxx_FR_CV_filt_sujet.groupby(['sujet', 'ROI', 'chan']).count().groupby(['sujet', 'ROI']).count().query(f"cond >= {thresh_plot_ALLCOND}").query(f"cond >= {thresh_plot_ALLCOND}").reset_index()
    df_Pxx_ALLCOND_filt_sujet_plot = pd.DataFrame()

    for sujet in df_count_plot['sujet'].unique():

        _ROI_sel_sujet = df_count_plot.query(f"sujet == '{sujet}'")['ROI'].values
        _df_Pxx_ALLCOND_filt = df_Pxx_FR_CV_filt_sujet.query(f"sujet == '{sujet}' and ROI in {_ROI_sel_sujet.tolist()}")
        df_Pxx_ALLCOND_filt_sujet_plot = pd.concat([df_Pxx_ALLCOND_filt_sujet_plot, _df_Pxx_ALLCOND_filt])

    df_Pxx_ALLCOND_filt_sujet_plot = df_Pxx_ALLCOND_filt_sujet_plot[[col for col in df_Pxx_ALLCOND_filt_sujet_plot.columns.values if col.find('Unnamed') == -1]]

    if monopol:
        df_Pxx_ALLCOND_filt_sujet_plot.to_excel('df_Pxx_ALLCOND_filt.xlsx')
    else:
        df_Pxx_ALLCOND_filt_sujet_plot.to_excel('df_Pxx_ALLCOND_filt_bi.xlsx')









################################
######## EXECUTE ########
################################

if __name__ == '__main__':

        
    #### export df
    #monopol = True
    for monopol in [True, False]:

        # sujet_list = sujet_list_FR_CV
        # compilation_export_df_allplot(sujet_list, monopol)

        aggregate_df_Cxy(monopol)
        aggregate_df_Pxx(monopol)

    for monopol in [True, False]:
        compilation_export_df_allplot_filtered(monopol)

    
    for monopol in [True, False]:

        get_df_aggregates_fc(monopol)
    






    