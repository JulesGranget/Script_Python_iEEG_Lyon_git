

import os
import pandas as pd

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False




########################################
######## COMPILATION FUNCTION ########
########################################




def compilation_export_df_allplot(sujet_list):

    #### generate df
    df_export_TF = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'band', 'phase', 'Pxx'])
    df_export_Cxy_MVL = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'Cxy', 'Cxy_surr', 'MVL', 'MVL_surr'])
    df_export_DFC = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'CPL', 'GE', 'SWN'])
    df_export_FC = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'CPL', 'GE', 'SWN'])
    df_export_HRV = pd.DataFrame(columns=['sujet', 'cond', 'session', 'RDorFR', 'compute_type', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_S'])

    #### fill
    for sujet in sujet_list:

        print(sujet)

        os.chdir(os.path.join(path_results, sujet, 'df'))
        df_export_TF_i = pd.read_excel(f'{sujet}_df_TF_IE.xlsx')
        df_export_Cxy_MVL_i = pd.read_excel(f'{sujet}_df_Cxy_MVL.xlsx')
        df_export_DFC_i = pd.read_excel(f'{sujet}_df_DFC.xlsx')
        df_export_FC_i = pd.read_excel(f'{sujet}_df_FC.xlsx')
        df_export_HRV_i = pd.read_excel(f'{sujet}_df_HRV.xlsx')

        df_export_TF = pd.concat([df_export_TF, df_export_TF_i])
        df_export_Cxy_MVL = pd.concat([df_export_Cxy_MVL, df_export_Cxy_MVL_i])
        df_export_DFC = pd.concat([df_export_DFC, df_export_DFC_i])
        df_export_FC = pd.concat([df_export_FC, df_export_FC_i])
        df_export_HRV = pd.concat([df_export_HRV, df_export_HRV_i])

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'df'))

    if os.path.exists('allplot_df_TF_IE.xlsx'):
        print('TF_IE : ALREADY COMPUTED')
    else:
        df_export_TF.to_excel('allplot_df_TF_IE.xlsx')
    
    if os.path.exists('allplot_df_Cxy_MVL.xlsx'):
        print('Cxy_MVL : ALREADY COMPUTED')
    else:
        df_export_Cxy_MVL.to_excel('allplot_df_Cxy_MVL.xlsx')
    
    if os.path.exists('allplot_df_DFC.xlsx'):
        print('DFC : ALREADY COMPUTED')
    else:
        df_export_DFC.to_excel('allplot_df_DFC.xlsx')
    
    if os.path.exists('allplot_df_FC.xlsx'):
        print('FC : ALREADY COMPUTED')
    else:
        df_export_FC.to_excel('allplot_df_FC.xlsx')

    if os.path.exists('allplot_df_HRV.xlsx'):
        print('FC : ALREADY COMPUTED')
    else:
        df_export_HRV.to_excel('allplot_df_HRV.xlsx')
    
    


    





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

        
    #### export df
    sujet_list = sujet_list_FR_CV
    compilation_export_df_allplot(sujet_list)
    
    