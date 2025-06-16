

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import copy

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False



################################
######## PLOT ########
################################




def plot_results(monopol):
    #cf_metric = 'WPLI'
    for cf_metric in ['ISPC', 'WPLI']:

        print(cf_metric, monopol)

        band_sel = list(freq_band_dict_FC['wb'].keys())

        ######## FR_CV ########

        #### extract data
        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))
        
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

        xr_allpairs = []

        for pair in pair_list:

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
            
                _xr_pair.append(_xr)

            _xr_pair = xr.concat(_xr_pair, dim='pair').median('pair')
            xr_allpairs.append(_xr_pair.expand_dims({'pair': [pair]}))

        xr_allpairs = xr.concat(xr_allpairs, dim='pair')

        #### plot
        g = xr_allpairs.median('cycle').plot(x='time', col='band', row='pair')
        g.fig.suptitle(f"c({params_pairs[pair]['min_count']}) s({len(params_pairs[pair]['sujet_list'])})")
        g.fig.tight_layout()
        # plt.show()
        
        os.chdir(os.path.join(path_results, 'allplot', 'FC'))
        if monopol:
            g.fig.savefig(f"{cf_metric}_FR_CV_dfc.png")
        else:
            g.fig.savefig(f"{cf_metric}_FR_CV_dfc_bi.png")

        ######## ALLCOND ########

        #### extract data
        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))
        
        xr_list = {}

        #sujet = sujet_list_dfc_allcond[1]
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
            _sujet_list_sel = [_sujet for _sujet in sujet_list_dfc_allcond if any(xr_list[_sujet].loc[:,'FR_CV']['pair'] == pair)]
            params_pairs[pair]['sujet_list'] = _sujet_list_sel
            params_pairs[pair]['min_count'] = np.array([(xr_list[_sujet].loc[:,'FR_CV']['pair'] == pair).sum() for _sujet in _sujet_list_sel]).min()

        xr_allpairs = []

        for pair in pair_list:

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
            
                _xr_pair.append(_xr)

            _xr_pair = xr.concat(_xr_pair, dim='pair').median('pair')
            xr_allpairs.append(_xr_pair.expand_dims({'pair': [pair]}))

        xr_allpairs = xr.concat(xr_allpairs, dim='pair')

        #### plot
        g = xr_allpairs.median('cycle').plot(x='time', hue='cond', col='band', row='pair')
        g.fig.suptitle(f"c({params_pairs[pair]['min_count']}) s({len(params_pairs[pair]['sujet_list'])})")
        g.fig.tight_layout()
        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'FC'))
        if monopol:
            g.fig.savefig(f"{cf_metric}_ALLCOND_dfc.png")
        else:
            g.fig.savefig(f"{cf_metric}_ALLCOND_dfc_bi.png")




def plot_results_stats(monopol):

    percentile_plot_fc = [2.5, 97.5]

    #cf_metric = 'WPLI'
    for cf_metric in ['ISPC', 'WPLI']:

        print(cf_metric, monopol)

        band_sel = list(freq_band_dict_FC['wb'].keys())

        ######## FR_CV ########

        #### extract data
        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))
        
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

        #### compute stats

        xr_allpairs_stats = []

        for pair in pair_list:

            print(pair)

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
            
                _xr_pair.append(_xr.median('cycle'))

            data_obs = xr.concat(_xr_pair, dim='pair').values
            data_surr = np.zeros((n_surr_fc, len(band_sel), stretch_point_FC))

            for surr_i in range(n_surr_fc):

                print_advancement(surr_i, n_surr_fc, [25, 50, 75])

                data_shuffle = np.zeros(data_obs.shape)
            
                for band_i, _ in enumerate(band_sel):
        
                    for pair_i in range(data_shuffle.shape[0]):

                        data_shuffle[pair_i, band_i] = shuffle_Cxy(data_obs[pair_i, band_i, :])

                data_surr[surr_i] = np.median(data_shuffle, axis=0)

            thresh_dw, thresh_up = np.percentile(data_surr, percentile_plot_fc[0], axis=0), np.percentile(data_surr, percentile_plot_fc[-1], axis=0)

            if debug:

                band_i = 0
                plt.plot(np.median(data_obs[:,band_i], axis=0))
                plt.plot(thresh_dw[band_i], color='r')
                plt.plot(thresh_up[band_i], color='r')
                plt.show()

            pair_data_export = np.stack([np.median(data_obs, axis=0), thresh_dw, thresh_up], axis=0)

            xr_pair_data_export = xr.DataArray(data=pair_data_export, dims=['data_type', 'band', 'time'],
                                               coords={'data_type' : ['obs', 'dw', 'up'], 'band' : band_sel, 'time' : np.arange(stretch_point_FC)})
            
            xr_allpairs_stats.append(xr_pair_data_export.expand_dims({'pair': [pair]}))

        xr_allpairs_stats = xr.concat(xr_allpairs_stats, dim='pair')

        #### plot
        
        fig, axs = plt.subplots(nrows=len(pair_list), ncols=len(band_sel), figsize=(15,15))
        
        for row_i, pair in enumerate(pair_list):

            for col_i, band in enumerate(band_sel):

                ax = axs[row_i, col_i]
                xr_allpairs_stats.loc[pair, 'obs', band].plot(x='time', ax=ax)
                xr_allpairs_stats.loc[pair, 'dw', band].plot(x='time', ax=ax, color='r', linestyle='--')
                xr_allpairs_stats.loc[pair, 'up', band].plot(x='time', ax=ax, color='r', linestyle='--')

                if row_i == 0:
                    ax.set_title(band)   
                if col_i == 0:
                    ax.set_ylabel(pair)  
                if row_i != 0 and col_i == 0:
                    ax.set_ylabel(pair)  
                    ax.set_title("") 
                if row_i != 0 and col_i != 0:
                    ax.set_title("")   

        plt.suptitle(f"c({params_pairs[pair]['min_count']}) s({len(params_pairs[pair]['sujet_list'])})")
        plt.tight_layout()
        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'FC'))
        if monopol:
            fig.savefig(f"STATS_{cf_metric}_FR_CV_dfc.png")
        else:
            fig.savefig(f"STATS_{cf_metric}_FR_CV_dfc_bi.png")

        plt.close('all')

        ######## ALLCOND ########

        #### extract data
        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))
        
        xr_list = {}

        #sujet = sujet_list_dfc_allcond[1]
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
            _sujet_list_sel = [_sujet for _sujet in sujet_list_dfc_allcond if any(xr_list[_sujet].loc[:,'FR_CV']['pair'] == pair)]
            params_pairs[pair]['sujet_list'] = _sujet_list_sel
            params_pairs[pair]['min_count'] = np.array([(xr_list[_sujet].loc[:,'FR_CV']['pair'] == pair).sum() for _sujet in _sujet_list_sel]).min()

        #### compute stats
        xr_allpairs_stats = []

        for pair in pair_list:

            print(pair)

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
            
                _xr_pair.append(_xr.median('cycle'))

            data_obs = xr.concat(_xr_pair, dim='pair').values
            data_surr = np.zeros((n_surr_fc, len(conditions), len(band_sel), stretch_point_FC))

            for surr_i in range(n_surr_fc):

                print_advancement(surr_i, n_surr_fc, [25, 50, 75])

                data_shuffle = np.zeros(data_obs.shape)

                for cond_i, _ in enumerate(conditions):
            
                    for band_i, _ in enumerate(band_sel):
            
                        for pair_i in range(data_shuffle.shape[0]):

                            data_shuffle[pair_i, cond_i, band_i] = shuffle_Cxy(data_obs[pair_i, cond_i ,band_i, :])

                data_surr[surr_i] = np.median(data_shuffle, axis=0)

            thresh_dw, thresh_up = np.percentile(data_surr, percentile_plot_fc[0], axis=0), np.percentile(data_surr, percentile_plot_fc[-1], axis=0)

            if debug:

                cond_i = 1
                band_i = 0
                plt.plot(np.median(data_obs[:,cond_i,band_i], axis=0))
                plt.plot(thresh_dw[cond_i,band_i], color='r')
                plt.plot(thresh_up[cond_i,band_i], color='r')
                plt.show()

            pair_data_export = np.stack([np.median(data_obs, axis=0), thresh_dw, thresh_up], axis=0)

            xr_pair_data_export = xr.DataArray(data=pair_data_export, dims=['data_type', 'cond', 'band', 'time'],
                                               coords={'data_type' : ['obs', 'dw', 'up'], 'cond' : conditions, 'band' : band_sel, 'time' : np.arange(stretch_point_FC)})
            
            xr_allpairs_stats.append(xr_pair_data_export.expand_dims({'pair': [pair]}))

        xr_allpairs_stats = xr.concat(xr_allpairs_stats, dim='pair')

        #### plot

        for band in band_sel:
        
            fig, axs = plt.subplots(nrows=len(pair_list), ncols=len(conditions), figsize=(15,15))
            
            for row_i, pair in enumerate(pair_list):

                for col_i, cond in enumerate(conditions):

                    ax = axs[row_i, col_i]
                    xr_allpairs_stats.loc[pair, 'obs', cond, band].plot(x='time', ax=ax)
                    xr_allpairs_stats.loc[pair, 'dw', cond, band].plot(x='time', ax=ax, color='r', linestyle='--')
                    xr_allpairs_stats.loc[pair, 'up', cond, band].plot(x='time', ax=ax, color='r', linestyle='--')

                    if row_i == 0:
                        ax.set_title(cond)   
                    if col_i == 0:
                        ax.set_ylabel(pair)  
                    if row_i != 0 and col_i == 0:
                        ax.set_ylabel(pair)  
                        ax.set_title("") 
                    if row_i != 0 and col_i != 0:
                        ax.set_title("")   

            plt.suptitle(f"{band} c({params_pairs[pair]['min_count']}) s({len(params_pairs[pair]['sujet_list'])})")
            plt.tight_layout()
            # plt.show()

            os.chdir(os.path.join(path_results, 'allplot', 'FC'))
            if monopol:
                fig.savefig(f"STATS_{band}_{cf_metric}_ALLCOND_dfc.png")
            else:
                fig.savefig(f"STATS_{band}_{cf_metric}_ALLCOND_dfc_bi.png")

            plt.close('all')





################################
######## PLOT ########
################################




def plot_results(monopol):
    #cf_metric = 'WPLI'
    for cf_metric in ['ISPC', 'WPLI']:

        print(cf_metric, monopol)

        band_sel = list(freq_band_dict_FC['wb'].keys())

        ######## FR_CV ########

        #### extract data
        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))
        
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

        xr_allpairs = []

        for pair in pair_list:

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
            
                _xr_pair.append(_xr)

            _xr_pair = xr.concat(_xr_pair, dim='pair').median('pair')
            xr_allpairs.append(_xr_pair.expand_dims({'pair': [pair]}))

        xr_allpairs = xr.concat(xr_allpairs, dim='pair')

        #### plot
        g = xr_allpairs.median('cycle').plot(x='time', col='band', row='pair')
        g.fig.suptitle(f"c({params_pairs[pair]['min_count']}) s({len(params_pairs[pair]['sujet_list'])})")
        g.fig.tight_layout()
        # plt.show()
        
        os.chdir(os.path.join(path_results, 'allplot', 'FC'))
        if monopol:
            g.fig.savefig(f"{cf_metric}_FR_CV_dfc.png")
        else:
            g.fig.savefig(f"{cf_metric}_FR_CV_dfc_bi.png")

        ######## ALLCOND ########

        #### extract data
        os.chdir(os.path.join(path_precompute, 'allplot', 'FC'))
        
        xr_list = {}

        #sujet = sujet_list_dfc_allcond[1]
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
            _sujet_list_sel = [_sujet for _sujet in sujet_list_dfc_allcond if any(xr_list[_sujet].loc[:,'FR_CV']['pair'] == pair)]
            params_pairs[pair]['sujet_list'] = _sujet_list_sel
            params_pairs[pair]['min_count'] = np.array([(xr_list[_sujet].loc[:,'FR_CV']['pair'] == pair).sum() for _sujet in _sujet_list_sel]).min()

        xr_allpairs = []

        for pair in pair_list:

            _xr_pair = []
                        
            for _sujet in params_pairs[pair]['sujet_list']:

                _xr = xr_list[_sujet].loc[pair][:params_pairs[pair]['min_count']]
            
                _xr_pair.append(_xr)

            _xr_pair = xr.concat(_xr_pair, dim='pair').median('pair')
            xr_allpairs.append(_xr_pair.expand_dims({'pair': [pair]}))

        xr_allpairs = xr.concat(xr_allpairs, dim='pair')

        #### plot
        g = xr_allpairs.median('cycle').plot(x='time', hue='cond', col='band', row='pair')
        g.fig.suptitle(f"c({params_pairs[pair]['min_count']}) s({len(params_pairs[pair]['sujet_list'])})")
        g.fig.tight_layout()
        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'FC'))
        if monopol:
            g.fig.savefig(f"{cf_metric}_ALLCOND_dfc.png")
        else:
            g.fig.savefig(f"{cf_metric}_ALLCOND_dfc_bi.png")








################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #monopol = True
    for monopol in [True, False]:

        # plot_results(monopol)
        plot_results_stats(monopol)









