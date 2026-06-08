

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import physio
import seaborn as sns
from statannotations.Annotator import Annotator

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False






############################
######## EXECUTE ########
############################



if __name__ == '__main__':

    #### get respi value
    df_respfeatures_allsujet = pd.DataFrame()

    for sujet in sujet_list:

        respfeatures = load_respfeatures(sujet)

        for cond in conditions:
        
            for cond_i in range(session_count[cond]):

                _respfeatures = respfeatures[cond][cond_i].query(f"select == 1")
                _respfeatures['cond'] = [f"{cond}_{cond_i+1}"] * _respfeatures.shape[0]
                _respfeatures['sujet'] = [f"{sujet}"] * _respfeatures.shape[0]
                df_respfeatures_allsujet = pd.concat([df_respfeatures_allsujet, _respfeatures]).drop(columns=['Unnamed: 0'])

    respfeatures_metric_list = ['cycle_duration', 'cycle_freq', 'total_amplitude', 'total_volume', 'select', 'cond', 'sujet']


    df_respfeatures_allsujet_compact = pd.DataFrame()

    for sujet in sujet_list:

        respfeatures = load_respfeatures(sujet)

        for cond in conditions:
        
            for cond_i in range(session_count[cond]):

                _respfeatures = respfeatures[cond][cond_i].query(f"select == 1")
                _respfeatures['cond'] = [f"{cond}"] * _respfeatures.shape[0]
                _respfeatures['sujet'] = [f"{sujet}"] * _respfeatures.shape[0]
                df_respfeatures_allsujet_compact = pd.concat([df_respfeatures_allsujet_compact, _respfeatures]).drop(columns=['Unnamed: 0'])

    respfeatures_metric_list = ['cycle_duration', 'cycle_freq', 'total_amplitude', 'total_volume', 'select', 'cond', 'sujet']

    ################################################
    ######## ALLSUJET VARIABILITY FR_CV ########
    ################################################

    #### get respi value
    dict_df = {'sujet' : [], 'cycle_freq_med' : []}

    for sujet in sujet_list_FR_CV:

        respfeatures = load_respfeatures(sujet)

        _respfeatures = respfeatures['FR_CV'][0].query(f"select == 1")
        dict_df['sujet'].append(sujet)
        dict_df['cycle_freq_med'].append(_respfeatures['cycle_freq'].median())

    df_respfeatures_allsujet_FR_CV = pd.DataFrame(dict_df)

    fig, ax = plt.subplots()
    ax.hist(df_respfeatures_allsujet_FR_CV['cycle_freq_med'].values, bins=25)
    ax.set_xlabel('cycle_freq')
    ax.set_ylabel('count')
    plt.suptitle('FR_CV variability')
    # plt.show()

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'plot'))

    fig.savefig(f"allsujet_FR_CV_cycle_freq_med.png")

    ################################
    ######## RD_CV ########
    ################################

    var_stats = {'sujet' : [], 'cond' : [], 'signi' : [], 'pval' : []}

    for sujet in sujet_list:

        baseline_val = df_respfeatures_allsujet_compact.query(f"cond == 'FR_CV' and sujet == '{sujet}'")['cycle_freq'].values

        for cond in ['RD_CV', 'RD_SV', 'RD_FV']:

            cond_val = df_respfeatures_allsujet_compact.query(f"cond == '{cond}' and sujet == '{sujet}'")['cycle_freq'].values
            stat, p = scipy.stats.levene(baseline_val, cond_val)

            var_stats['sujet'].append(sujet)
            var_stats['cond'].append(cond)
            var_stats['signi'].append(p < 0.05)
            var_stats['pval'].append(p)

    df_var_stats = pd.DataFrame(var_stats)
    df_var_stats['cond_baseline'] = ['FR_CV'] * df_var_stats.shape[0]

    df_plot = df_respfeatures_allsujet_compact
    metric = 'cycle_freq'

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='sujet', y=metric, hue='cond', showfliers=False, ax=ax)
    plt.title(metric)

    pairs = []
    for sujet in df_plot['sujet'].unique():
        pairs += [
            ((sujet, 'FR_CV'), (sujet, 'RD_CV')),
            ((sujet, 'FR_CV'), (sujet, 'RD_SV')),
            ((sujet, 'FR_CV'), (sujet, 'RD_FV')),
        ]

    pvals = []
    for pair in pairs:
        sujet, c1 = pair[0]
        _,    c2 = pair[1]
        row = df_var_stats.query("sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))")
        pvals.append(row['pval'].iat[0] if not row.empty else 1.0)

    filtered_pairs = []
    filtered_pvals = []

    # Loop through your pairs and keep only significant ones
    for pair in pairs:
        sujet, c1 = pair[0]
        _,    c2 = pair[1]
        row = df_var_stats.query(
            "sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))"
        )
        if not row.empty:
            pval = row['pval'].iat[0]
            if pval < 0.05: 
                filtered_pairs.append(pair)
                filtered_pvals.append(pval)


    annot = Annotator(
        ax, pairs=filtered_pairs, data=df_plot,
        x='sujet', y=metric, hue='cond'
    )
    annot.configure(test=None, text_format='star', loc='inside')
    annot.set_pvalues_and_annotate(filtered_pvals)

    ax.set_title(metric)
    plt.tight_layout()

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'plot'))

    # plt.show()

    fig.savefig(f"allsujet_VARIANCE_RD_CV_{metric}.png")

    plt.close('all')






    ################################
    ######## RD_CV ########
    ################################

    
    for metric in respfeatures_metric_list:

        df_RD_CV_resp = pd.DataFrame()

        for sujet in sujet_list:

            print(sujet)

            for cond_i in range(session_count['RD_CV']):

                df_stats = df_respfeatures_allsujet.query(f"cond in ['FR_CV_1', 'RD_CV_{cond_i+1}'] and sujet == '{sujet}'")

                n_cycle_baseline = df_stats.query(f"cond == 'FR_CV_1'").shape[0]
                n_cycle_cond = df_stats.query(f"cond == 'RD_CV_{cond_i+1}'").shape[0]
                n_cycle_sel = np.array([n_cycle_baseline, n_cycle_cond]).min()

                n_surr_respi = 250
                p_surr = []

                for surr_i in range(n_surr_respi):

                    baseline_sel_i = np.random.choice(np.arange(n_cycle_baseline), size=n_cycle_sel, replace=False)
                    cond_sel_i = np.random.choice(np.arange(n_cycle_cond), size=n_cycle_sel, replace=False)
                    df_baseline = df_stats.query(f"cond == 'FR_CV_1'").iloc[baseline_sel_i,:]
                    df_cond = df_stats.query(f"cond == 'RD_CV_{cond_i+1}'").iloc[cond_sel_i,:]

                    stat, p = scipy.stats.wilcoxon(df_baseline[metric].values, 
                                                df_cond[metric].values)
                    
                    p_surr.append(p)

                p_perm = np.percentile(np.array(p_surr), 99)

                df_RD_CV_resp = pd.concat([df_RD_CV_resp, pd.DataFrame({'sujet' : [sujet], 'cond' : [f'RD_CV_{cond_i+1}'], 'p_perm' : [p_perm]})])

        df_RD_CV_resp['cond_baseline'] = ['FR_CV_1'] * df_RD_CV_resp.shape[0]

        # Filter your data
        df_plot = df_respfeatures_allsujet.query("cond in ['FR_CV_1', 'RD_CV_1', 'RD_CV_2']")

        # Create base plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_plot, x='sujet', y=metric, hue='cond', showfliers=False, ax=ax)
        plt.title(metric)

        pairs = []
        for sujet in df_plot['sujet'].unique():
            pairs += [
                ((sujet, 'FR_CV_1'), (sujet, 'RD_CV_1')),
                ((sujet, 'FR_CV_1'), (sujet, 'RD_CV_2')),
                ((sujet, 'RD_CV_1'), (sujet, 'RD_CV_2'))
            ]

        pvals = []
        for pair in pairs:
            sujet, c1 = pair[0]
            _,    c2 = pair[1]
            row = df_RD_CV_resp.query("sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))")
            pvals.append(row['p_perm'].iat[0] if not row.empty else 1.0)

        filtered_pairs = []
        filtered_pvals = []

        # Loop through your pairs and keep only significant ones
        for pair in pairs:
            sujet, c1 = pair[0]
            _,    c2 = pair[1]
            row = df_RD_CV_resp.query(
                "sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))"
            )
            if not row.empty:
                pval = row['p_perm'].iat[0]
                if pval < 0.05: 
                    filtered_pairs.append(pair)
                    filtered_pvals.append(pval)


        annot = Annotator(
            ax, pairs=filtered_pairs, data=df_plot,
            x='sujet', y=metric, hue='cond'
        )
        annot.configure(test=None, text_format='star', loc='inside')
        annot.set_pvalues_and_annotate(filtered_pvals)

        ax.set_title(metric)
        plt.tight_layout()

        os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'plot'))

        # plt.show()

        fig.savefig(f"allsujet_RD_CV_{metric}.png")

        plt.close('all')


    ################################
    ######## ALLCOND ########
    ################################

    df_respfeatures_allsujet_compact_ano = df_respfeatures_allsujet_compact.copy()
    df_respfeatures_allsujet_compact_ano['sujet'] = df_respfeatures_allsujet_compact['sujet'].map(mapping_sujet_paper)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_respfeatures_allsujet_compact_ano.query(f"cond != 'FR_CV'"), x='sujet', y='cycle_freq', hue='cond', hue_order=['RD_SV', 'RD_CV', 'RD_FV'],
                showfliers=False, ax=ax, palette=['tab:red', 'tab:blue', 'tab:green'])

    ax.set_title('respi_allsujet')
    plt.tight_layout()

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'plot'))

    # plt.show()

    fig.savefig(f"allsujet_cycle_freq.png")

    plt.close('all')



    df_ALLCOND_resp = pd.DataFrame()
    df_diff = pd.DataFrame()

    for sujet in sujet_list:

        print(sujet)

        ref_respi = {'RD_CV' : df_respfeatures_allsujet_compact.query(f"cond == 'FR_CV' and sujet == '{sujet}'")['cycle_freq'].median(), 
                        'RD_SV' : 0.15, 'RD_FV' : 0.5}

        for cond in ['RD_CV', 'RD_SV', 'RD_FV']:

            data_stats = df_respfeatures_allsujet_compact.query(f"cond == '{cond}' and sujet == '{sujet}'")['cycle_freq'].values

            diff = [x - ref_respi[cond] for x in data_stats]

            stat, p = scipy.stats.wilcoxon(diff)

            if debug:

                count, _, _ = plt.hist(data_stats, bins=50)
                plt.vlines([ref_respi[cond]], ymin=0, ymax=count.max(), color='r')
                plt.show()

            df_ALLCOND_resp = pd.concat([df_ALLCOND_resp, pd.DataFrame({'sujet' : [sujet], 'cond' : [f'{cond}'], 'p' : [p]})])
            df_diff = pd.concat([df_diff, pd.DataFrame({'sujet' : [sujet] * len(diff), 'cond' : [f'{cond}'] * len(diff), 'cycle_freq' : diff})])

    df_ALLCOND_resp['cond_baseline'] = ['REF'] * df_ALLCOND_resp.shape[0]

    for sujet in sujet_list:

        df_diff = pd.concat([df_diff, pd.DataFrame({'sujet' : [sujet], 'cond' : [f'REF'], 'cycle_freq' : [0]})])

    df_plot = df_diff

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='sujet', y='cycle_freq', hue='cond', showfliers=False, ax=ax)
    plt.title(metric)

    pairs = []
    for sujet in df_plot['sujet'].unique():
        pairs += [
            ((sujet, 'REF'), (sujet, 'RD_FV')),
            ((sujet, 'REF'), (sujet, 'RD_SV')),
            ((sujet, 'REF'), (sujet, 'RD_CV'))
        ]

    pvals = []
    for pair in pairs:
        sujet, c1 = pair[0]
        _,    c2 = pair[1]
        row = df_ALLCOND_resp.query("sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))")
        pvals.append(row['p'].iat[0] if not row.empty else 1.0)

    filtered_pairs = []
    filtered_pvals = []

    # Loop through your pairs and keep only significant ones
    for pair in pairs:
        sujet, c1 = pair[0]
        _,    c2 = pair[1]
        row = df_ALLCOND_resp.query(
            "sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))"
        )
        if not row.empty:
            pval = row['p'].iat[0]
            if pval < 0.05: 
                filtered_pairs.append(pair)
                filtered_pvals.append(pval)


    annot = Annotator(
        ax, pairs=filtered_pairs, data=df_plot,
        x='sujet', y='cycle_freq', hue='cond'
    )
    annot.configure(test=None, text_format='star', loc='inside')
    annot.set_pvalues_and_annotate(filtered_pvals)

    ax.set_title('respi_follow')
    plt.tight_layout()

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'plot'))

    # plt.show()

    fig.savefig(f"allsujet_respi_follow.png")

    plt.close('all')


    #### with outliers
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='sujet', y='cycle_freq', hue='cond', ax=ax)
    plt.title(metric)

    pairs = []
    for sujet in df_plot['sujet'].unique():
        pairs += [
            ((sujet, 'REF'), (sujet, 'RD_FV')),
            ((sujet, 'REF'), (sujet, 'RD_SV')),
            ((sujet, 'REF'), (sujet, 'RD_CV'))
        ]

    pvals = []
    for pair in pairs:
        sujet, c1 = pair[0]
        _,    c2 = pair[1]
        row = df_ALLCOND_resp.query("sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))")
        pvals.append(row['p'].iat[0] if not row.empty else 1.0)

    filtered_pairs = []
    filtered_pvals = []

    # Loop through your pairs and keep only significant ones
    for pair in pairs:
        sujet, c1 = pair[0]
        _,    c2 = pair[1]
        row = df_ALLCOND_resp.query(
            "sujet == @sujet and ((cond == @c1 and cond_baseline == @c2) or (cond == @c2 and cond_baseline == @c1))"
        )
        if not row.empty:
            pval = row['p'].iat[0]
            if pval < 0.05: 
                filtered_pairs.append(pair)
                filtered_pvals.append(pval)


    annot = Annotator(
        ax, pairs=filtered_pairs, data=df_plot,
        x='sujet', y='cycle_freq', hue='cond'
    )
    annot.configure(test=None, text_format='star', loc='inside')
    annot.set_pvalues_and_annotate(filtered_pvals)

    ax.set_title('respi_follow')
    plt.tight_layout()

    plt.show()






    