# Required libraries
library(readxl)
library(lme4)
library(lmerTest)  # For p-values
library(emmeans)
library(ggplot2)
library(e1071)  # For skewness and kurtosis
library(pbkrtest)
library(sjPlot)
library(broom.mixed)
library(writexl)
library(dplyr)


################
#### FC ####
################

######## FR_CV ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_WPLI_FC_FR_CV.xlsx", sep  = "/"))

pair_list <- unique(df_raw$pair)
band_list <- unique(df_raw$band)
phase_list <- unique(df_raw$phase)

pair_sel = pair_list[2]
band_sel = band_list[2]

for (band_sel in band_list) { 
  
  for (pair_sel in pair_list) {
    
    withCallingHandlers({
    
    print(band_sel)
    print(pair_sel)
    
    df_onepair <- subset(df_raw, band == band_sel & pair == pair_sel)
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_onepair[c("sujet", "phase", "pair_i", "fc")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$phase <- as.factor(df$phase)
    df$pair_ <- as.factor(df$pair_i)
    
    df$phase <- relevel(df$phase, ref = "I")
    
    #### FIG BOXPLOT
    p <- ggplot(df, aes(x = sujet, y = fc, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste(pair_sel, "fc", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    # NAME
    file_boxplot_subjectwise = paste("boxplot", pair_sel, band_sel, "fc_WPLI_subjectwise_FR_CV.png", sep = "_")
    
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### model
    complex_form <- fc ~ phase + (phase | sujet/pair_i)
    simple_form  <- fc ~ phase + (1 | sujet/pair_i)
    
    model <- tryCatch({
      mod_attempt <- lmer(
        simple_form,
        data = df,
        control = lmerControl(optCtrl = list(maxfun = 2e5))
      )
      
      # Check for convergence issue
      if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
          isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️ lmer() model failed or was singular → switching to lm().")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return the valid lmer model
      
    }, error = function(e) {
      # On error or forced fallback: use lm instead
      lm(fc ~ phase, data = df)
    })
    
    summary(model)
    
    #### FIG HIST
    # NAME
    filename_hist = paste("histogram", pair_sel, band_sel, "fc_FR_CV.png", sep = "_")
    
    skew_chan = round(skewness(df$fc), 2)
    kurt_chan = round(kurtosis(df$fc), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$fc,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "fc values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste(pair_sel, "fc", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG QQPLOT
    # NAME
    filename_qqplot = paste("qqplot", pair_sel, band_sel, "fc_FR_CV.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, "qqplot_bi"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### mixed model df
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    # NAME
    filesxlsx_ROI = paste("lmm", pair_sel, band_sel, "res_FC_FR_CV.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
    
  }
}



######## ALLCOND ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_WPLI_FC_ALLCOND.xlsx", sep  = "/"))

pair_list <- unique(df_raw$pair)
band_list <- unique(df_raw$band)
phase_list <- unique(df_raw$phase)

pair_sel = pair_list[2]
band_sel = band_list[2]

for (band_sel in band_list) { 
  
  for (pair_sel in pair_list) {
    
    withCallingHandlers({
    
    print(band_sel)
    print(pair_sel)
    
    df_onepair <- subset(df_raw, band == band_sel & pair == pair_sel)
    df_onepair$pair_i <- paste(df_onepair$sujet, df_onepair$pair_i, sep = "_")
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_onepair[c("sujet", "phase", "pair_i", "fc", "cond")]
    
    # Convert categorical variables to factors
    #df$sujet <- as.factor(df$sujet)
    df$phase <- as.factor(df$phase)
    df$pair_i <- as.factor(df$pair_i)
    df$cond <- as.factor(df$cond)
    
    df$phase <- relevel(df$phase, ref = "I")
    
    #### FIG BOXPLOT
    p <- ggplot(df, aes(x = sujet, y = fc, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste(pair_sel, "fc", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    # NAME
    file_boxplot_subjectwise = paste("boxplot", pair_sel, band_sel, "fc_WPLI_subjectwise_ALLCOND.png", sep = "_")
    
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### model
    complex_form <- fc ~ phase * cond + (phase | sujet/pair_i)
    simple_form  <- fc ~ phase * cond + (1 | sujet/pair_i)
    
    model <- tryCatch({
      mod_attempt <- lmer(
        simple_form,
        data = df,
        control = lmerControl(optCtrl = list(maxfun = 2e5))
      )
      
      # Check for convergence issue
      if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
          isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️ lmer() model failed or was singular → switching to lm().")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return the valid lmer model
      
    }, error = function(e) {
      # On error or forced fallback: use lm instead
      lm(fc ~ phase * cond, data = df)
    })
    
    summary(model)
    
    #### FIG HIST
    # NAME
    filename_hist = paste("histogram", pair_sel, band_sel, "fc_ALLCOND.png", sep = "_")
    
    skew_chan = round(skewness(df$fc), 2)
    kurt_chan = round(kurtosis(df$fc), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$fc,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "fc values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste(pair_sel, band_sel, "fc", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG QQPLOT
    # NAME
    filename_qqplot = paste("qqplot", pair_sel, band_sel, "fc_FR_CV.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, "qqplot_bi"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### mixed model df
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    # NAME
    filesxlsx_ROI = paste("lmm", pair_sel, band_sel, "res_FC_ALLCOND.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
    
  }
}




################
#### FC BI ####
################

######## FR_CV BI ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_WPLI_FC_FR_CV_bi.xlsx", sep  = "/"))

pair_list <- unique(df_raw$pair)
band_list <- unique(df_raw$band)
phase_list <- unique(df_raw$phase)

pair_sel = pair_list[2]
band_sel = band_list[2]

for (band_sel in band_list) { 
  
  for (pair_sel in pair_list) {
    
    withCallingHandlers({
    
    print(band_sel)
    print(pair_sel)
    
    df_onepair <- subset(df_raw, band == band_sel & pair == pair_sel)
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_onepair[c("sujet", "phase", "pair_i", "fc")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$phase <- as.factor(df$phase)
    df$pair_ <- as.factor(df$pair_i)
    
    df$phase <- relevel(df$phase, ref = "I")
    
    #### FIG BOXPLOT
    p <- ggplot(df, aes(x = sujet, y = fc, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste(pair_sel, "fc", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    # NAME
    file_boxplot_subjectwise = paste("boxplot", pair_sel, band_sel, "fc_WPLI_subjectwise_FR_CV_bi.png", sep = "_")
    
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### model
    complex_form <- fc ~ phase + (phase | sujet/pair_i)
    simple_form  <- fc ~ phase + (1 | sujet/pair_i)
    
    model <- tryCatch({
      mod_attempt <- lmer(
        simple_form,
        data = df,
        control = lmerControl(optCtrl = list(maxfun = 2e5))
      )
      
      # Check for convergence issue
      if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
          isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️ lmer() model failed or was singular → switching to lm().")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return the valid lmer model
      
    }, error = function(e) {
      # On error or forced fallback: use lm instead
      lm(fc ~ phase, data = df)
    })
    
    summary(model)
    
    #### FIG HIST
    # NAME
    filename_hist = paste("histogram", pair_sel, band_sel, "fc_FR_CV_bi.png", sep = "_")
    
    skew_chan = round(skewness(df$fc), 2)
    kurt_chan = round(kurtosis(df$fc), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$fc,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "fc values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste(pair_sel, "fc", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG QQPLOT
    # NAME
    filename_qqplot = paste("qqplot", pair_sel, band_sel, "fc_FR_CV_bi.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, "qqplot_bi"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### mixed model df
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    # NAME
    filesxlsx_ROI = paste("lmm", pair_sel, band_sel, "res_FC_FR_CV_bi.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
    
  }
}



######## ALLCOND BI ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/FC/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_WPLI_FC_ALLCOND_bi.xlsx", sep  = "/"))

pair_list <- unique(df_raw$pair)
band_list <- unique(df_raw$band)
phase_list <- unique(df_raw$phase)

pair_sel = pair_list[2]
band_sel = band_list[2]

for (band_sel in band_list) { 
  
  for (pair_sel in pair_list) {
    
    withCallingHandlers({
    
    print(band_sel)
    print(pair_sel)
    
    df_onepair <- subset(df_raw, band == band_sel & pair == pair_sel)
    df_onepair$pair_i <- paste(df_onepair$sujet, df_onepair$pair_i, sep = "_")
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_onepair[c("sujet", "phase", "pair_i", "fc", "cond")]
    
    # Convert categorical variables to factors
    #df$sujet <- as.factor(df$sujet)
    df$phase <- as.factor(df$phase)
    df$pair_i <- as.factor(df$pair_i)
    df$cond <- as.factor(df$cond)
    
    df$phase <- relevel(df$phase, ref = "I")
    
    #### FIG BOXPLOT
    p <- ggplot(df, aes(x = sujet, y = fc, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste(pair_sel, "fc", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
      
    # NAME
    file_boxplot_subjectwise = paste("boxplot", pair_sel, band_sel, "fc_WPLI_subjectwise_ALLCOND_bi.png", sep = "_")
    
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### model
    complex_form <- fc ~ phase * cond + (phase | sujet/pair_i)
    simple_form  <- fc ~ phase * cond + (1 | sujet/pair_i)
    
    model <- tryCatch({
      mod_attempt <- lmer(
        simple_form,
        data = df,
        control = lmerControl(optCtrl = list(maxfun = 2e5))
      )
      
      # Check for convergence issue
      if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
          isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️ lmer() model failed or was singular → switching to lm().")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return the valid lmer model
      
    }, error = function(e) {
      # On error or forced fallback: use lm instead
      lm(fc ~ phase * cond, data = df)
    })
    
    summary(model)
    
    #### FIG HIST
    # NAME
    filename_hist = paste("histogram", pair_sel, band_sel, "fc_ALLCOND_bi.png", sep = "_")
    
    skew_chan = round(skewness(df$fc), 2)
    kurt_chan = round(kurtosis(df$fc), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$fc,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "fc values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste(pair_sel, band_sel, "fc", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### MODEL RES
    summary(model)
    
    #### FIG QQPLOT
    # NAME
    filename_qqplot = paste("qqplot", pair_sel, band_sel, "fc_FR_CV_bi.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, "qqplot_bi"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### mixed model df
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    # NAME
    filesxlsx_ROI = paste("lmm", pair_sel, band_sel, "res_FC_ALLCOND_bi.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
    
  }
}
