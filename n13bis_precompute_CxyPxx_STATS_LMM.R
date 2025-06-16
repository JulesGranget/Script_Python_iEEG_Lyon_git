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
#### Cxy ####
################

######## FR_CV ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_Cxy_FR_CV_filt.xlsx", sep  = "/"))
df_raw$chan <- paste0(df_raw$sujet, "_", df_raw$chan)

ROI_list <- unique(df_raw$ROI)

ROI_sel = ROI_list[1]


for (ROI_sel in ROI_list) {
  
  withCallingHandlers({
    
    print(ROI_sel)
    
    df_oneROI <- subset(df_raw, ROI == ROI_sel)
    
    print(subset(df_oneROI, cond == 'FR_CV') %>% count(sujet))
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_oneROI[c("sujet", "chan", "Cxy", "resp")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$chan <- as.factor(df$chan)
    #df$cond <- as.factor(df$cond)
    
    #### FIG 1
    p <- ggplot(df, aes(x = sujet, y = Cxy, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste("FR_CV", ROI_sel, "Cxy", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    file_boxplot_subjectwise = paste("FR_CV_boxplot", ROI_sel, "Cxy_subjectwise.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### MODEL
    complex_form <- Cxy ~ resp + (resp | sujet/chan)
    #simple_form  <- Cxy ~ resp + (1 | sujet/chan)
    simple_form  <- Cxy ~ resp + (1 | sujet)
    
    model <- tryCatch({
      mod_attempt <- glmer(
        simple_form,
        data = df,
        family = Gamma(link = "log"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )
      
      # Check for convergence issue
      if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
          isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️ glmer() model failed or was singular → switching to glm().")
        stop("Trigger fallback to glm")
      }
      
      mod_attempt  # return the valid glmer model
      
    }, error = function(e) {
      # On error or forced fallback: use glm instead
      glm(Cxy ~ resp, data = df, family = Gamma(link = "log"))
    })
    
    summary(model)
    
    #### FIG 2
    filename_hist = paste("FR_CV_histogram", ROI_sel, "Cxy.png", sep = "_")
    
    skew_chan = round(skewness(df$Cxy), 2)
    kurt_chan = round(kurtosis(df$Cxy), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$Cxy,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "Cxy values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste("FR_CV", ROI_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG 3
    filename_qqplot = paste("FR_CV_qqplot", ROI_sel, "Cxy.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, "qqplot"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### EXPORT MODEL RES
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    filesxlsx_ROI = paste("FR_CV_Cxy_lmm", ROI_sel, "res.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
  }, warning = function(w) {
    message("Warning: ", conditionMessage(w))  # shows immediately
    invokeRestart("muffleWarning")
  })
  
  
}




######## ALLCOND ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/df_lmm"


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Cxy_ALLCOND_filt.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

ROI_list <- unique(df_raw$ROI)

ROI_sel = ROI_list[18]

for (ROI_sel in ROI_list) {
  
  withCallingHandlers({
    
    print(ROI_sel)
    
    df_oneROI <- subset(df_raw, ROI == ROI_sel)
    
    print(subset(df_oneROI, cond == 'FR_CV') %>% count(sujet))
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_oneROI[c("sujet", "chan", "Cxy", "resp")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$chan <- as.factor(df$chan)
    #df$cond <- as.factor(df$cond)
    
    #### FIG 1
    p <- ggplot(df, aes(x = sujet, y = Cxy, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste("ALLCOND", ROI_sel, "Cxy", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    file_boxplot_subjectwise = paste("ALLCOND_boxplot", ROI_sel, "Cxy_subjectwise.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### MODEL
    complex_form <- Cxy ~ resp + (resp | sujet/chan)
    #simple_form  <- Cxy ~ resp + (1 | sujet/chan)
    simple_form  <- Cxy ~ resp + (1 | sujet)
    
    model <- tryCatch({
      mod_attempt <- glmer(
        simple_form,
        data = df,
        family = Gamma(link = "log"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )
      
      # Check for convergence issue
      if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
          isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️ glmer() model failed or was singular → switching to glm().")
        stop("Trigger fallback to glm")
      }
      
      mod_attempt  # return the valid glmer model
      
    }, error = function(e) {
      # On error or forced fallback: use glm instead
      glm(Cxy ~ resp, data = df, family = Gamma(link = "log"))
    })
    
    summary(model)
    
    #### FIG 2
    filename_hist = paste("ALLCOND_histogram", ROI_sel, "Cxy.png", sep = "_")
    
    skew_chan = round(skewness(df$Cxy), 2)
    kurt_chan = round(kurtosis(df$Cxy), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$Cxy,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "Cxy values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste("ALLCOND", ROI_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG 3
    filename_qqplot = paste("ALLCOND_qqplot", ROI_sel, "Cxy.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, "qqplot"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### EXPORT MODEL RES
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    filesxlsx_ROI = paste("ALLCOND_Cxy_lmm", ROI_sel, "res.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
  }, warning = function(w) {
    message("Warning: ", conditionMessage(w))  # shows immediately
    invokeRestart("muffleWarning")
  })
  
  
}




################
#### Cxy BI ####
################

######## FR_CV BI ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_Cxy_FR_CV_filt_bi.xlsx", sep  = "/"))
df_raw$chan <- paste0(df_raw$sujet, "_", df_raw$chan)

ROI_list <- unique(df_raw$ROI)

ROI_sel = ROI_list[1]


for (ROI_sel in ROI_list) {
  
  withCallingHandlers({
    
    print(ROI_sel)
    
    df_oneROI <- subset(df_raw, ROI == ROI_sel)
    
    print(subset(df_oneROI, cond == 'FR_CV') %>% count(sujet))
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_oneROI[c("sujet", "chan", "Cxy", "resp")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$chan <- as.factor(df$chan)
    #df$cond <- as.factor(df$cond)
    
    #### FIG 1
    p <- ggplot(df, aes(x = sujet, y = Cxy, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste("FR_CV", ROI_sel, "Cxy_bi", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    file_boxplot_subjectwise = paste("FR_CV_boxplot", ROI_sel, "Cxy_subjectwise_bi.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### MODEL
    complex_form <- Cxy ~ resp + (resp | sujet/chan)
    #simple_form  <- Cxy ~ resp + (1 | sujet/chan)
    simple_form  <- Cxy ~ resp + (1 | sujet)
    
    model <- tryCatch({
      mod_attempt <- glmer(
        simple_form,
        data = df,
        family = Gamma(link = "log"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )
      
      # Check for convergence issue
      if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
          isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️ glmer() model failed or was singular → switching to glm().")
        stop("Trigger fallback to glm")
      }
      
      mod_attempt  # return the valid glmer model
      
    }, error = function(e) {
      # On error or forced fallback: use glm instead
      glm(Cxy ~ resp, data = df, family = Gamma(link = "log"))
    })
    
    summary(model)
    
    #### FIG 2
    filename_hist = paste("FR_CV_histogram", ROI_sel, "Cxy_bi.png", sep = "_")
    
    skew_chan = round(skewness(df$Cxy), 2)
    kurt_chan = round(kurtosis(df$Cxy), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$Cxy,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "Cxy values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste("FR_CV", ROI_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG 3
    filename_qqplot = paste("FR_CV_qqplot", ROI_sel, "Cxy_bi.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, "qqplot"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### EXPORT MODEL RES
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    filesxlsx_ROI = paste("FR_CV_Cxy_lmm", ROI_sel, "res_bi.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
  }, warning = function(w) {
    message("Warning: ", conditionMessage(w))  # shows immediately
    invokeRestart("muffleWarning")
  })
  
  
}




######## ALLCOND BI ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/df_lmm"


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Cxy_ALLCOND_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

ROI_list <- unique(df_raw$ROI)

ROI_sel = ROI_list[18]

for (ROI_sel in ROI_list) {
  
  withCallingHandlers({
  
  print(ROI_sel)
    
  df_oneROI <- subset(df_raw, ROI == ROI_sel)
  
  print(subset(df_oneROI, cond == 'FR_CV') %>% count(sujet))
  
  #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
  df <- df_oneROI[c("sujet", "chan", "Cxy", "resp")]
  
  # Convert categorical variables to factors
  df$sujet <- as.factor(df$sujet)
  df$chan <- as.factor(df$chan)
  #df$cond <- as.factor(df$cond)
  
  #### FIG 1
  p <- ggplot(df, aes(x = sujet, y = Cxy, color = sujet, fill = sujet)) +
    geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                 position = position_dodge(.9)) +
    stat_summary(fun = median, geom = "point", size = 2,
                 position = position_dodge(.9), color = "white")+
    labs(
      title    = paste("ALLCOND", ROI_sel, "Cxy_bi", sep = "_")
    ) +
    theme(
      plot.title    = element_text(hjust = 0.5),
    )
  
  p
  
  file_boxplot_subjectwise = paste("ALLCOND_boxplot", ROI_sel, "Cxy_subjectwise_bi.png", sep = "_")
  # then explicitly:
  ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
  
  #### MODEL
  complex_form <- Cxy ~ resp + (resp | sujet/chan)
  #simple_form  <- Cxy ~ resp + (1 | sujet/chan)
  simple_form  <- Cxy ~ resp + (1 | sujet)
  
  model <- tryCatch({
    mod_attempt <- glmer(
      simple_form,
      data = df,
      family = Gamma(link = "log"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
    
    # Check for convergence issue
    if (!is.null(mod_attempt@optinfo$conv$lme4$messages) ||
        isSingular(mod_attempt, tol = 1e-4)) {
      message("⚠️ glmer() model failed or was singular → switching to glm().")
      stop("Trigger fallback to glm")
    }
    
    mod_attempt  # return the valid glmer model
    
  }, error = function(e) {
    # On error or forced fallback: use glm instead
    glm(Cxy ~ resp, data = df, family = Gamma(link = "log"))
  })
  
  summary(model)
  
  #### FIG 2
  filename_hist = paste("ALLCOND_histogram", ROI_sel, "Cxy_bi.png", sep = "_")
  
  skew_chan = round(skewness(df$Cxy), 2)
  kurt_chan = round(kurtosis(df$Cxy), 2)
  
  png(
    filename = paste(outputdir_fig, filename_hist, sep = "/"),
    width    = 800,    # width in pixels
    height   = 600,    # height in pixels
    res      = 100     # resolution (pixels per inch)
  )
  
  hist(
    df$Cxy,
    breaks = 30,
    main   = "",          # leave main blank for now
    xlab   = "Cxy values",
    ylab   = "Resp",
    col    = "lightblue",
    border = "white"
  )
  
  title(
    main     = paste("ALLCOND", ROI_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
    adj      = 0.5,       # 0.5 = center
    cex.main = 1.5,       # main title size
    font.main= 2,         # bold
    cex.sub  = 1.0        # subtitle size
  )
  
  dev.off()
  
  #### FIG 3
  filename_qqplot = paste("ALLCOND_qqplot", ROI_sel, "Cxy_bi.png", sep = "_")
  
  png(
    filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
    width    = 800,    # width in pixels
    height   = 600,    # height in pixels
    res      = 100     # resolution (pixels per inch)
  )
  
  qqnorm(resid(model))
  qqline(resid(model))  # points fall nicely onto the line - good!
  
  title(
    sub     = paste(ROI_sel, "qqplot"),
    adj      = 0.5,       # 0.5 = center
    cex.main = 1.5,       # main title size
    font.main= 2,         # bold
    cex.sub  = 1.0        # subtitle size
  )
  
  dev.off()
  
  #### EXPORT MODEL RES
  tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
  
  model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
  
  filesxlsx_ROI = paste("ALLCOND_Cxy_lmm", ROI_sel, "res_bi.xlsx", sep = "_")
  writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
  
  }, warning = function(w) {
    message("Warning: ", conditionMessage(w))  # shows immediately
    invokeRestart("muffleWarning")
  })
  
  
}








################
#### PXX ####
################

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/df_lmm"


######## FR_CV ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Pxx_FR_CV_filt.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

ROI_list <- unique(df_raw$ROI)
band_list <- c('theta', 'alpha', 'beta', 'gamma')

band_sel = 'theta'
ROI_sel = ROI_list[1]

for (ROI_sel in ROI_list) {
  
  for (band_sel in band_list) {
    
    withCallingHandlers({
      
      print(ROI_sel)
      print(band_sel)
      
      df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel & phase %in% phase_list)
      
      print(subset(df_oneROI, phase == phase_sel) %>% count(sujet))
      
      df <- df_oneROI[c("sujet", "chan", "Pxx", "phase")]
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$chan <- as.factor(df$chan)
      df$phase <- as.factor(df$phase)
      #df$cond <- as.factor(df$cond)
      
      df$phase <- relevel(df$phase, ref = "expi")
      
      #### FIG 1
      p <- ggplot(df, aes(x = sujet, y = Pxx, color = sujet, fill = sujet)) +
        geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                     position = position_dodge(.9)) +
        stat_summary(fun = median, geom = "point", size = 2,
                     position = position_dodge(.9), color = "white")+
        labs(
          title    = paste(ROI_sel, band_sel, "Pxx", sep = "_")
        ) +
        theme(
          plot.title    = element_text(hjust = 0.5),
        )
      
      p
      
      file_boxplot_subjectwise = paste("FR_CV_boxplot", ROI_sel, band_sel, "Pxx_subjectwise.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- Pxx ~ resp + (resp | sujet/chan)
      #simple_form  <- Pxx ~ resp + (1 | sujet/chan)
      simple_form  <- Pxx ~ phase + (1 | sujet)
      
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
        lm(Pxx ~ phase, data = df)
      })
      
      summary(model)
      
      #### FIG 2
      filename_hist = paste("FR_CV_histogram", ROI_sel, band_sel, "Pxx.png", sep = "_")
      
      skew_chan = round(skewness(df$Pxx), 2)
      kurt_chan = round(kurtosis(df$Pxx), 2)
      
      png(
        filename = paste(outputdir_fig, filename_hist, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      hist(
        df$Pxx,
        breaks = 30,
        main   = "",          # leave main blank for now
        xlab   = "Pxx values",
        ylab   = "Resp",
        col    = "lightblue",
        border = "white"
      )
      
      title(
        main     = paste("FR_CV", ROI_sel, band_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### FIG 3
      filename_qqplot = paste("FR_CV_qqplot", ROI_sel, band_sel, "Pxx.png", sep = "_")
      
      png(
        filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      qqnorm(resid(model))
      qqline(resid(model))  # points fall nicely onto the line - good!
      
      title(
        sub     = paste(ROI_sel, band_sel, "qqplot"),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### EXPORT MODEL RES
      tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
      
      model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
      
      filesxlsx_ROI = paste("FR_CV_Pxx_lmm", ROI_sel, band_sel, "res.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
      
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
  }
}





######## ALLCOND ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Pxx_ALLCOND_filt.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

ROI_list <- unique(df_raw$ROI)
cond_list <- unique(df_raw$cond)
band_list <- c('theta', 'alpha', 'beta', 'gamma')

band_sel = 'theta'
ROI_sel = ROI_list[1]

for (ROI_sel in ROI_list) {
  
  for (band_sel in band_list) {
    
    withCallingHandlers({
      
      print(ROI_sel)
      print(band_sel)
      
      df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel & phase %in% phase_list)
      
      print(subset(df_oneROI, phase == phase_sel) %>% count(sujet))
      
      df <- df_oneROI[c("sujet", "chan", "Pxx", "phase", "cond")]
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$chan <- as.factor(df$chan)
      df$phase <- as.factor(df$phase)
      df$cond <- as.factor(df$cond)
      
      df$phase <- relevel(df$phase, ref = "expi")
      df$cond <- relevel(df$cond, ref = "FR_CV")
      
      #### FIG 1
      p <- ggplot(df, aes(x = sujet, y = Pxx, color = sujet, fill = sujet)) +
        geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                     position = position_dodge(.9)) +
        stat_summary(fun = median, geom = "point", size = 2,
                     position = position_dodge(.9), color = "white")+
        labs(
          title    = paste(ROI_sel, band_sel, "Pxx", sep = "_")
        ) +
        theme(
          plot.title    = element_text(hjust = 0.5),
        )
      
      p
      
      file_boxplot_subjectwise = paste("ALLCOND_boxplot", ROI_sel, band_sel, "Pxx_subjectwise.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- Pxx ~ resp + (resp | sujet/chan)
      #simple_form  <- Pxx ~ resp + (1 | sujet/chan)
      simple_form  <- Pxx ~ phase * cond + (1 | sujet)
      
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
        lm(Pxx ~ phase * cond, data = df)
      })
      
      summary(model)
      
      #### FIG 2
      filename_hist = paste("ALLCOND_histogram", ROI_sel, band_sel, "Pxx.png", sep = "_")
      
      skew_chan = round(skewness(df$Pxx), 2)
      kurt_chan = round(kurtosis(df$Pxx), 2)
      
      png(
        filename = paste(outputdir_fig, filename_hist, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      hist(
        df$Pxx,
        breaks = 30,
        main   = "",          # leave main blank for now
        xlab   = "Pxx values",
        ylab   = "Resp",
        col    = "lightblue",
        border = "white"
      )
      
      title(
        main     = paste("ALLCOND", ROI_sel, band_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### FIG 3
      filename_qqplot = paste("ALLCOND_qqplot", ROI_sel, band_sel, "Pxx.png", sep = "_")
      
      png(
        filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      qqnorm(resid(model))
      qqline(resid(model))  # points fall nicely onto the line - good!
      
      title(
        sub     = paste(ROI_sel, band_sel, "qqplot"),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### EXPORT MODEL RES
      tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
      
      model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
      
      filesxlsx_ROI = paste("ALLCOND_Pxx_lmm", ROI_sel, band_sel, "res.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
      
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
  }
}





################
#### PXX BI ####
################

root = "Z:/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "Z:/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/fig"
outputdir_df_lmm = "Z:/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/df_lmm"


root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/df_lmm"


######## FR_CV BI ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Pxx_FR_CV_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

ROI_list <- unique(df_raw$ROI)
band_list <- c('theta', 'alpha', 'beta', 'gamma')

band_sel = 'theta'
ROI_sel = ROI_list[1]

for (ROI_sel in ROI_list) {
  
  for (band_sel in band_list) {
    
    withCallingHandlers({
    
    print(ROI_sel)
    print(band_sel)

    df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel & phase %in% phase_list)
    
    print(subset(df_oneROI, phase == phase_sel) %>% count(sujet))
    
    df <- df_oneROI[c("sujet", "chan", "Pxx", "phase")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$chan <- as.factor(df$chan)
    df$phase <- as.factor(df$phase)
    #df$cond <- as.factor(df$cond)
    
    df$phase <- relevel(df$phase, ref = "expi")
    
    #### FIG 1
    p <- ggplot(df, aes(x = sujet, y = Pxx, color = sujet, fill = sujet)) +
      geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                   position = position_dodge(.9)) +
      stat_summary(fun = median, geom = "point", size = 2,
                   position = position_dodge(.9), color = "white")+
      labs(
        title    = paste(ROI_sel, band_sel, "Pxx_bi", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    file_boxplot_subjectwise = paste("FR_CV_boxplot", ROI_sel, band_sel, "Pxx_subjectwise_bi.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### MODEL
    #complex_form <- Pxx ~ resp + (resp | sujet/chan)
    #simple_form  <- Pxx ~ resp + (1 | sujet/chan)
    simple_form  <- Pxx ~ phase + (1 | sujet)
    
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
      lm(Pxx ~ phase, data = df)
    })
    
    summary(model)
    
    #### FIG 2
    filename_hist = paste("FR_CV_histogram", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
    
    skew_chan = round(skewness(df$Pxx), 2)
    kurt_chan = round(kurtosis(df$Pxx), 2)
    
    png(
      filename = paste(outputdir_fig, filename_hist, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    hist(
      df$Pxx,
      breaks = 30,
      main   = "",          # leave main blank for now
      xlab   = "Pxx values",
      ylab   = "Resp",
      col    = "lightblue",
      border = "white"
    )
    
    title(
      main     = paste("FR_CV", ROI_sel, band_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG 3
    filename_qqplot = paste("FR_CV_qqplot", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
    
    png(
      filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(ROI_sel, band_sel, "qqplot"),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### EXPORT MODEL RES
    tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
    
    model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
    
    filesxlsx_ROI = paste("FR_CV_Pxx_lmm", ROI_sel, band_sel, "res_bi.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })

  }
}





######## ALLCOND BI ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Pxx_ALLCOND_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

ROI_list <- unique(df_raw$ROI)
cond_list <- unique(df_raw$cond)
band_list <- c('theta', 'alpha', 'beta', 'gamma')

band_sel = 'theta'
ROI_sel = ROI_list[1]

for (ROI_sel in ROI_list) {
  
  for (band_sel in band_list) {
    
    withCallingHandlers({
      
      print(ROI_sel)
      print(band_sel)
      
      df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel & phase %in% phase_list)
      
      print(subset(df_oneROI, phase == phase_sel) %>% count(sujet))
      
      df <- df_oneROI[c("sujet", "chan", "Pxx", "phase", "cond")]
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$chan <- as.factor(df$chan)
      df$phase <- as.factor(df$phase)
      df$cond <- as.factor(df$cond)
      
      df$phase <- relevel(df$phase, ref = "expi")
      df$cond <- relevel(df$cond, ref = "FR_CV")
      
      #### FIG 1
      p <- ggplot(df, aes(x = sujet, y = Pxx, color = sujet, fill = sujet)) +
        geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                     position = position_dodge(.9)) +
        stat_summary(fun = median, geom = "point", size = 2,
                     position = position_dodge(.9), color = "white")+
        labs(
          title    = paste(ROI_sel, band_sel, "Pxx_bi", sep = "_")
        ) +
        theme(
          plot.title    = element_text(hjust = 0.5),
        )
      
      p
      
      file_boxplot_subjectwise = paste("ALLCOND_boxplot", ROI_sel, band_sel, "Pxx_subjectwise_bi.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- Pxx ~ resp + (resp | sujet/chan)
      #simple_form  <- Pxx ~ resp + (1 | sujet/chan)
      simple_form  <- Pxx ~ phase * cond + (1 | sujet)
      
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
        lm(Pxx ~ phase * cond, data = df)
      })
      
      summary(model)
      
      #### FIG 2
      filename_hist = paste("ALLCOND_histogram", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
      
      skew_chan = round(skewness(df$Pxx), 2)
      kurt_chan = round(kurtosis(df$Pxx), 2)
      
      png(
        filename = paste(outputdir_fig, filename_hist, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      hist(
        df$Pxx,
        breaks = 30,
        main   = "",          # leave main blank for now
        xlab   = "Pxx values",
        ylab   = "Resp",
        col    = "lightblue",
        border = "white"
      )
      
      title(
        main     = paste("ALLCOND", ROI_sel, band_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### FIG 3
      filename_qqplot = paste("ALLCOND_qqplot", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
      
      png(
        filename = paste(outputdir_fig, filename_qqplot, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      qqnorm(resid(model))
      qqline(resid(model))  # points fall nicely onto the line - good!
      
      title(
        sub     = paste(ROI_sel, band_sel, "qqplot"),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### EXPORT MODEL RES
      tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
      
      model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
      
      filesxlsx_ROI = paste("ALLCOND_Pxx_lmm", ROI_sel, band_sel, "res_bi.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
      
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
  }
}



