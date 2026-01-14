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
        title    = paste("FR_CV", ROI_sel, "Cxy", sep = "_")
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
    simple_form_refit  <- Cxy ~ resp
    
    model <- tryCatch({
      
      warn_triggered <- FALSE  # will catch if any warning is raised
      
      mod_attempt <- withCallingHandlers(
        expr = {
          glmer(
            simple_form,
            data = df,
            family = Gamma(link = "log"),
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
          )
        },
        warning = function(w) {
          message("⚠️ Warning during lmer(): ", conditionMessage(w))
          warn_triggered <<- TRUE
          invokeRestart("muffleWarning")  # suppress so execution continues
        }
      )
      
      # Force fallback if warning was raised or model is singular
      if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️️ Fallback to lm() due to warning or singular fit")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return valid model if all checks passed
      
    }, error = function(e) {
      glm(simple_form_refit, data = df, family = Gamma(link = "log"))
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


######## ATTENTION ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_Cxy_ALLCOND_filt_bi.xlsx", sep  = "/"))
df_raw$chan <- paste0(df_raw$sujet, "_", df_raw$chan)

df_raw <- subset(df_raw, cond %in% c('FR_CV', 'RD_CV'))

ROI_list <- unique(df_raw$ROI)

ROI_sel = ROI_list[1]


for (ROI_sel in ROI_list) {
  
  withCallingHandlers({
    
    print(ROI_sel)
    
    df_oneROI <- subset(df_raw, ROI == ROI_sel)
    
    print(subset(df_oneROI, cond == 'FR_CV') %>% count(sujet))
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_oneROI[c("sujet", "chan", "Cxy", "cond")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$chan <- as.factor(df$chan)
    df$cond <- as.factor(df$cond)
    
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
    
    file_boxplot_subjectwise = paste("FR_CV_ATTENTION_boxplot", ROI_sel, "Cxy_subjectwise_bi.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### MODEL
    complex_form <- Cxy ~ resp + (resp | sujet/chan)
    #simple_form  <- Cxy ~ resp + (1 | sujet/chan)
    simple_form  <- Cxy ~ cond + (1 | sujet)
    simple_form_refit  <- Cxy ~ cond
    
    model <- tryCatch({
      
      warn_triggered <- FALSE  # will catch if any warning is raised
      
      mod_attempt <- withCallingHandlers(
        expr = {
          glmer(
            simple_form,
            data = df,
            family = Gamma(link = "log"),
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
          )
        },
        warning = function(w) {
          message("⚠️ Warning during lmer(): ", conditionMessage(w))
          warn_triggered <<- TRUE
          invokeRestart("muffleWarning")  # suppress so execution continues
        }
      )
      
      # Force fallback if warning was raised or model is singular
      if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️️ Fallback to lm() due to warning or singular fit")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return valid model if all checks passed
      
    }, error = function(e) {
      glm(simple_form_refit, data = df, family = Gamma(link = "log"))
    })
    
    summary(model)
    
    #### FIG 2
    filename_hist = paste("FR_CV_ATTENTION_histogram", ROI_sel, "Cxy_bi.png", sep = "_")
    
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
    filename_qqplot = paste("FR_CV_ATTENTION_qqplot", ROI_sel, "Cxy_bi.png", sep = "_")
    
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
    
    filesxlsx_ROI = paste("ALLCOND_ATTENTION_Cxy_lmm", ROI_sel, "res_bi.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
    
  }, warning = function(w) {
    message("Warning: ", conditionMessage(w))  # shows immediately
    invokeRestart("muffleWarning")
  })
  
  
}


######## RD_CV ########

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Cxy/df_lmm"

# Load the Excel data
df_raw <- read_excel(paste(root, "df_Cxy_ALLCOND_filt_bi.xlsx", sep  = "/"))
df_raw$chan <- paste0(df_raw$sujet, "_", df_raw$chan)

df_raw <- subset(df_raw, cond %in% c('FR_CV', 'RD_CV'))

ROI_list <- unique(df_raw$ROI)

ROI_sel = ROI_list[1]


for (ROI_sel in ROI_list) {
  
  withCallingHandlers({
    
    print(ROI_sel)
    
    df_oneROI <- subset(df_raw, ROI == ROI_sel)
    
    print(subset(df_oneROI, cond == 'FR_CV') %>% count(sujet))
    
    #df <- df_oneROI[c("sujet", "Cxy", "cond", "resp")]
    df <- df_oneROI[c("sujet", "chan", "Cxy", "cond", "resp")]
    
    # Convert categorical variables to factors
    df$sujet <- as.factor(df$sujet)
    df$chan <- as.factor(df$chan)
    df$cond <- as.factor(df$cond)
    
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
    
    file_boxplot_subjectwise = paste("RD_CV_boxplot", ROI_sel, "Cxy_subjectwise_bi.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### MODEL
    complex_form <- Cxy ~ resp + (resp | sujet/chan)
    #simple_form  <- Cxy ~ resp + (1 | sujet/chan)
    simple_form  <- Cxy ~ cond * resp + (1 | sujet)
    simple_form_refit  <- Cxy ~ cond * resp
    
    model <- tryCatch({
      
      warn_triggered <- FALSE  # will catch if any warning is raised
      
      mod_attempt <- withCallingHandlers(
        expr = {
          glmer(
            simple_form,
            data = df,
            family = Gamma(link = "log"),
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
          )
        },
        warning = function(w) {
          message("⚠️ Warning during lmer(): ", conditionMessage(w))
          warn_triggered <<- TRUE
          invokeRestart("muffleWarning")  # suppress so execution continues
        }
      )
      
      # Force fallback if warning was raised or model is singular
      if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️️ Fallback to lm() due to warning or singular fit")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return valid model if all checks passed
      
    }, error = function(e) {
      glm(simple_form_refit, data = df, family = Gamma(link = "log"))
    })
    
    summary(model)
    
    #### FIG 2
    filename_hist = paste("RD_CV_histogram", ROI_sel, "Cxy_bi.png", sep = "_")
    
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
      main     = paste("RD_CV", ROI_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    #### FIG 3
    filename_qqplot = paste("RD_CV_qqplot", ROI_sel, "Cxy_bi.png", sep = "_")
    
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
    
    filesxlsx_ROI = paste("RD_CV_Cxy_lmm", ROI_sel, "res_bi.xlsx", sep = "_")
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
df_raw <- read_excel(paste(root, "df_Cxy_ALLCOND_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

df_raw <- subset(df_raw, cond %in% c('RD_CV', 'RD_SV', 'RD_FV'))

ROI_list <- unique(df_raw$ROI)

ROI_sel = ROI_list[18]

for (ROI_sel in ROI_list) {
  
  withCallingHandlers({
    
    print(ROI_sel)
    
    df_oneROI <- subset(df_raw, ROI == ROI_sel)
    
    print(subset(df_oneROI, cond == 'RD_CV') %>% count(sujet))
    
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
    
    file_boxplot_subjectwise = paste("ALLCOND_boxplot", ROI_sel, "Cxy_subjectwise_bi.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    #### MODEL
    complex_form <- Cxy ~ resp + (resp | sujet/chan)
    #simple_form  <- Cxy ~ resp + (1 | sujet/chan)
    simple_form  <- Cxy ~ resp + (1 | sujet)
    simple_form_refit  <- Cxy ~ resp
    
    model <- tryCatch({
      
      warn_triggered <- FALSE  # will catch if any warning is raised
      
      mod_attempt <- withCallingHandlers(
        expr = {
          glmer(
            simple_form,
            data = df,
            family = Gamma(link = "log"),
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
          )
        },
        warning = function(w) {
          message("⚠️ Warning during lmer(): ", conditionMessage(w))
          warn_triggered <<- TRUE
          invokeRestart("muffleWarning")  # suppress so execution continues
        }
      )
      
      # Force fallback if warning was raised or model is singular
      if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
        message("⚠️️ Fallback to lm() due to warning or singular fit")
        stop("Trigger fallback to lm")
      }
      
      mod_attempt  # return valid model if all checks passed
      
    }, error = function(e) {
      glm(simple_form_refit, data = df, family = Gamma(link = "log"))
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


######## FR_CV WHOLE ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Pxx_FR_CV_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))
df_raw <- subset(df_raw, phase == 'whole')

ROI_list <- unique(df_raw$ROI)
band_list <- c('theta', 'alpha', 'beta', 'gamma')

band_sel = 'theta'
ROI_sel = ROI_list[1]

for (ROI_sel in ROI_list) {
  
  for (band_sel in band_list) {
    
    withCallingHandlers({
      
      print(ROI_sel)
      print(band_sel)
      
      df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel)
      
      print(subset(df_oneROI) %>% count(sujet))
      
      df <- df_oneROI[c("sujet", "chan", "Pxx", "resp")]
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$chan <- as.factor(df$chan)
      #df$cond <- as.factor(df$cond)
      
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
      
      file_boxplot_subjectwise = paste("FR_CV_WHOLE_boxplot", ROI_sel, band_sel, "Pxx_subjectwise_bi.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- Pxx ~ resp + (resp | sujet/chan)
      #simple_form  <- Pxx ~ resp + (1 | sujet/chan)
      simple_form  <- Pxx ~ resp + (1 | sujet)
      simple_form_refit  <- Pxx ~ resp
      
      model <- tryCatch({
        
        warn_triggered <- FALSE  # will catch if any warning is raised
        
        mod_attempt <- withCallingHandlers(
          expr = {
            lmer(
              simple_form,
              data = df,
              control = lmerControl(optCtrl = list(maxfun = 2e5))
            )
          },
          warning = function(w) {
            message("⚠️ Warning during lmer(): ", conditionMessage(w))
            warn_triggered <<- TRUE
            invokeRestart("muffleWarning")  # suppress so execution continues
          }
        )
        
        # Force fallback if warning was raised or model is singular
        if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
          message("⚠️️ Fallback to lm() due to warning or singular fit")
          stop("Trigger fallback to lm")
        }
        
        mod_attempt  # return valid model if all checks passed
        
      }, error = function(e) {
        lm(simple_form_refit, data = df)
      })
      
      summary(model)
      
      #### FIG 2
      filename_hist = paste("FR_CV_WHOLE_histogram", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
      
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
        main     = paste("FR_CV_WHOLE", ROI_sel, band_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### FIG 3
      filename_qqplot = paste("FR_CV_WHOLE_qqplot", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
      
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
      
      filesxlsx_ROI = paste("FR_CV_WHOLE_Pxx_lmm", ROI_sel, band_sel, "res_bi.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
      
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
  }
}


######## FR_CV I/E ########


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
          title    = paste(ROI_sel, band_sel, "Pxx", sep = "_")
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
      simple_form_refit  <- Pxx ~ phase
      
      model <- tryCatch({
        
        warn_triggered <- FALSE  # will catch if any warning is raised
        
        mod_attempt <- withCallingHandlers(
          expr = {
            lmer(
              simple_form,
              data = df,
              control = lmerControl(optCtrl = list(maxfun = 2e5))
            )
          },
          warning = function(w) {
            message("⚠️ Warning during lmer(): ", conditionMessage(w))
            warn_triggered <<- TRUE
            invokeRestart("muffleWarning")  # suppress so execution continues
          }
        )
        
        # Force fallback if warning was raised or model is singular
        if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
          message("⚠️️ Fallback to lm() due to warning or singular fit")
          stop("Trigger fallback to lm")
        }
        
        mod_attempt  # return valid model if all checks passed
        
      }, error = function(e) {
        lm(simple_form_refit, data = df)
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



######## ALLCOND WHOLE ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_Pxx_ALLCOND_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

df_raw <- subset(df_raw, phase == 'whole')

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
      
      df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel)
      
      print(subset(df_oneROI, cond == 'FR_CV') %>% count(sujet))
      
      df <- df_oneROI[c("sujet", "chan", "Pxx", "resp", "cond")]
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$chan <- as.factor(df$chan)
      df$cond <- as.factor(df$cond)
      
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
      
      file_boxplot_subjectwise = paste("ALLCOND_WHOLE_boxplot", ROI_sel, band_sel, "Pxx_subjectwise_bi.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- Pxx ~ resp + (resp | sujet/chan)
      #simple_form  <- Pxx ~ resp + (1 | sujet/chan)
      simple_form  <- Pxx ~ resp + (1 | sujet)
      simple_form_refit  <- Pxx ~ resp
      
      model <- tryCatch({
        
        warn_triggered <- FALSE  # will catch if any warning is raised
        
        mod_attempt <- withCallingHandlers(
          expr = {
            lmer(
              simple_form,
              data = df,
              control = lmerControl(optCtrl = list(maxfun = 2e5))
            )
          },
          warning = function(w) {
            message("⚠️ Warning during lmer(): ", conditionMessage(w))
            warn_triggered <<- TRUE
            invokeRestart("muffleWarning")  # suppress so execution continues
          }
        )
        
        # Force fallback if warning was raised or model is singular
        if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
          message("⚠️️ Fallback to lm() due to warning or singular fit")
          stop("Trigger fallback to lm")
        }
        
        mod_attempt  # return valid model if all checks passed
        
      }, error = function(e) {
        lm(simple_form_refit, data = df)
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



######## ALLCOND I/E ########


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
      
      print(subset(df_oneROI, phase == 'inspi' & cond == 'FR_CV') %>% count(sujet))
      
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
      
      file_boxplot_subjectwise = paste("ALLCOND_IE_boxplot", ROI_sel, band_sel, "Pxx_subjectwise_bi.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- Pxx ~ resp + (resp | sujet/chan)
      #simple_form  <- Pxx ~ resp + (1 | sujet/chan)
      simple_form  <- Pxx ~ phase * cond + (1 | sujet)
      simple_form_refit  <- Pxx ~ phase * cond
      
      model <- tryCatch({
        
        warn_triggered <- FALSE  # will catch if any warning is raised
        
        mod_attempt <- withCallingHandlers(
          expr = {
            lmer(
              simple_form,
              data = df,
              control = lmerControl(optCtrl = list(maxfun = 2e5))
            )
          },
          warning = function(w) {
            message("⚠️ Warning during lmer(): ", conditionMessage(w))
            warn_triggered <<- TRUE
            invokeRestart("muffleWarning")  # suppress so execution continues
          }
        )
        
        # Force fallback if warning was raised or model is singular
        if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
          message("⚠️️ Fallback to lm() due to warning or singular fit")
          stop("Trigger fallback to lm")
        }
        
        mod_attempt  # return valid model if all checks passed
        
      }, error = function(e) {
        lm(simple_form_refit, data = df)
      })
      
      summary(model)
      
      #### FIG 2
      filename_hist = paste("ALLCOND_IE_histogram", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
      
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
      filename_qqplot = paste("ALLCOND_IE_qqplot", ROI_sel, band_sel, "Pxx_bi.png", sep = "_")
      
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
      
      filesxlsx_ROI = paste("ALLCOND_IE_Pxx_lmm", ROI_sel, band_sel, "res_bi.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
      
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
  }
}




################
#### MI ####
################

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/df/df_aggregates/"
outputdir_fig = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/fig"
outputdir_df_lmm = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso201810_ieeg_respi_jules_valentin/iEEG_Lyon_VJ/Analyses/results/allplot/LMM/Pxx/df_lmm"


######## FR_CV WHOLE ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_MI_FR_CV_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))

ROI_list <- unique(df_raw$ROI)
band_list <- c('theta', 'alpha', 'beta', 'gamma')

band_sel = 'beta'
ROI_sel = ROI_list[23]

for (ROI_sel in ROI_list) {
  
  for (band_sel in band_list) {
    
    withCallingHandlers({
      
      print(ROI_sel)
      print(band_sel)
      
      df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel)
      
      print(subset(df_oneROI) %>% count(sujet))
      
      df <- df_oneROI[c("sujet", "chan", "MI", "resp")]
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$chan <- as.factor(df$chan)
      #df$cond <- as.factor(df$cond)
      
      #### FIG 1
      p <- ggplot(df, aes(x = sujet, y = MI, color = sujet, fill = sujet)) +
        geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                     position = position_dodge(.9)) +
        stat_summary(fun = median, geom = "point", size = 2,
                     position = position_dodge(.9), color = "white")+
        labs(
          title    = paste(ROI_sel, band_sel, "MI", sep = "_")
        ) +
        theme(
          plot.title    = element_text(hjust = 0.5),
        )
      
      p
      
      file_boxplot_subjectwise = paste("FR_CV_WHOLE_boxplot", ROI_sel, band_sel, "MI_subjectwise_bi.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- MI ~ resp + (resp | sujet/chan)
      #simple_form  <- MI ~ resp + (1 | sujet/chan)
      simple_form  <- MI ~ resp + (1 | sujet)
      simple_form_refit  <- MI ~ resp
      
      model <- tryCatch({
        
        warn_triggered <- FALSE  # will catch if any warning is raised
        
        mod_attempt <- withCallingHandlers(
          expr = {
            lmer(
              simple_form,
              data = df,
              control = lmerControl(optCtrl = list(maxfun = 2e5))
            )
          },
          warning = function(w) {
            message("⚠️ Warning during lmer(): ", conditionMessage(w))
            warn_triggered <<- TRUE
            invokeRestart("muffleWarning")  # suppress so execution continues
          }
        )
        
        # Force fallback if warning was raised or model is singular
        if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
          message("⚠️️ Fallback to lm() due to warning or singular fit")
          stop("Trigger fallback to lm")
        }
        
        mod_attempt  # return valid model if all checks passed
        
      }, error = function(e) {
        lm(simple_form_refit, data = df)
      })
      
      summary(model)
      
      #### FIG 2
      filename_hist = paste("FR_CV_WHOLE_histogram", ROI_sel, band_sel, "MI_bi.png", sep = "_")
      
      skew_chan = round(skewness(df$MI), 2)
      kurt_chan = round(kurtosis(df$MI), 2)
      
      png(
        filename = paste(outputdir_fig, filename_hist, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      hist(
        df$MI,
        breaks = 30,
        main   = "",          # leave main blank for now
        xlab   = "MI values",
        ylab   = "Resp",
        col    = "lightblue",
        border = "white"
      )
      
      title(
        main     = paste("FR_CV_WHOLE", ROI_sel, band_sel, "MI", "kurtosis:", kurt_chan, "skewness", skew_chan),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### FIG 3
      filename_qqplot = paste("FR_CV_WHOLE_qqplot", ROI_sel, band_sel, "MI_bi.png", sep = "_")
      
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
      
      filesxlsx_ROI = paste("FR_CV_WHOLE_MI_lmm", ROI_sel, band_sel, "res_bi.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
      
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
  }
}




######## ALLCOND WHOLE ########


# Load the Excel data
df_raw <- read_excel(paste(root, "df_MI_ALLCOND_filt_bi.xlsx", sep  = "/"))
df_raw <- df_raw %>%
  mutate(chan = paste0(sujet, "_", chan))
df_raw <- subset(df_raw, cond %in% c('RD_CV', 'RD_SV', 'RD_FV'))

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
      
      df_oneROI <- subset(df_raw, ROI == ROI_sel & band == band_sel)
      
      print(subset(df_oneROI, cond == 'RD_CV') %>% count(sujet))
      
      df <- df_oneROI[c("sujet", "chan", "MI", "resp", "cond")]
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$chan <- as.factor(df$chan)
      df$cond <- as.factor(df$cond)
      
      df$cond <- relevel(df$cond, ref = "RD_CV")
      
      #### FIG 1
      p <- ggplot(df, aes(x = sujet, y = MI, color = sujet, fill = sujet)) +
        geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                     position = position_dodge(.9)) +
        stat_summary(fun = median, geom = "point", size = 2,
                     position = position_dodge(.9), color = "white")+
        labs(
          title    = paste(ROI_sel, band_sel, "MI", sep = "_")
        ) +
        theme(
          plot.title    = element_text(hjust = 0.5),
        )
      
      p
      
      file_boxplot_subjectwise = paste("ALLCOND_WHOLE_boxplot", ROI_sel, band_sel, "MI_subjectwise_bi.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir_fig, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      #### MODEL
      #complex_form <- MI ~ resp + (resp | sujet/chan)
      #simple_form  <- MI ~ resp + (1 | sujet/chan)
      simple_form  <- MI ~ resp + (1 | sujet)
      simple_form_refit  <- MI ~ resp
      
      model <- tryCatch({
        
        warn_triggered <- FALSE  # will catch if any warning is raised
        
        mod_attempt <- withCallingHandlers(
          expr = {
            lmer(
              simple_form,
              data = df,
              control = lmerControl(optCtrl = list(maxfun = 2e5))
            )
          },
          warning = function(w) {
            message("⚠️ Warning during lmer(): ", conditionMessage(w))
            warn_triggered <<- TRUE
            invokeRestart("muffleWarning")  # suppress so execution continues
          }
        )
        
        # Force fallback if warning was raised or model is singular
        if (warn_triggered || isSingular(mod_attempt, tol = 1e-4)) {
          message("⚠️️ Fallback to lm() due to warning or singular fit")
          stop("Trigger fallback to lm")
        }
        
        mod_attempt  # return valid model if all checks passed
        
      }, error = function(e) {
        lm(simple_form_refit, data = df)
      })
      
      summary(model)
      
      #### FIG 2
      filename_hist = paste("ALLCOND_histogram", ROI_sel, band_sel, "MI_bi.png", sep = "_")
      
      skew_chan = round(skewness(df$MI), 2)
      kurt_chan = round(kurtosis(df$MI), 2)
      
      png(
        filename = paste(outputdir_fig, filename_hist, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      hist(
        df$MI,
        breaks = 30,
        main   = "",          # leave main blank for now
        xlab   = "MI values",
        ylab   = "Resp",
        col    = "lightblue",
        border = "white"
      )
      
      title(
        main     = paste("ALLCOND", ROI_sel, band_sel, "MI", "kurtosis:", kurt_chan, "skewness", skew_chan),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      #### FIG 3
      filename_qqplot = paste("ALLCOND_qqplot", ROI_sel, band_sel, "MI_bi.png", sep = "_")
      
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
      
      filesxlsx_ROI = paste("ALLCOND_MI_lmm", ROI_sel, band_sel, "res_bi.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir_df_lmm, filesxlsx_ROI, sep = "/"))
      
    }, warning = function(w) {
      message("Warning: ", conditionMessage(w))  # shows immediately
      invokeRestart("muffleWarning")
    })
    
  }
}



