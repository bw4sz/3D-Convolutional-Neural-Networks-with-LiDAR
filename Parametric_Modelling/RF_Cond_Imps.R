library(party,  lib="/data/home/eayrey/R")
library(readr,  lib="/data/home/eayrey/R")

FUSION_Metrics <- read_csv("H:/Temp/convnet_3d/Leaf_On_Data/FUSION_Metrics.csv")
withheld <- read_csv("H:/Temp/convnet_3d/Leaf_On_Data/withheld.csv", col_names = FALSE)

FUSION_Metrics$Biomass_AG[is.na(FUSION_Metrics$Biomass_AG)] <- 0

testF=FUSION_Metrics[as.integer(c(withheld[0:1000,1])$X1+1),]
validF=FUSION_Metrics[as.integer(c(withheld[1001:2000,1])$X1+1),]
trainF=FUSION_Metrics[-as.integer(c(withheld[,1])$X1+1),]

trainF=trainF[sample(nrow(trainF),350),]

X='Biomass_AG'

cov=colnames(trainF)[7:length(colnames(trainF))]

Biomass_Mod=cforest(as.formula((paste(X,'~',paste(cov,collapse='+')))), data=trainF, controls= cforest_control(mtry=round(length(cov)/3), ntree=25, trace=TRUE))

errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(Biomass_Mod,newdata=testF[i,7:length(colnames(trainF))])
  trueF=testF$Biomass_AG[i]
  errorF=append(errorF,(predF-trueF))
  preds=append(preds, predF)
  print(i)
}

RMSEf=sqrt(mean(errorF^2))
RMSEf_Perc=RMSEf/mean(testF$Biomass_AG)
Bias=mean(errorF)
Bias_Perc=mean(errorF)/mean(testF$Biomass_AG)
Abs_Bias=mean(abs(errorF))
R_squared=1-sum((testF$Biomass_AG-preds)^2)/sum((testF$Biomass_AG-mean(testF$Biomass_AG))^2)

X='Biomass_AG'
importance=varimp(Biomass_Mod, conditional = TRUE, threshold=.65)
importance=stack(importance)
colnames(importance)=c("imp", "name")
importance= importance[order(-importance[,1]),]
bads=tail(importance, round(nrow(importance)*.15))
cov=as.character(importance$name)

all_metricsF=c(R_squared, RMSEf,RMSEf_Perc,Bias,Bias_Perc,Abs_Bias,paste(X,'~',paste(cov,collapse='+')) )
print(paste(RMSEf, 'starting iterations...'))
while (length(cov)>5){
  cov=as.character(importance[!importance[,1] %in% bads[,1],]$name)
  Biomass_Mod_F2=cforest(as.formula((paste(X,'~',paste(cov,collapse='+')))),
                         data=trainF, controls= cforest_control(mtry=round(length(cov)/3), ntree=50, trace=TRUE))
  
  errorF=c()
  predsF=c()
  for (i in 1:nrow(testF)){
    predF=predict(Biomass_Mod_F2,newdata=testF[i,7:length(colnames(trainF))])
    trueF=testF$Biomass_AG[i]
    errorF=append(errorF,(predF-trueF))
    predsF=append(predsF, predF)
  }
  #RMSE
  rmse=sqrt(mean(errorF^2))
  #RMSE Perc
  rmse_perc=rmse/mean(testF$Biomass_AG)
  #Bias
  bias=mean(errorF)
  #Perc_Bias
  perc_bias=mean(errorF)/mean(testF$Biomass_AG)
  
  R_squared=1-sum((testF$Biomass_AG-predsF)^2)/sum((testF$Biomass_AG-mean(testF$Biomass_AG))^2)
  #MAE
  mae=mean(abs(errorF))
  all_metricsF=rbind(all_metricsF, c(R_squared,rmse,rmse_perc,bias,perc_bias,mae, paste(X,'~',paste(cov,collapse='+'))))
  
  importance=varimp(Biomass_Mod_F2, conditional = TRUE, threshold=.65)
  importance=stack(importance)
  colnames(importance)=c("imp", "name")
  importance= importance[order(-importance[,1]),]
  
  number=ifelse(round(nrow(importance)/10)>0,round(nrow(importance)/10),1)
  bads=tail(importance, number)
  print(paste(rmse ,paste(X,'~',paste(cov,collapse='+'))))
}
colnames(metrics)=c("r2", "RMSE", "RMSE_Perc", "bias", "bias_Perc", "MAE", "covs")



#######################################################################################################################






TC_Mod_F=cforest(SW_Percent ~ Percentage_all_returns_above_mean+Percentage_all_returns_above_2+Total_return_count+Total_all_returns+Elev_variance+Return_1_count+Elev_P70+Elev_AAD+Canopy_relief_ratio+all_rtns_above_mean__frst_rtns+Elev_stddev+Elev_P10+Total_first_returns+First_returns_above_mode+Elev_P05+Elev_L_kurtosis+All_returns_above_mean+Elev_skewness+Elev_L_skewness+Elev_MAD_median+Elev_CV+Elev_L_CV+Percentage_all_returns_above_mode+Elev_P40+All_returns_above_mode+Elev_P20+Percentage_first_returns_above_mean+Percentage_first_returns_above_2+Elev_L4+Elev_mean+Elev_MAD_mode+Elev_P60+Elev_P25+Elev_kurtosis+Elev_L1+Elev_P30+Elev_P80+Elev_P90+Elev_P95+Elev_SQRT_mean_SQ+Elev_P75+Elev_mode+Elev_P99+All_returns_above_2+Percentage_first_returns_above_mode+all_rtns_above_mode__frst_rtns+First_returns_above_2+First_returns_above_mean+Elev_P01+Elev_CURT_mean_CUBE+Elev_maximum+Elev_IQ,
                      data=trainF,importance=TRUE, ntree=1000)

errorsF=c()
preds=c()
for (i in 1:nrow(testF)){
  predF=predict(TC_Mod_F,newdata=testF[i,6:53])
  trueF=testF$Tree_Count[i]
  errorF=append(errorF,(predF-trueF))
  preds=append(preds, predF)
}

RMSEf=sqrt(mean(errorF^2))
RMSEf_Perc=RMSEf/mean(testF$Tree_Count)
Bias=mean(errorF)
Bias_Perc=mean(errorF)/mean(testF$Tree_Count)
Abs_Bias=mean(abs(errorF))
R_squared=1-sum((testF$Tree_Count-preds)^2)/sum((testF$Tree_Count-mean(testF$Tree_Count))^2)

#Biomass_Mod_F=randomForest(Biomass_AG ~ total_mean+Percentile_20+Perc_Above_15m+Perc_Above_20m+heightB_90+countB_P20+Percentile_40+Percentile_60+heightB_max+basal_area_in+mean_top_ht+heightB_75+rugoseL_mean+total_median+SIMH+Percentile_80+scan_angle_sd+Perc_Above_5m+mean_ht+Percentile_95+mean_obscured+skew_dists_NB+heightLM_mean+pt2WSedge_max+SD_crown_area_B+total_sd+height_75+mean_pointy+sd_residuals+height_max+biomass_tot+height_mean+heightB_25+basal_area_tot+perc_in_tally+max_crown_area_B+crown_volume_tot+heightB_mean+biomass_in+prod+heightB_med+Perc_Above_P80+encroaching_tree_count+Perc_Above_25m+count_P10+countB_P10+tot_top_ht+Perc_Above_10m+Perc_Above_P60+rugoseL_sd+kurt_dists_B+mean_growing_spaceNB+csr+countLM_P10+rugose_WS_perc+out_in_ratioC+sd_off_cent+mean_top_positivity+Perc_Above_P95+height_25+rugoseS_sd+sd_growing_spaceNB+mean_off_cent+total_range+Perc_Above_P20+Perc_Above_P40+Perc_Above_35m+kurt_dists_NB+mean_dists_B+SD_dists_B+max_growing_space+pt2Sedge_mean+GINIH+out_in_ratioH+mean_crown_area_B+watershed_count_inplot+mean_crown_area_NB+encroach_area+crown_volume_in+mean_residuals+mean_positivity+mean_raw_pointy+pt2WSedge_sd+countB_P30+countB+height_sd+count_P30+countLM_P30+sd_shadowed+sky_view_areas_m+neighbor_top_dist_m+rugoseS_count+softwoodyness+heightB_min+cell_ppm+sky_view_areas_sd+count_P20+pt2WSedge_mean+pt2Sedge_max+heightB_sd+max_crown_area_NB+sd_obscured+countLM+SD_dists_NB+rugoseS_mean+neighbor_top_dist_sd+sd_positivity,
#                           data=trainF,importance=TRUE, ntree=1000)
