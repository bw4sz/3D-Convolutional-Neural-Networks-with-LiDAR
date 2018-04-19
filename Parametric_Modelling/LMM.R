library(readr)
library(randomForest)

FUSION_Metrics <- read_csv("H:/Temp/convnet_3d/Leaf_On_Data/FUSION_Metrics.csv")
withheld <- read_csv("H:/Temp/convnet_3d/Leaf_On_Data/withheld.csv", col_names = FALSE)

FUSION_Metrics$Biomass_AG[is.na(FUSION_Metrics$Biomass_AG)] <- 0


FUSION_Metrics$SW_Percent[is.na(FUSION_Metrics$SW_Percent)] <- 0

files=strsplit(FUSION_Metrics$DataFile, '\\\\')
INVS=c()
for (i in 1:length(files)){
  INV=files[i][[1]][5]
  INVS=append(INVS, INV)
}

FUSION_Metrics$INVf=as.factor(INVS)

testF=FUSION_Metrics[as.integer(c(withheld[0:1000,1])$X1+1),]
validF=FUSION_Metrics[as.integer(c(withheld[1001:2000,1])$X1+1),]
trainF=FUSION_Metrics[-as.integer(c(withheld[,1])$X1+1),]

Biomass_Mod=lmer(Biomass_AG~Elev_SQRT_mean_SQ+Elev_P70+Elev_P40+Elev_P25+Elev_CURT_mean_CUBE+Elev_P30+Elev_P75+Elev_mean+Elev_maximum+Elev_P80 +(1 | INV), data=trainF)

Biomass_Mod_Final=lmer(formula = Biomass_AG ~ Elev_SQRT_mean_SQ + Elev_P40 + 
       Elev_P25 + Elev_CURT_mean_CUBE + Elev_mean + Elev_maximum + 
       Elev_P80 + (1 | INVf), data = trainF)

step(Biomass_Mod)

errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(Biomass_Mod_Final,newdata=testF[i,7:ncol(trainF)])
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



hmm=lmer(Biomass_AG~Elev_CURT_mean_CUBE+Elev_P30+Elev_P75+Elev_mean+Elev_maximum+ (1 | INV), data=trainF)


###############################################################################################
#top RF importances
#Percentage_all_returns_above_mean+Elev_stddev+Elev_P90+Elev_L_CV+Elev_CURT_mean_CUBE+Elev_P99+Percentage_first_returns_above_mean+Percentage_all_returns_above_2+Elev_variance+all_rtns_above_mean__frst_rtns+Elev_P10+All_returns_above_2+Elev_AAD+Elev_P60+Elev_P20+All_returns_above_mode+Elev_L_kurtosis+Elev_MAD_median+Elev_P01+First_returns_above_mean+Percentage_first_returns_above_mode

Tree_Count_Mod=lmer(Tree_Count~Percentage_all_above_mean+Elev_stddev+Elev_P90+Elev_L_CV+Elev_CURT_mean_CUBE+Elev_P99+Percentage_first_returns_above_mean+Percentage_all_abv+Elev_variance+All__Total_first_returns +(1 | INV), data=trainF)
step(Tree_Count_Mod)
Tree_Count_Mod_Final=lmer(formula = Tree_Count ~ Percentage_all_above_mean + 
                        Elev_stddev + Elev_P90 + Elev_L_CV + Percentage_first_returns_above_mean + 
                        Percentage_all_abv + Elev_variance + All__Total_first_returns + 
                        (1 | INV), data = trainF)



errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(Tree_Count_Mod_Final,newdata=testF[i,7:ncol(trainF)])
  trueF=testF$Tree_Count[i]
  errorF=append(errorF,(predF-trueF))
  preds=append(preds, predF)
  print(i)
}

RMSEf=sqrt(mean(errorF^2))
RMSEf_Perc=RMSEf/mean(testF$Tree_Count)
Bias=mean(errorF)
Bias_Perc=mean(errorF)/mean(testF$Tree_Count)
Abs_Bias=mean(abs(errorF))
R_squared=1-sum((testF$Tree_Count-preds)^2)/sum((testF$Tree_Count-mean(testF$Tree_Count))^2)

###############################################################################################
#Percentage_all_returns_above_mean+Total_all_returns+Percentage_all_returns_above_2+Percentage_first_returns_above_mean+Elev_skewness+Total_return_count+Percentage_first_returns_above_2+Total_first_returns+Elev_MAD_median+Elev_AAD+all_rtns_above_mean__frst_rtns+Return_1_count+Elev_P60+Elev_L4+Elev_L_skewness+Elev_P75+Elev_P40+Elev_P20+Elev_P80+Elev_MAD_mode+Elev_mean+Elev_P70+Elev_SQRT_mean_SQ
SW_Percent_Mod=glmer(SW_Percent~Percentage_all_above_mean+Percentage_all_abv+Percentage_first_returns_above_mean+Elev_skewness+Percentage_first_abv+Elev_MAD_median+Elev_AAD+All__Total_first_returns+Elev_P60 +(1 | INV), data=trainF, family="binomial")
step(SW_Percent_Mod)
SW_Percent_Mod_Final=glmer(SW_Percent~Percentage_all_above_mean+Percentage_all_abv+Elev_skewness+Elev_MAD_median+Elev_AAD+(1 | INV), data=trainF, family="binomial",control=glmerControl(optimizer="bobyqa",check.conv.grad = .makeCC("warning", tol = 1e-2)))



errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(SW_Percent_Mod_Final,newdata=testF[i,7:ncol(trainF)], type = "response")
  trueF=testF$SW_Percent[i]
  errorF=append(errorF,(predF-trueF))
  preds=append(preds, predF)
  print(i)
}

RMSEf=sqrt(mean(errorF^2))
RMSEf_Perc=RMSEf/mean(testF$SW_Percent)
Bias=mean(errorF)
Bias_Perc=mean(errorF)/mean(testF$SW_Percent)
Abs_Bias=mean(abs(errorF))
R_squared=1-sum((testF$SW_Percent-preds)^2)/sum((testF$SW_Percent-mean(testF$SW_Percent))^2)

###########################################################################################

Biomass_Mod=randomForest(Biomass_AG ~ Elev_SQRT_mean_SQ+Elev_P70+Elev_P40+Elev_P25+Elev_CURT_mean_CUBE+Elev_L1+Elev_P30+Elev_P75+Elev_mean+Elev_maximum+Elev_P80+Elev_P60+Elev_P90+Elev_P20+Percentage_all_abv+Elev_P05+Elev_P01+Percentage_first_abv+Elev_mode+Elev_L_skewness+Elev_P10+Elev_variance+Percentage_all_above_mode+Elev_P99+Elev_kurtosis+All__Total_first_returns+All_returns_mode_Total_returns+Elev_L_CV+Elev_AAD+Elev_skewness+Percentage_all_above_mean+Percentage_first_returns_above_mode+Elev_MAD_median+Elev_MAD_mode+Elev_L_kurtosis+Elev_L4+Elev_stddev,
                         data=trainF,importance=TRUE, ntree=1000)

Biomass_Mod2=cforest(Biomass_AG ~ Elev_SQRT_mean_SQ+Elev_P70+Elev_P40+Elev_P25+Elev_CURT_mean_CUBE+Elev_L1+Elev_P30+Elev_P75+Elev_mean+Elev_maximum+Elev_P80+Elev_P60+Elev_P90+Elev_P20+Percentage_all_abv+Elev_P05+Elev_P01+Percentage_first_abv+Elev_mode+Elev_L_skewness+Elev_P10+Elev_variance+Percentage_all_above_mode+Elev_P99+Elev_kurtosis+All__Total_first_returns+All_returns_mode_Total_returns+Elev_L_CV+Elev_AAD+Elev_skewness+Percentage_all_above_mean+Percentage_first_returns_above_mode+Elev_MAD_median+Elev_MAD_mode+Elev_L_kurtosis+Elev_L4+Elev_stddev,
                     data=trainF, controls= cforest_control(mtry=round((ncol(trainF)-7)/3), ntree=1000, trace=TRUE))

errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(Biomass_Mod_F,newdata=testF[i,7:ncol(trainF)])
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

###########################################################################################

Tree_Count_Mod=randomForest(Tree_Count ~ Percentage_all_above_mean+Elev_stddev+Elev_P90+Elev_L_CV+Elev_CURT_mean_CUBE+Elev_P99+Percentage_first_returns_above_mean+Percentage_all_abv+Elev_variance+All__Total_first_returns+Elev_P10+Elev_AAD+Elev_P60+Elev_P20+Elev_L_kurtosis+Elev_MAD_median+Elev_P01+Percentage_first_returns_above_mode+Elev_P30+Elev_P70+Elev_MAD_mode+Elev_CV+Canopy_relief_ratio+Elev_kurtosis+Elev_L4+Percentage_all_above_mode+Percentage_first_abv+Elev_P40+Elev_skewness+Elev_P95+Elev_P05+Elev_P80+Elev_SQRT_mean_SQ+Elev_mean+Elev_L1+Elev_L_skewness,
                         data=trainF,importance=TRUE, ntree=1000)

errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(Tree_Count_Mod,newdata=testF[i,7:ncol(trainF)])
  trueF=testF$Tree_Count[i]
  errorF=append(errorF,(predF-trueF))
  preds=append(preds, predF)
  print(i)
}

RMSEf=sqrt(mean(errorF^2))
RMSEf_Perc=RMSEf/mean(testF$Tree_Count)
Bias=mean(errorF)
Bias_Perc=mean(errorF)/mean(testF$Tree_Count)
Abs_Bias=mean(abs(errorF))
R_squared=1-sum((testF$Tree_Count-preds)^2)/sum((testF$Tree_Count-mean(testF$Tree_Count))^2)

###########################################################################################

SW_Percent_Mod=randomForest(as.formula((paste('SW_Percent','~',paste(cov,collapse='+')))),
                            data=trainF,importance=TRUE, ntree=1000)


errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(SW_Percent_Mod,newdata=testF[i,7:ncol(trainF)])
  trueF=testF$SW_Percent[i]
  errorF=append(errorF,(predF-trueF))
  preds=append(preds, predF)
  print(i)
}

RMSEf=sqrt(mean(errorF^2))
RMSEf_Perc=RMSEf/mean(testF$SW_Percent)
Bias=mean(errorF)
Bias_Perc=mean(errorF)/mean(testF$SW_Percent)
Abs_Bias=mean(abs(errorF))
R_squared=1-sum((testF$SW_Percent-preds)^2)/sum((testF$SW_Percent-mean(testF$SW_Percent))^2)

################################################################################################
FUSION_Metrics <- read_csv("H:/Temp/convnet_3d/Leaf_On_Data/Rlidar_Metrics_nocounts.csv")
withheld <- read_csv("H:/Temp/convnet_3d/Leaf_On_Data/withheld.csv", col_names = FALSE)

FUSION_Metrics$Biomass_AG[is.na(FUSION_Metrics$Biomass_AG)] <- 0
FUSION_Metrics$SW_Percent[is.na(FUSION_Metrics$SW_Percent)] <- 0

testF=FUSION_Metrics[as.integer(c(withheld[0:1000,1])$X1+1),]
validF=FUSION_Metrics[as.integer(c(withheld[1001:2000,1])$X1+1),]
trainF=FUSION_Metrics[-as.integer(c(withheld[,1])$X1+1),]

Biomass_Mod=randomForest(as.formula((paste('Biomass_AG','~',paste(colnames(FUSION_Metrics)[5:ncol(FUSION_Metrics)],collapse='+')))),  data=trainF,importance=TRUE, ntree=1000)

##Biomass_Mod=randomForest(Biomass_AG ~ Elev_SQRT_mean_SQ+Elev_P70+Elev_P40+Elev_P25+Elev_CURT_mean_CUBE+Elev_L1+Elev_P30+Elev_P75+Elev_mean+Elev_maximum+Elev_P80+Elev_P60+Elev_P90+Elev_P20+Percentage_all_abv+Elev_P05+Elev_P01+Percentage_first_abv+Elev_mode+Elev_L_skewness+Elev_P10+Elev_variance+Percentage_all_above_mode+Elev_P99+Elev_kurtosis+All__Total_first_returns+All_returns_mode_Total_returns+Elev_L_CV+Elev_AAD+Elev_skewness+Percentage_all_above_mean+Percentage_first_returns_above_mode+Elev_MAD_median+Elev_MAD_mode+Elev_L_kurtosis+Elev_L4+Elev_stddev,
#                         data=trainF,importance=TRUE, ntree=1000)



errorF=c()
preds=c()
for (i in 1:nrow(testF)){
  #predF=exp(predict(Tree_Count_Mod_F,testF[i,11:65]))
  predF=predict(Biomass_Mod,newdata=testF[i,5:ncol(trainF)])
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