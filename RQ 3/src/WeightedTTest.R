library(broom)
library(weights)
library(readr)

weighted_ttest <- function(dataset, model, res_key){
    fn = paste("RQ 3/results/", res_key, "/R_input_",model, "_",dataset,".csv", sep="")
    df <- read_csv(fn)
    x0 = df$subgroup_0_accuracy
    #x0 = df["subgroup_0_accuracy"]
    x1 = df$subgroup_1_accuracy
    w0 = df$subgroup_0_coverage
    w1 = df$subgroup_1_coverage
    res = wtd.t.test(x=x0,
             y=x1,
             weight=w0,
             weighty=w1,
             samedata=FALSE,
             alternative="greater",
             mean1=FALSE)
    chars <- capture.output(print(res))
    output_file = paste("RQ 3/results/", res_key, "/R_output_",model, "_",dataset,".txt", sep="")
    writeLines(chars, con = file(output_file))

    }
args = commandArgs(trailingOnly=TRUE)
print(args[1])

weighted_ttest("trauma_uk", "NN-dropout", args[1])
weighted_ttest("trauma_uk", "catboost-ve", args[1])
weighted_ttest("diabetes", "NN-dropout", args[1])
weighted_ttest("diabetes", "catboost-ve", args[1])
weighted_ttest("critical_outcome", "NN-dropout", args[1])
weighted_ttest("critical_outcome", "catboost-ve", args[1])
weighted_ttest("ED_3day_readmit", "NN-dropout", args[1])
weighted_ttest("ED_3day_readmit", "catboost-ve", args[1])
weighted_ttest("hospitalization_prediction", "NN-dropout", args[1])
weighted_ttest("hospitalization_prediction", "catboost-ve", args[1])