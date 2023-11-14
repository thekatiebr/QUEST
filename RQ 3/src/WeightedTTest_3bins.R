library(broom)
library(weights)
library(readr)

weighted_ttest <- function(dataset, model, res_key){
    fn = paste("RQ 3/results/", res_key, "/R_input_",model, "_",dataset,"_3bins.csv", sep="")
    df <- read_csv(fn)
    print(df)
    # vector[!is.na(vector)]
    x0 = df$subgroup_0_accuracy
    x1 = df$subgroup_1_accuracy
    x2 = df$subgroup_2_accuracy
    w0 = df$subgroup_0_coverage
    w1 = df$subgroup_1_coverage
    w2 = df$subgroup_2_coverage
    # x0 = x0[!is.na(x0)]
    # x1 = x1[!is.na(x1)]
    # x2 = x2[!is.na(x2)]
    # w0 = w0[!is.na(w0)]
    # w1 = w1[!is.na(w1)]
    # w2 = w2[!is.na(w2)]
    print(colnames(df))
    print(x0)
    print(x1)
    print(x2)
    print(w0)
    print(w1)
    print(w2)
    res <- wtd.t.test(x=x0,
             y=x1,
             weight=w0,
             weighty=w1,
             samedata=FALSE,
             alternative="greater",
             mean1=FALSE)
    chars <- capture.output(print(res))
    output_file = paste("RQ 3/results/", res_key, "/R_output_",model, "_",dataset,"_0v1.txt", sep="")
    writeLines(chars, con = file(output_file))
    
    res <- wtd.t.test(x=x1,
             y=x2,
             weight=w1,
             weighty=w2,
             samedata=FALSE,
             alternative="greater",
             mean1=FALSE)
    
    chars <- capture.output(print(res))
    output_file = paste("RQ 3/results/", res_key, "/R_output_",model, "_",dataset,"_1v2.txt", sep="")
    writeLines(chars, con = file(output_file))
    
    
    res <- wtd.t.test(x=x0,
             y=x2,
             weight=w0,
             weighty=w2,
             samedata=FALSE,
             alternative="greater",
             mean1=FALSE)
    
    chars <- capture.output(print(res))
    output_file = paste("RQ 3/results/", res_key, "/R_output_",model, "_",dataset,"_0v2.txt", sep="")
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