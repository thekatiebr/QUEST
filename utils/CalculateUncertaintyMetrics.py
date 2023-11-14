import os, sys
sys.path.append("..")
from metrics import *
import pandas as pd

dataset = sys.argv[1] 
methods = ["NN-dropout", "catboost-ve", "RF-naive", "RF-dropout"]

# method = sys.argv[2] if len(sys.argv) > 3 else "NN-dropout"
for method in methods:
    print(method)
    i = 0 #for now, this will be built into a loop
    fn_uq = "results/{1}/{2}/rejection-classification_{0}_{1}.csv".format(method, dataset, i)
    fn_rc = "results/{1}/{2}/rejection-classification_{0}_rnd-cntl_{1}.csv".format(method, dataset, i)
    # "input/{1}/{2}/uncertainty-info_{0}_{1}.csv".format(method[1], dataset, i)

    uq_data = pd.read_csv(fn_uq, index_col="Unnamed: 0")
    rc_data = pd.read_csv(fn_rc, index_col="Unnamed: 0")
    uq = pd.read_csv("input/{1}/{2}/uncertainty-info_{0}_{1}.csv".format(method, dataset, i))

    uq.sort_values(by="uncertainty", ascending=True, inplace=True)
    model_metrics = compute_metrics(uq["p(positive class)"], uq["truth"])
    base_accuracy = model_metrics["accuracy"]
    print("Model Accuracy: ", model_metrics["accuracy"])
    area, new_acc = numerical_integration_rejection_classification(accuracy=uq_data["accuracy"],
                                                   percent_removed=uq_data["% most uncertain removed"],
                                                   base_accuracy=base_accuracy)
    print("AURCC (uq): ", area)
    uq_data["accuracy to plot"] = new_acc

    area, new_acc = numerical_integration_rejection_classification(accuracy=rc_data["accuracy"],
                                                   percent_removed=rc_data["% most uncertain removed"],
                                                   base_accuracy=base_accuracy)
    print("AURCC (cntl): ", area)
    rc_data["accuracy to plot"] = new_acc

    # print("AULC: ", relative_area_under_lift_curve(uq))

    plot_calibration(X_pr_unc=uq["uncertainty"], y_cls_unc=uq["class uncertainty"], method=method, dataset=dataset)
    plot_calibration_rate_correct(X_pr_unc=uq["uncertainty"], y_cls_pro=uq["rate corrected predicted"], truth=uq["truth"], method=method, dataset=dataset)

    plot_rejection_classification(unc_results=uq_data, rnd_results=rc_data, method_nm=method, dataset=dataset)