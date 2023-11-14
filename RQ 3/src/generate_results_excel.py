import pandas as pd
import numpy as np
import itertools
import warnings, sys
warnings.filterwarnings('ignore')

def read_results_file(model, dataset, group):
    results_file = "RQ 3/results/{group}/subgroup_statistics_{model}_{dataset}.csv".format(model=model, dataset=dataset, group=group)
    df = pd.read_csv(results_file, index_col="Unnamed: 0")
    return df

def get_sd_res(alpha_gain, n_bins, min_support, df):
    cond_1 = (df["alpha_gain"] == alpha_gain)
    cond_2 = (df["n_bins"] == n_bins)
    cond_3 = (df["min_support"] == min_support)
    sub_df = df[cond_1 & cond_2 & cond_3]
    return sub_df

def generate_classification_accuracy_row(model, dataset, group, alpha_gain, n_bins, min_support):
    df = read_results_file(model, dataset, group)
    df = get_sd_res(alpha_gain, n_bins, min_support, df)
    df = df[["dataset", "model", "no_rules", "bin assignment accuracy overall", "coverage"]]
    to_return = {
        "dataset": dataset,
        "model": model,
        "no_rules": np.mean(df["no_rules"]),
        "no_rules std": np.std(df["no_rules"]),
        "bin assignment accuracy": np.mean(df["bin assignment accuracy overall"]),
        "bin assignment accuracy std": np.std(df["bin assignment accuracy overall"]),
        "coverage": np.mean(df["coverage"]),
        "coverage std": np.std(df["coverage"])
    }
    return to_return

def generate_classification_accuracy_tables(alpha_gain, n_bins, min_support, group, model, datasets):
    results = []
    for dataset in datasets:
        res = generate_classification_accuracy_row(model, dataset, group, alpha_gain, n_bins, min_support)
        results.append(res)
    results = pd.DataFrame(results)
    # display(results)
    return results


key = "pred"
def generate_2bin_accuracy_row(model, dataset, group, alpha_gain, min_support):
    df = read_results_file(model, dataset, group)
    n_bins = 2
    df = get_sd_res(alpha_gain, n_bins, min_support, df)
    df = df[["dataset", "model", "classsification accuracy 0 by {0}".format(key), "classsification accuracy 1 by {0}".format(key)]]
    to_return = {
        "dataset": dataset,
        "model": model,
        "bin 0 accuracy": np.mean(df["classsification accuracy 0 by {0}".format(key)]),
        "bin 0 accuracy std": np.std(df["classsification accuracy 0 by {0}".format(key)]),
        "bin 1 accuracy": np.mean(df["classsification accuracy 1 by {0}".format(key)]),
        "bin 1 accuracy std": np.std(df["classsification accuracy 1 by {0}".format(key)])
    }
    return to_return

def generate_2bin_accuracy_accuracy_tables(alpha_gain, min_support, group, model, datasets):
    results = []
    for dataset in datasets:
        res = generate_2bin_accuracy_row(model, dataset, group, alpha_gain, min_support)
        results.append(res)
    results = pd.DataFrame(results)
    # display(results)
    return results

def generate_3bin_accuracy_row(model, dataset, group, alpha_gain, min_support):
    df = read_results_file(model, dataset, group)
    n_bins=3
    df = get_sd_res(alpha_gain, n_bins, min_support, df)
    df = df[["dataset", "model", "classsification accuracy 0 by {0}".format(key), "classsification accuracy 1 by {0}".format(key), "classsification accuracy 2 by {0}".format(key)]]
    to_return = {
        "dataset": dataset,
        "model": model,
        "bin 0 accuracy": np.mean(df["classsification accuracy 0 by {0}".format(key)]),
        "bin 0 accuracy std": np.std(df["classsification accuracy 0 by {0}".format(key)]),
        "bin 1 accuracy": np.mean(df["classsification accuracy 1 by {0}".format(key)]),
        "bin 1 accuracy std": np.std(df["classsification accuracy 1 by {0}".format(key)]),
        "bin 2 accuracy": np.mean(df["classsification accuracy 2 by {0}".format(key)]),
        "bin 2 accuracy std": np.std(df["classsification accuracy 2 by {0}".format(key)])
        
    }
    return to_return

def generate_3bin_accuracy_accuracy_tables(alpha_gain, min_support, group, model, datasets):
    results = []
    for dataset in datasets:
        # print(model, dataset)
        try:
            res = generate_3bin_accuracy_row(model, dataset, group, alpha_gain, min_support)
        except:
            res = generate_2bin_accuracy_row(model, dataset, group, alpha_gain, min_support)
        results.append(res)
    results = pd.DataFrame(results)
    # display(results)
    return results

def generate_workbook(alpha_gain, min_support, group, model, datasets):
    wkbk = {}
    wkbk["2 bin sd metrics"] = generate_classification_accuracy_tables(alpha_gain=alpha_gain, n_bins=2, min_support=min_support, group=group, model=model, datasets=datasets)
    wkbk["3 bin sd metrics"] = generate_classification_accuracy_tables(alpha_gain=alpha_gain, n_bins=3, min_support=min_support, group=group, model=model, datasets=datasets)
    wkbk["2 bin accuracy table"] = generate_2bin_accuracy_accuracy_tables(alpha_gain=alpha_gain, min_support=min_support, group=group, model=model, datasets=datasets)
    wkbk["3 bin accuracy table"] = generate_3bin_accuracy_accuracy_tables(alpha_gain=alpha_gain, min_support=min_support, group=group, model=model, datasets=datasets)
    xlsx_fn = "RQ 3/results/{group}_{model}_alpha-gain{alpha_gain}_min-support{min_support}.xlsx".format(group=group, alpha_gain=alpha_gain, min_support=min_support, model=model)
    with pd.ExcelWriter(xlsx_fn) as writer:  
        for key in wkbk:
            df = wkbk[key]
            df.to_excel(writer, sheet_name=key)  

if __name__ == "__main__":  
    groups = ["dissertation" ]
    alpha_gains = [0, 0.5]
    min_support = [1]
    models = ["catboost-ve", "NN-dropout"]
    datasets = ["diabetes", "trauma_uk", "critical_outcome", "ED_3day_readmit", "hospitalization_prediction"]

    combos = list(itertools.product(groups, alpha_gains, min_support, models))
    for group, alpha_gain, min_support, model in combos:
        generate_workbook(alpha_gain, min_support, group, model, datasets)