import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import sys, itertools, traceback
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
res_key = sys.argv[1]
def calculate_metrics_by_bin_prediction(df, n_bins):
    metrics = {}
    for bin_ in range(n_bins):
        sub_df = df[df["uncertainty_bin_assignments"] == bin_]
        classification_true = sub_df["truth"]
        classification_pred = sub_df["p(positive class)"] >= 0.5
        bin_assignment_true = sub_df["true_uncertainty_bin_assignments"]
        bin_assignment_pred = sub_df["uncertainty_bin_assignments"]
        metrics["bin assignment accuracy {0} by pred".format(bin_)] = accuracy_score(bin_assignment_true, bin_assignment_pred)
        metrics["classsification accuracy {0} by pred".format(bin_)] = accuracy_score(classification_true, classification_pred)
    # print(metrics)
    return metrics

def calcuate_metrics_by_bin_majority(df):
    subgroups = np.unique(df["subgroup_assignment"])
    keys = []
    metrics = {}
    
    for subgroup in subgroups:
        sub_df = df[df["subgroup_assignment"] == subgroup]
        classification_true = sub_df["truth"]
        classification_pred = sub_df["p(positive class)"] >= 0.5
        bin_assignment_true = sub_df["true_uncertainty_bin_assignments"]
        bin_assignment_pred = sub_df["uncertainty_bin_assignments"]

        uq_assigned = sub_df["uncertainty_bin_assignments"]
        unique, counts = np.unique(uq_assigned, return_counts=True)
        max_index = np.argmax(counts)
        uq_assigned_keys = unique[max_index]

        if uq_assigned_keys in keys:

            metrics["bin assignment accuracy {0} by majority".format(uq_assigned_keys)].append(accuracy_score(bin_assignment_true, bin_assignment_pred))
            metrics["classsification accuracy {0} by majority".format(uq_assigned_keys)].append(accuracy_score(classification_true, classification_pred))
        else:
            keys.append(uq_assigned_keys)
            metrics["bin assignment accuracy {0} by majority".format(uq_assigned_keys)] = []
            metrics["bin assignment accuracy {0} by majority".format(uq_assigned_keys)].append(accuracy_score(bin_assignment_true, bin_assignment_pred))
            metrics["classsification accuracy {0} by majority".format(uq_assigned_keys)] = []
            metrics["classsification accuracy {0} by majority".format(uq_assigned_keys)].append(accuracy_score(classification_true, classification_pred))
    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])
    s_k = list(metrics.keys())
    s_k.sort()
    new_metrics = {i: metrics[i] for i in s_k}
    # print(new_metrics)
    return new_metrics


def calcuate_metrics_by_bin(df):
    subgroups = np.unique(df["subgroup_assignment"])
    keys = []
    metrics = {}
    
    for subgroup in subgroups:
        sub_df = df[df["subgroup_assignment"] == subgroup]
        classification_true = sub_df["truth"]
        classification_pred = sub_df["p(positive class)"] >= 0.5
        bin_assignment_true = sub_df["true_uncertainty_bin_assignments"]
        bin_assignment_pred = sub_df["uncertainty_bin_assignments"]

        uq_assigned = sub_df["uncertainty_bin_assignments"]
        uq_assigned_keys = str(np.sort(np.unique(uq_assigned)))
        print(uq_assigned_keys)
        if uq_assigned_keys in keys:

            metrics["bin assignment accuracy {0}".format(uq_assigned_keys)].append(accuracy_score(bin_assignment_true, bin_assignment_pred))
            metrics["classsification accuracy {0}".format(uq_assigned_keys)].append(accuracy_score(classification_true, classification_pred))
        else:
            keys.append(uq_assigned_keys)
            metrics["bin assignment accuracy {0}".format(uq_assigned_keys)] = []
            metrics["bin assignment accuracy {0}".format(uq_assigned_keys)].append(accuracy_score(bin_assignment_true, bin_assignment_pred))
            metrics["classsification accuracy {0}".format(uq_assigned_keys)] = []
            metrics["classsification accuracy {0}".format(uq_assigned_keys)].append(accuracy_score(classification_true, classification_pred))
    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])
    

    return metrics

def calculate_accuracy_by_subgroup(df, subgroup):
    sub_df = df[df["subgroup_assignment"] == subgroup]
    true = sub_df["truth"]
    pred = sub_df["p(positive class)"] >= 0.5
    print(true, pred)
    return {"subgroup {0} classsification accuracy".format(subgroup): accuracy_score(true, pred)}

# def calculate_uncertainty_()

def calculate_uncertainty_statistics(df, subgroup):
    sub_df = df[df["subgroup_assignment"] == subgroup]
    min_ = np.min(sub_df["uncertainty"])
    max_ = np.max(sub_df["uncertainty"])
    avg_ = np.mean(sub_df["uncertainty"])
    range_ = max_ - min_
    sub_df_uq_bin_assignment = sub_df["uncertainty_bin_assignments"]
    sub_df_true_bins = sub_df["true_uncertainty_bin_assignments"]
    uncertainty_bin_acc = accuracy_score(sub_df_true_bins, sub_df_uq_bin_assignment)
    return {
            "subgroup {0} size".format(subgroup): sub_df.shape[0],
            "subgroup {0} min uncertainty".format(subgroup): min_, 
            "subgroup {0} average uncertainty".format(subgroup): avg_, 
            "subgroup {0} max uncertainty".format(subgroup): max_,
            "subgroup {0} uncertainty range".format(subgroup): range_,
            "subgroup {0} uncertainty assignment accuracy".format(subgroup): uncertainty_bin_acc}

def calculate_bin_statistics(df, subgroup, n_bins):
    if isinstance(n_bins, int):
        sub_df = df[df["subgroup_assignment"] == subgroup]
        grps ,dist = np.unique(sub_df["uncertainty_bin_assignments"],return_counts=True)
        dist = dict(zip(grps,dist))

        true_grps ,true_dist = np.unique(sub_df["true_uncertainty_bin_assignments"],return_counts=True)
        true_dist = dict(zip(true_grps ,true_dist))

        return {
                "subgroup {0} uncertainty bin distribution".format(subgroup): dist,
                "subgroup {0} true uncertainty bin distribution".format(subgroup): true_dist}
    else:
        return {}

def calculate_bin_assignment_metrics(df):
    bin_assignment_true = df["true_uncertainty_bin_assignments"]
    bin_assignment_pred = df["uncertainty_bin_assignments"]
    return {"bin assignment accuracy overall": accuracy_score(bin_assignment_true, bin_assignment_pred)}

def calculate_statistics(df, subgroup, n_bins):
    res = {}
    res.update(calculate_uncertainty_statistics(df, subgroup))
    res.update(calculate_accuracy_by_subgroup(df, subgroup))
    res.update(calculate_bin_statistics(df, subgroup, n_bins))
    return res

def stats_driver(dataset, model, train_test, iteration, max_rules, min_support, alpha_gain, n_bins):
    print(dataset, model, train_test, iteration, max_rules, min_support, alpha_gain, n_bins)
    id_ = {
        "continuous": "subgroup_assignments_{model}_{dataset}_{train_test}_{iteration}_{max_rules}_{min_support}_{alpha_gain}_continuous".format(model=model,
                                                                                                                                                dataset=dataset, 
                                                                                                                                                train_test=train_test,
                                                                                                                                                iteration=iteration,
                                                                                                                                                max_rules=max_rules,
                                                                                                                                                min_support=min_support,
                                                                                                                                                alpha_gain=alpha_gain),
        "categorical": "subgroup_assignments_{model}_{dataset}_{train_test}_{iteration}_{max_rules}_{min_support}_{alpha_gain}_bins{bins}".format(model=model, 
                                                                                                                                                dataset=dataset, 
                                                                                                                                                train_test=train_test,
                                                                                                                                                iteration=iteration,
                                                                                                                                                max_rules=max_rules,
                                                                                                                                                min_support=min_support,
                                                                                                                                                alpha_gain=alpha_gain,
                                                                                                                                                bins=n_bins)
    }
    fn_id = id_["categorical"] if isinstance(n_bins, int) else id_["continuous"]
    filename = "RQ 3/input/{1}/{0}.csv".format(fn_id, res_key)
    df = pd.read_csv(filename)
    ### remove later
    df = df.dropna()
    #####
    subgroups = np.unique(df["subgroup_assignment"])
    results = {
                "dataset": dataset,
                "model": model,
                "train_test": train_test,
                "iteration": iteration,
                "max_rules": max_rules,
                "min_support": min_support,
                "alpha_gain": alpha_gain,
                "n_bins":n_bins}
    results["no_rules"] = get_num_subgroups(df, model, dataset, iteration, min_support, alpha_gain, n_bins)
    results["coverage"] = get_coverage(df, model, dataset, iteration, min_support, alpha_gain, n_bins)
    results.update(calculate_bin_assignment_metrics(df))
    for subgroup in subgroups:
        res = calculate_statistics(df, subgroup, n_bins)
        results.update(res)

    # results.update(calcuate_metrics_by_bin(df))
    # results.update(calcuate_metrics_by_bin_majority(df))
    results.update(calculate_metrics_by_bin_prediction(df, n_bins))
    return results
    # df_res = pd.DataFrame([results]).transpose()
    # print(df_res)
    # df_res.to_csv("RQ 3/results/subgroup_statistics_{0}.csv".format(fn_id))


def sort_keys(df):
    not_to_sort = ["dataset", "model", "train_test", "iteration", "max_rules", "min_support", "alpha_gain", "n_bins"]
    keys = list(df)
    for key in not_to_sort: 
        while key in keys: 
            keys.remove(key)
    keys.sort()
    all_keys = not_to_sort
    all_keys.extend(keys)
    return df[keys]


def get_num_subgroups(df, model, dataset, iteration, min_support, alpha_gain, bins):
    ds_df = pd.read_csv("RQ 3/results/{res_key}/subgroup-discovery_{model}_{dataset}_categorical.csv".format(res_key = res_key, model=model, dataset=dataset))
    v1 = (ds_df["model"] == model)
    v2 = (ds_df["dataset"] == dataset)
    v3 = (ds_df["iteration"] == iteration)
    v4 = (ds_df["min_support"] == min_support)
    v5 = (ds_df["alpha_gain"] == alpha_gain)
    v6 = (ds_df["n_bins"] == bins)
    val =  v1 & v2 & v3 & v4 & v5 & v6
    r = ds_df[val]
    to_return = r["no_rules"].values[0]
    
    return to_return

def get_coverage(df, model, dataset, iteration, min_support, alpha_gain, bins):
    ds_df = pd.read_csv("RQ 3/results/{res_key}/subgroup-discovery_{model}_{dataset}_categorical.csv".format(res_key = res_key, model=model, dataset=dataset))
    v1 = (ds_df["model"] == model)
    v2 = (ds_df["dataset"] == dataset)
    v3 = (ds_df["iteration"] == iteration)
    v4 = (ds_df["min_support"] == min_support)
    v5 = (ds_df["alpha_gain"] == alpha_gain)
    v6 = (ds_df["n_bins"] == bins)
    val =  v1 & v2 & v3 & v4 & v5 & v6
    r = ds_df[val]
    to_return = r["coverage"].values[0]
    
    return to_return


def stats_for_dataset(model, dataset):
    iterations = [0,1,2,3,4,5,6,7,8,9]
    n_bins_list = [2, 3, 4, 5] 
    # n_bins_list = [2, 3, 4, 5, "continuous"] 
    max_rules_list = [np.inf]
    min_supports_lst = [ 1, 10, 100, 1000]
    alpha_gains = [0.0, 0.5, 1.0]

    combos = list(itertools.product(iterations, min_supports_lst, alpha_gains, n_bins_list))
    all_results = []
    for iteration, min_support, alpha_gain, n_bins in combos:    
        train_test = "test"
        max_rules = "inf"
        try: 
            result = stats_driver(dataset, model, train_test, iteration, max_rules, min_support, alpha_gain, n_bins)
            all_results.append(result)
        except Exception as e: 
            all_results.append({
                                "dataset": dataset,
                                "model": model,
                                "train_test": train_test,
                                "iteration": iteration,
                                "max_rules": max_rules,
                                "min_support": min_support,
                                "alpha_gain": alpha_gain,
                                "n_bins":n_bins})
            print(e)

    df = pd.DataFrame(all_results)
    sort_keys(df)

    df.to_csv("RQ 3/results/{res_key}/subgroup_statistics_{model}_{dataset}.csv".format(res_key = res_key, model=model, dataset=dataset))

if __name__ == "__main__":
    datasets = ["diabetes", "trauma_uk", "critical_outcome", "ED_3day_readmit", "hospitalization_prediction"]
    models = ["catboost-ve", "NN-dropout"]
    
    for dataset, model in list(itertools.product(datasets,models)):
         stats_for_dataset(model, dataset)