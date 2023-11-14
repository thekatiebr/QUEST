import pandas as pd
import numpy as np
from functools import reduce
# from sklearn import *
from sklearn.metrics import confusion_matrix
import pysubgroup as ps #test comment
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import sys, itertools
# from rulelist.rulelistmodel.categoricalmodel.categoricalrulelist import CategoricalRuleList
from rulelist import RuleList, SubgroupListGaussian
# from RuleList import SubgroupListCategorical
from rulelist.measures import subgroup_measures

# # read in data



def get_uncertainty_values(bin, df):
    sub = df[df["uncertainty group"] == bin]
    unc = sub["uncertainty"]
    return {
        "mean": np.mean(unc),
        "min": np.min(unc),
        "max": np.max(unc)
    }

# for i in range(n_bins):
#     res = get_uncertainty_values(i, df_new)
#     print(res)



def read_data(iteration, model, dataset, n_bins, bin_split="qcut"):
    print("IN read_data")
    train_fn = "input/{0}/uncertainty-info_{1}-train_{2}.csv".format(iteration, model, dataset)
    test_fn = "input/{0}/uncertainty-info_{1}_{2}.csv".format(iteration, model, dataset)

    train_df = pd.read_csv(train_fn, index_col="Unnamed: 0")
    test_df = pd.read_csv(test_fn, index_col="Unnamed: 0")
    #drop truth column
    train_ycls = train_df["truth"]
    test_ycls = test_df["truth"]

    # train_df = train_df.drop(["p(positive class)", "truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)
    # test_df = test_df.drop(["p(positive class)", "truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)
    
    train_df = train_df.drop(["truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)
    test_df = test_df.drop(["truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)

#     # pysubgroup code for discrete
#     train_df = train_df.sort_values(by="uncertainty", ascending=True)
#     if bin_split == "cut":
#         bins,cps = pd.cut(train_df['uncertainty'], duplicates="drop", bins=n_bins, retbins=True, labels=list(range(n_bins)))
#         bins_,cps_ = pd.cut(train_df['uncertainty'], duplicates="drop", bins=n_bins, retbins=True)
#     else:
#         bins,cps = pd.qcut(train_df['uncertainty'], duplicates="drop", q=n_bins, retbins=True, labels=list(range(n_bins)))
#         bins_,cps_ = pd.qcut(train_df['uncertainty'], duplicates="drop", q=n_bins, retbins=True)

#     bins.rename("uncertainty group", inplace=True)
#     bins_.value_counts().to_csv("RQ 3/results/bins_{dataset}_{model}_{bins}bins.csv".format(dataset=dataset, model=model, bins=n_bins))
#     train_df["uncertainty group"] = bins
#     train_df = train_df.sample(frac=1)
    

#     test_df["uncertainty group"] = pd.cut(test_df["uncertainty"], right=False, duplicates="drop", bins=cps, labels=False, include_lowest=True)
#     # df.loc[df['A'] > 2, 'B'] = new_val

#     print(cps)
#     print(cps[0])
#     print(test_df["uncertainty"] <= cps[0])
#     test_df.loc[test_df["uncertainty"] >= cps[-1], "uncertainty group"] = len(cps)-2#.fillna(len(cps)-2, inplace=True)
#     test_df.loc[test_df["uncertainty"] <= cps[0], "uncertainty group"] = 0#.fillna(0, inplace=True)


#     # test_df.dropna(inplace=True)
#     # test_df =  pd.concat([test_df, sub_1, sub_2], axis=0)


#     print("!! UQ GROUP VALUES: ", np.unique(test_df["uncertainty group"].values))
#     print(test_df.loc[np.isnan(test_df["uncertainty group"])])
#     print(test_df[["uncertainty group", "uncertainty"]])
#     print(np.unique(test_df["uncertainty group"], return_counts=True))

    # train_df = train_df.drop(["uncertainty"], axis=1)
    # test_df = test_df.drop(["uncertainty"], axis=1)

    return train_df, test_df, train_ycls, test_ycls



def run_sgd(n_bins, max_rules, min_support, alpha_gain, iteration, model, dataset, bin_split="qcut"):

    
    

    train_df, test_df, train_ycls, test_ycls = read_data(iteration, model, dataset, n_bins)
    print("In run_sgd")
    print("test_df.shape = ", test_df.shape)
    model_name = model
    features = list(test_df.columns)
    for x in ["uncertainty", "uncertainty group"]:
        while x in features: features.remove(x)


    # train, test = train_test_split(df_new, train_size=0.8, shuffle=True)

    trainX = train_df[features]
    print(list(trainX))
    trainY_cat = train_df["uncertainty"]
    # trainY_continuous = train["uncertainty"]

    testX = test_df[features]
    print(list(testX))
    testY_cat = test_df["uncertainty"]
    # testY_continuous = test["uncertainty"]

    # print("about to init data object")
    # data_args = {
    #             "input_data": trainX,
    #             "n_cutpoints": 5,
    #             "discretization": "static",
    #             "target_data": trainY_cat,
    #             "target_model": "categorical", #"single-nominal",# AnyStr #Literal["gaussian", "single-nominal"]
    #             "min_support": 1,
    #             # "attributes":features
    # }
    # data = Data(**data_args)
    # print("data obj init")
    #Args: input_data', 'n_cutpoints', 'discretization', 'target_data', 'target_model', and 'min_support'
    args = {
            # "data": data,
            "max_rules": max_rules,
            "beam_width": 100,
            "min_support": min_support,
            "beam_width": 1,
            "max_depth": 5, 
            "alpha_gain": alpha_gain,
    }

    print(args)
    model = SubgroupListGaussian(**args)
    # SubgroupListGaussian(**args)
    model.fit(trainX, trainY_cat)

    print(model)
    print(dir(model))

    pred = model.predict(testX)

    true = testY_cat.values #.astype(int)
    pred = pred.astype(int)
    r2 = r2_score(true, pred)
    no_rules = model.number_rules
    _, prediction_dist = np.unique(pred, return_counts=True)
    _, true_dist = np.unique(true, return_counts=True)
    cnma = confusion_matrix(true, pred)

    sbg_measures = {} #calculate_subgroup_metrics(model, testX, true, pred)# subgroup_measures.nominal_discovery_measures(model._rulelist, trainX.reset_index(), trainY_cat.reset_index())
    
    train_fn = "input/{0}/uncertainty-info_{1}-train_{2}.csv".format(iteration, model_name, dataset)
    test_fn = "input/{0}/uncertainty-info_{1}_{2}.csv".format(iteration, model_name, dataset)
    assign_subgroups(model, testX, test_fn, testY_cat, test_ycls, "test", model_name, dataset, iteration, n_bins, max_rules, min_support, alpha_gain)
    assign_subgroups(model, trainX, train_fn, trainY_cat, train_ycls, "train", model_name, dataset, iteration, n_bins, max_rules, min_support, alpha_gain)
    results = {
            "iteration": iteration,
            "dataset": dataset,
            "model": model_name,
            "n_bins": n_bins,
            "no_rules": no_rules,
            "accuracy": accuracy,  
            "true_dist": true_dist, 
            "prediction_dist": prediction_dist,
            "confusion_matrix": cnma}
    results.update(args)
    results.update(sbg_measures)
    return results, model

# run_sgd(n_bins, max_rules, iteration, model, dataset)

def sub_assign(rl, X):
    rulelist = rl._rulelist
    n_predictions = X.shape[0]
    instances_covered = np.zeros(n_predictions, dtype=bool)
    matrix = []
    subgro_no = 0
    for subgroup in rulelist.subgroups:
        num_predicates = len(subgroup.pattern)
        instances_subgroup = np.asarray([item.activation_function(X).values.astype(int) for item in subgroup.pattern])
        instances_subgroup = np.sum(instances_subgroup, axis=0) == num_predicates
        matrix.append(instances_subgroup)
        subgro_no += 1
    matrix_ = np.stack(matrix).astype(int)
    catchall = np.sum(matrix_, axis=0)
    catchall = np.asarray([1 if c == 0 else 0 for c in catchall])
    matrix.append(catchall)
    matrix = np.stack(matrix).astype(int)
    # print("ELSE: ", catchall.shape)
    subgroup_assignments = np.argmax(matrix, axis=0)
    print(subgroup_assignments.shape)
    # display(subgroup_assignments)
    
    return subgroup_assignments

def assign_subgroups(rsd, X, df_fn_toappend, y_unc, y_cls, train_test, model_name, dataset, iteration, n_bins, max_rules, min_support, alpha_gain):
    subgroup_assignments = sub_assign(rsd, X)
    print("df_fn_toappend = ", df_fn_toappend)
    df = pd.read_csv(df_fn_toappend)
    uncertainty_bin_assignments = rsd.predict(X)
    subgroup_assignments_ = pd.DataFrame({"subgroup_assignment": subgroup_assignments, "uncertainty_bin_assignments": uncertainty_bin_assignments, "true_uncertainty_bin_assignments": y_unc})
    df.reset_index(inplace=True, drop=True)
    print("X.shape = ", X.shape)
    print("df.shape = ", df.shape)
    print("subgroup_assignments.shape = ", subgroup_assignments.shape)
    to_output = pd.concat([df, subgroup_assignments_], axis=1)
    to_output.to_csv("RQ 3/input/subgroup_assignments_{model}_{dataset}_{train_test}_{iteration}_{max_rules}_{min_support}_{alpha_gain}_bins{bins}_continuous.csv".format( model=model_name,
                                                                                                                                                                dataset=dataset, 
                                                                                                                                                                train_test=train_test,
                                                                                                                                                                iteration=iteration,
                                                                                                                                                                max_rules=max_rules,
                                                                                                                                                                min_support=min_support,
                                                                                                                                                                alpha_gain=alpha_gain,
                                                                                                                                                                bins=n_bins))
    return to_output


def _calc_coverage(matrix, n_samples):
    rule_supports = np.sum(matrix, axis=1)/n_samples
    print(rule_supports)
    return np.mean(rule_supports)
    

def _calc_support(matrix, predictions, true, n_samples):
    correct_by_pred = np.logical_and(predictions, true)
    print(correct_by_pred)

def _calculate_subgoup_sizes(matrix): 
    rule_supports = np.sum(matrix, axis=1)
    subgroups = ["subgroup_{0}".format(i) for i in range(rule_supports.shape[0])]
    return dict(zip(subgroups, rule_supports))

def calculate_subgroup_metrics(rsd, X, y, predictions):
    rulelist = rsd._rulelist
    n_predictions = X.shape[0]
    instances_covered = np.zeros(n_predictions, dtype=bool)
    matrix = []
    for subgroup in rulelist.subgroups:
        instances_subgroup = ~instances_covered &\
                             reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        matrix.append(instances_subgroup)
    matrix = np.stack(matrix) #.astype(int)
    # print(matrix) # rows represent subgroups, columns represent data items
                  # a 1 represents that the data point specified in the column is in the subgroup specified by the row
    default = np.logical_not(matrix.any(axis=0)).astype(int)
    matrix = matrix.astype(int)
    matrix = np.vstack([matrix, default])    
    # matrix.append(default)

    print(matrix)
    print(np.sum(matrix, axis=1))
    print(np.sum(matrix, axis=0))
    print(np.sum(np.sum(matrix, axis=0)))
    print(np.unique(np.sum(matrix, axis=0), return_counts=True))
    print(X.shape)

    coverage = _calc_coverage(matrix, n_predictions)
    print("coverage: ", coverage)
    # support = _calc_support(matrix, predictions, y, n_predictions)
    results = {"coverage": coverage}
    sub_sizes = _calculate_subgoup_sizes(matrix)
    results.update(sub_sizes)


    return results


if __name__ == "__main__":
    global model
    global dataset
    model = "NN-dropout" if len(sys.argv) < 3 else sys.argv[2]
    dataset = "diabetes" if len(sys.argv) < 2 else sys.argv[1]
    bs = "qcut"
    target_model = 'gaussian'
    task = "discovery"
    n_bins_list = [2, 3, 4, 5] #list(range(2, 5, 1)) #list(range(2, 10, 1))
    max_rules_list = [np.inf]#[5, 10, 25, np.inf] #list(range(5,25,5))
    combos_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    min_supports_lst = [1, 10, 100, 1000]
    alpha_gains = [0.0, 0.5, 1.0]
    # max_rules_list.append(np.inf)
    min_acc = -1
    best = {"accuracy": -1}
    combos = list(itertools.product(combos_list, n_bins_list, max_rules_list, min_supports_lst, alpha_gains))
    all_results = []

    for iteration, n_bins, max_rules, min_support, alpha_gain in combos:
        try:
            settings = {
                        "iteration": iteration, 
                        "n_bins": n_bins, 
                        "max_rules": max_rules, 
                        "min_support": min_support, 
                        "alpha_gain": alpha_gain
                        }
            print(settings)
            # try:
            res, rsd = run_sgd(n_bins, max_rules, min_support, alpha_gain, iteration, model, dataset, bin_split=bs)
            print(res)
            best = res if res["accuracy"] > min_acc else best
            min_acc = best["accuracy"]
            print(res)
            res["iteration"] = iteration
            all_results.append(res)
        except Exception as e:
            print("!!!! Settings {0} produced an excpetion".format(settings))
            print(e)
            all_results.append(settings)



        # except Exception as e:
        #     print("{0} {1} caused the following error".format(n_bins, max_rules))
        #     print(e)
        #     print("---Continuing---")
        #     all_results.append({"n_bins": n_bins, "max_rules": max_rules,})

    all_res_df = pd.DataFrame(all_results)
    print(all_res_df)
    all_res_df.to_csv("RQ 3/results/subgroup-discovery_{0}_{1}_continuous.csv".format(model, dataset)) # change to formatted name