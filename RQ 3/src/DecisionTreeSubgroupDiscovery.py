import sys
sys.path.append("RQ 3/")
import DecisionTreeSubgroupDiscovery as dtsd
from kneed import KneeLocator

from collections import Counter
import numpy as np
import pandas as pd
import itertools
from tensorflow import keras
from functools import reduce
from catboost import CatBoostClassifier
from imblearn.over_sampling import * 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils.data import *
from utils.KerasNNs import *
from utils.metrics import *
from numpy.random import seed 
from utils.GenerateUncertaintyData import *
import sklearn
from sklearn.tree import _tree
import matplotlib.pyplot as plt
seed(1) # keras seed fixing import

import tensorflow
tensorflow.random.set_seed(2) # tensorflow seed fixing

def print_rules(rules):
    for rule in rules:
        print(rule)
        
def write_rules_to_file(rules, iteration, cls_model, dataset, n_bins, res_key=None):
    if res_key is not None:
        fn = f"RQ 3/results/{res_key}/DTRules_{cls_model}_{dataset}_{iteration}_bins{n_bins}.txt"
    else:
        fn = f"RQ 3/results/DTRules_{cls_model}_{dataset}_{iteration}_bins{n_bins}.txt"
    with open(fn, 'w') as f:
        for rule in rules:
            f.write(rule)
            f.write('\n')

def read_data(iteration, model, dataset, n_bins, bin_split="qcut", include_probability=False, res_key = None):
    # print("IN read_data")
    train_fn = "input/{0}/uncertainty-info_{1}-train_{2}.csv".format(iteration, model, dataset)
    test_fn = "input/{0}/uncertainty-info_{1}_{2}.csv".format(iteration, model, dataset)

    train_df = pd.read_csv(train_fn, index_col="Unnamed: 0")
    test_df = pd.read_csv(test_fn, index_col="Unnamed: 0")
    #drop truth column
    train_ycls = train_df["truth"]
    test_ycls = test_df["truth"]

    if not include_probability:
        train_df = train_df.drop(["p(positive class)", "truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)
        test_df = test_df.drop(["p(positive class)", "truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)
    else:
        train_df = train_df.drop(["truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)
        test_df = test_df.drop(["truth", "ratio 1 predicted", "rate corrected predicted", "class uncertainty"], axis=1)

    # pysubgroup code for discrete
    train_df = train_df.sort_values(by="uncertainty", ascending=True)
    if bin_split == "cut":
        bins,cps = pd.cut(train_df['uncertainty'], duplicates="drop", bins=n_bins, retbins=True, labels=list(range(n_bins)))
        bins_,cps_ = pd.cut(train_df['uncertainty'], duplicates="drop", bins=n_bins, retbins=True)
    else:
        bins,cps = pd.qcut(train_df['uncertainty'], duplicates="drop", q=n_bins, retbins=True, labels=list(range(n_bins)))
        bins_,cps_ = pd.qcut(train_df['uncertainty'], duplicates="drop", q=n_bins, retbins=True)

    bins.rename("uncertainty group", inplace=True)
    if res_key is not None:
        fn = f"RQ 3/results/{res_key}/bins_{dataset}_{model}_{bins}bins.csv"
    else:
        fn = f"RQ 3/results/bins_{dataset}_{model}_{bins}bins.csv"
    bins_.value_counts().to_csv()
    train_df["uncertainty group"] = bins
    # train_df = train_df.sample(frac=1)
    

    test_df["uncertainty group"] = pd.cut(test_df["uncertainty"], right=False, duplicates="drop", bins=cps, labels=False, include_lowest=True)
    # df.loc[df['A'] > 2, 'B'] = new_val

    # print(cps)
    # print(cps[0])
    # print(test_df["uncertainty"] <= cps[0])
    test_df.loc[test_df["uncertainty"] >= cps[-1], "uncertainty group"] = len(cps)-2#.fillna(len(cps)-2, inplace=True)
    test_df.loc[test_df["uncertainty"] <= cps[0], "uncertainty group"] = 0#.fillna(0, inplace=True)


    # test_df.dropna(inplace=True)
    # test_df =  pd.concat([test_df, sub_1, sub_2], axis=0)


    # print("!! UQ GROUP VALUES: ", np.unique(test_df["uncertainty group"].values))
    # print(test_df.loc[np.isnan(test_df["uncertainty group"])])
    # print(test_df[["uncertainty group", "uncertainty"]])
    # print(np.unique(test_df["uncertainty group"], return_counts=True))

    train_df = train_df.drop(["uncertainty"], axis=1)
    test_df = test_df.drop(["uncertainty"], axis=1)

    return train_df, test_df, train_ycls, test_ycls


def tune_decision_tree(train_X, train_y, selection_criteria="amia" ):
    """
        selection_criteria = 
            * amia = sorting as described in amia paper
            * most_leaves = select the tree with teh most leaves
            * elbow = select the tree at the elbow
    """
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.15)
    print("TRAIN SHAPE: ", train_X.shape)
    tune_res = []
    for min_samples_leaf in [0.01]: # [1, 100, 250, 500, 1000, 2500, 5000, 10000]:
        dtc = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=int(0.2*train_X.shape[1]))
        dtc.fit(train_X, train_y)
        path = dtc.cost_complexity_pruning_path(train_X, train_y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        scores, subs, fit_scores = [], [], []
        for ccp_alpha in ccp_alphas:
            dtc = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, max_depth=int(0.5*train_X.shape[1]))
            dtc.fit(train_X, train_y)
            val_score = dtc.score(val_X, val_y)
            num_subgroups = dtc.get_n_leaves()
            max_rule_len = dtc.get_depth()
            max_rule_len = max_rule_len if max_rule_len > 0 else 1
            tune_res.append({
                "min_samples_leaf": min_samples_leaf,
                "ccp_alpha": ccp_alpha,
                "score": val_score,
                "num_subgroups": num_subgroups,
                "max rule len": max_rule_len
            })
            scores.append(val_score)
            subs.append(num_subgroups)
            print(f"min_samples_leaf = {min_samples_leaf} | "+
                  f"ccp_alpha = {ccp_alpha} | "+
                  f"val_score = {val_score} | "+
                  f"no. subgroups = {num_subgroups} | "+
                 f"max rule len = {max_rule_len}")
        plot_tuning(subs, scores)
    res_df = pd.DataFrame(tune_res)
    if selection_criteria == "most_leaves":
        res_df.sort_values(by=["num_subgroups"], ascending=[False], inplace=True)
    elif selection_criteria == "amia":    
        res_df.sort_values(by=["score", "num_subgroups"], ascending=[False, True], inplace=True)
    else: # Need to think through how to do this
        # res_df.sort_values(by=["num_subgroups"], ascending=[True], inplace=True)
        try:
            kn = KneeLocator(res_df["num_subgroups"], res_df["score"], curve='concave', direction='increasing')
            n_ = kn.knee
            if n_ is None:
                res_df.sort_values(by=["num_subgroups"], ascending=[True], inplace=True)
            else:
                res_df = res_df[res_df["num_subgroups"] == n_]
        except:
            res_df.sort_values(by=["num_subgroups"], ascending=[True], inplace=True)
    print(res_df.iloc[0])

    # n_ = int(res_df.shape[0] * 0.15) + 1
    # res_df = res_df.head(n=n_)
    # res_df.sort_values(by="num_subgroups", ascending=True, inplace=True)
    # print(res_df.head())
    # return res_df["min_samples_leaf"].values[0], res_df["ccp_alpha"].values[0], res_df["max rule len"].values[0]
    return res_df


def _get_leaf_indices(clf):
    leaf_indices = []
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
            leaf_indices.append(node_id)
    return leaf_indices

def assign_subgroups(tree, X):
    leaf_indices = np.asarray(_get_leaf_indices(tree))
    leaf_indices = np.sort(leaf_indices)
    subgroup_ids = list(range(leaf_indices.shape[0]))
    subgroup_mapper = dict(zip(leaf_indices, subgroup_ids))
    subgroups = tree.apply(X)
    subgroups = [subgroup_mapper[i] for i in subgroups]
    return subgroups

def produce_assignment_file(tree, X, y_unc, iteration, bins, cls_model, dataset, is_train, res_key=None):
    if is_train:
        df_fn_to_append = f"input/{iteration}/uncertainty-info_{cls_model}-train_{dataset}.csv"
    else:
        df_fn_to_append = f"input/{iteration}/uncertainty-info_{cls_model}_{dataset}.csv"
    df_to_append = pd.read_csv(df_fn_to_append)
    subgroup_assignment = assign_subgroups(tree, X)
    bin_predicted = tree.predict(X)
    subgroup_assignments_ = pd.DataFrame({"subgroup_assignment": subgroup_assignment, 
                                          "uncertainty_bin_assignments": bin_predicted, 
                                          "true_uncertainty_bin_assignments": y_unc})
    df_to_append.reset_index(inplace=True, drop=True)
    to_output = pd.concat([df_to_append, subgroup_assignments_], axis=1)
    train_test="train" if is_train else "test"
    if res_key is not None:
        fn = f"RQ 3/input/{res_key}/subgroup_assignments_{cls_model}_{dataset}_{train_test}_{iteration}_DT_bins{bins}.csv"
    else:
        fn = f"RQ 3/input/subgroup_assignments_{cls_model}_{dataset}_{train_test}_{iteration}_DT_bins{bins}.csv"
    to_output.to_csv(fn)
    return to_output
    
def produce_statistics_file(input_df, train_test, n_subgroups, model, dataset, variables=None):
    results = {}
    n_total = input_df.shape[0]
    # results["no subgroups"] = n_subgroups
    for subgroup in range(n_subgroups):
        sub_df = input_df[input_df["subgroup_assignment"] == subgroup]
        size = sub_df.shape[0]
        coverage = size/n_total
        y_pred = (sub_df["p(positive class)"] >= 0.5).astype(int)
        class_accuracy = accuracy_score(y_pred, sub_df["truth"])
        bin_assignment_accuracy = accuracy_score(sub_df["uncertainty_bin_assignments"], sub_df["true_uncertainty_bin_assignments"])
        true_bin_dist = str(Counter(sub_df["true_uncertainty_bin_assignments"]))
        predicted_bin_dist = str(Counter(sub_df["uncertainty_bin_assignments"]))
        no_unique = -1
        no_total = -1
        vars_used = []
        if variables is not None:
            vars_used = variables[subgroup]
            no_total = len(vars_used)
            vars_used = list(set(vars_used))
            no_unique = len(vars_used)
            
        results.update({
            f"subgroup {subgroup} {train_test} size": size,
            f"subgroup {subgroup} {train_test} coverage": coverage,
            f"subgroup {subgroup} {train_test} classification accuracy": class_accuracy,
            f"subgroup {subgroup} {train_test} bin assignment accuracy": bin_assignment_accuracy,
            f"subgroup {subgroup} {train_test} true bin distribution": true_bin_dist,
            f"subgroup {subgroup} {train_test} predicted bin distribution": predicted_bin_dist,
            f"subgroup {subgroup} {train_test} no. unique variables": no_unique,
            f"subgroup {subgroup} {train_test} no. total variables": no_total,
            f"subgroup {subgroup} {train_test} variables used": str(vars_used)
        })
    # results =  pd.DataFrame(results)
    
    return results

def get_variables(tree, feature_names, class_names):
    # code from https://mljar.com/blog/extract-rules-decision-tree/
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    orders = []
    def recurse(node, path, paths, orders):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths, orders)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths, orders)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            orders += [node]
            
    recurse(0, path, paths, orders)
    print("Leaf Order: ", orders)
    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    # ii = list(np.argsort(samples_count))
    # paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    i = 0
    for path in paths:
        rule = []
        
        for p in path[:-1]:
            p = p[1:]
            p = p[:-1]
            p = p.split(" ")
            p = p[0]
            rule.append(p)
        #     if rule != "if ":
        #         rule += " and "
        #     rule += str(p)
        # rule += " then "
        # if class_names is None:
        #     rule += "response: "+str(np.round(path[-1][0][0][0],3))
        # else:
        #     classes = path[-1][0][0]
        #     l = np.argmax(classes)
        #     rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        # rule += f" | based on {path[-1][1]:,} samples | {orders[i]}, {i}"
        rules += [rule]
        i += 1
        
    return rules

def get_rules(tree, feature_names, class_names):
    # code from https://mljar.com/blog/extract-rules-decision-tree/
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    orders = []
    def recurse(node, path, paths, orders):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths, orders)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths, orders)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            orders += [node]
            
    recurse(0, path, paths, orders)
    print("Leaf Order: ", orders)
    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    # ii = list(np.argsort(samples_count))
    # paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    i = 0
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples | {orders[i]}, {i}"
        rules += [rule]
        i += 1
        
    return rules

def plot_tuning(subgroups, accuracy, alphas=None):
    plt.plot(subgroups, accuracy)
    # if alpha is not None:
    #     plt.plot(subgroups, alphas)
    # plt.savefig(f"RQ 3/results/DT_{dataset}_{model}_i{iteration}_nbins{n_bins}.png")
    plt.show()

def decision_tree_sgd(dataset, model, iteration, train_X_unc, train_y_unc, n_bins = 0, res_key=None, selection_criteria="amia"):
    res_df = tune_decision_tree(train_X_unc, train_y_unc, selection_criteria)
    min_samples_leaf, ccpa, max_rule_len  = res_df["min_samples_leaf"].values[0], res_df["ccp_alpha"].values[0], res_df["max rule len"].values[0]
    if res_key is not None:
        fn = f"RQ 3/results/{res_key}/DTTuning_{dataset}_{model}_{iteration}_nbins{n_bins}.csv"
    else:
        fn = f"RQ 3/results/DTTuning_{dataset}_{model}_{iteration}_nbins{n_bins}.csv"
    res_df.to_csv(fn)
    dtc = DecisionTreeClassifier(min_samples_leaf = min_samples_leaf, ccp_alpha=ccpa, max_depth = max_rule_len)
    dtc.fit(train_X_unc, train_y_unc)
    return dtc