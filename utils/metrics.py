from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from data import *
# import tensorflow as tf
# from KerasNNs import *
def specificity_score(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred).ravel()
    tn = cm[0]
    fp = cm[1] if len(cm) > 1 else -1
    fn = cm[2] if len(cm) > 2 else -1
    tp = cm[3] if len(cm) > 3 else -1
    specificity = tn / (tn+fp) if len(cm) > 1 else -1
    return specificity

def area_under_precision_recall_curve(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def compute_metrics(probs, trues, threshold=0.5):
    """Computes accuracy, f1, precision, recall, AUC

    Args:
        probs (1-d array; Real): probability of class 1
        trues (1-d discrete): classes
    """
    results = {}
    pred_class = [int(p >= threshold) for p in probs]
    # print(probs, np.unique(pred_class, return_counts=True), threshold)
    results["accuracy"] = accuracy_score(y_true = trues, y_pred=pred_class) 
    results["f1"] = f1_score(y_true = trues, y_pred=pred_class)
    results["precision"] = precision_score(y_true = trues, y_pred=pred_class)
    results["recall/sensitivity"] = recall_score(y_true = trues, y_pred=pred_class)
    results["specificity"] = specificity_score(y_true = trues, y_pred=pred_class)
    results["auc prc"] = area_under_precision_recall_curve(y_true = trues, y_pred=probs)
    try:
        results["auc roc"] = roc_auc_score(y_true = trues, y_score=probs)
    except:
        results["auc roc"] = -1
    return results

def numerical_integration_rejection_classification(accuracy, percent_removed, base_accuracy=1.0):
    # TO DO: Re-do with manual sorting
    # print("Accuracy: ", accuracy[0])
    new_accuracy = accuracy - base_accuracy
    percent_removed = percent_removed/100
    # new_accuracy = accuracy - accuracy[0]
    # new_accuracy = accuracy/np.max(accuracy)
    # plot(percent_removed, new_accuracy)
    return sp.integrate.simpson(y=new_accuracy, x=percent_removed), new_accuracy

def relative_area_under_lift_curve(uq):
    # see https://arxiv.org/pdf/2107.00649.pdf
    rnd = uq.sample(frac=1)
    N = 10#uq.shape[0]
    delta = 0 #0.00001
    s = 1.0/N
    uq.sort_values(by="uncertainty", ascending=False, inplace=True)
    start = 0
    end = s
    aulc = -1 # sum is added to -1
    for i in range(N):
        start_index = int(start * uq.shape[0])
        end_index = int(end*uq.shape[0])+1
        sub_uq = uq.iloc[start_index:end_index]
        sub_rnd = rnd.iloc[start_index:end_index]

        uq_cls = [int(p >= 0.5) for p in sub_uq["p(positive class)"] ]
        rnd_cls = [int(p >= 0.5) for p in sub_rnd["p(positive class)"] ]
        Fqi = accuracy_score(sub_uq["truth"].values, uq_cls)
        Frqi = 1 #accuracy_score(sub_rnd["truth"].values, rnd_cls) + delta
        aulc += s * (Fqi / Frqi)
        start = end
        end += s

    return aulc


def plot_rejection_classification(X_uq,y_uq,X_cntl = None, y_cntl=None, method="", dataset=""):
    plt.plot(X_uq, y_uq, label="Removed by Uncertainty")
    if X_cntl is not None and y_cntl is not None:
        plt.plot(X_cntl, y_cntl, label="Randomly Removed")

    plt.legend()
    plt.xlabel("% Removed")
    plt.ylabel("Accuracy")
    plt.savefig("results/rejection-classification_{0}_{1}.png".format(method, dataset))
    plt.show()
    plt.clf()

def plot_calibration(X_pr_unc, y_cls_unc, method="", dataset=""):
    plt.scatter(X_pr_unc, y_cls_unc, s=10)
    # plt.legend()
    plt.xlabel("Uncertainty of Output Probability")
    plt.ylabel("Class Uncertainty")
    plt.savefig("results/cls-pr_calibration_{0}_{1}.png".format(method, dataset))
    # plt.show()
    plt.clf()


def plot_calibration_rate_correct(X_pr_unc, y_cls_pro, truth, method="", dataset=""):
    plt.scatter(X_pr_unc, y_cls_pro, s=10)

    # plt.legend()
    plt.xlabel("Uncertainty of Output Probability")
    plt.ylabel("Rate Correct")
    plt.savefig("results/cls-pr-correct_calibration_{0}_{1}.png".format(method, dataset))
    # plt.show()
    plt.clf()

def plot_rejection_classification(unc_results, rnd_results, method_nm, dataset):
    plt.plot(unc_results["% most uncertain removed"], unc_results["accuracy to plot"], label="Removed by Uncertainty")
    plt.plot(rnd_results["% most uncertain removed"], rnd_results["accuracy to plot"], label="Randomly Removed")
    plt.legend()
    plt.xlabel("% Most Uncertain Removed")
    plt.ylabel("Accuracy")
    plt.savefig("results/rejection-classification_{0}_{1}.png".format(method_nm, dataset))
    plt.clf()


def cross_validation(X, y, mdl_, mdl_string, dataset, threshold=0.5):
    num_classes = 10 if dataset == "mnist" else 2
    # metrics = {"accuracy": make_scorer(accuracy_score),
    #            "f1": make_scorer(f1_score),
    #            "precision": make_scorer(precision_score),
    #            "recall/sensitivity": make_scorer(recall_score),
    #            "roc": make_scorer(roc_auc_score),
    #            "specificity": make_scorer(specificity_score)}
    #
    # if mdl_string != "NeuralNet":
    #     mapper = preprocessing_pipeline(dataset)
    #     pipeline = make_pipeline(mapper, mdl_())
    #     results = cross_validate(pipeline,
    #                              X,
    #                              y,
    #                              scoring=metrics,
    #                              cv=10)
    #     results_df_2 = pd.DataFrame(results)
    #     results_df_2.loc["mean"] = results_df_2.mean(axis=0)
    #     results_df_2.to_csv("results/cv-10fold_{0}_{1}.csv".format(mdl_string, dataset))
    # else:
    mdl_c = mdl_
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = []
    for train, test in kfold.split(X, y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        X_train, y_train = oversample(dataset, X_train, y_train)

        X_train, X_test = preprocess_data(dataset, X_train, X_test)
        if mdl_string == "NeuralNet":
            y_train_ = tf.keras.utils.to_categorical(y_train, 2)
            mdl_.fit(X_train, y_train_, epochs=20)
            y_pred = mdl_.predict(X_test)
            res = compute_metrics(y_pred[:, 1], y_test, threshold=threshold)
            print(res)
            results.append(res)
            mdl_= load_keras(dataset)
        else:
            mdl_ = mdl_c()
            y_train_ = y_train
            mdl_.fit(X = X_train, y = y_train_)
            y_pred = mdl_.predict_proba(X_test)
            res = compute_metrics(y_pred[:, 1], y_test, threshold=threshold)
            print(res)
            results.append(res)
            mdl_ = mdl_c()
            

    results_df_2 = pd.DataFrame(results)
    results_df_2.loc["mean"] = results_df_2.mean(axis=0)
    results_df_2.to_csv("results/cv-10fold_{0}_{1}.csv".format(mdl_string, dataset))
    print(results_df_2)