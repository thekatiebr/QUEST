import os, sys, itertools, traceback
sys.path.append("~/dissertation_code")
import random
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from scipy.special import expit as sigmoid, logit as inverse_sigmoid
import pandas as pd
import scipy as sp
import json, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(2) # tensorflow seed fixing
<<<<<<< HEAD
from KerasNNs import * #load_keras, remove_softmax
from data import *
from metrics import *
=======
from utils.KerasNNs import * #load_keras, remove_softmax
from utils.data import *
from utils.metrics import *
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
# from TuneCatboost import tune_catboost
from numpy.random import seed 
seed(1) # keras seed fixing import




def split_data(X, y, test_size=0.33, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train = pd.DataFrame(y_train).reset_index(drop=True).squeeze()
    y_test = pd.DataFrame(y_test).reset_index(drop=True).squeeze()
    # print(y_train)
    # sys.exit()
    return X_train, X_test, y_train, y_test
    


# def read_cv_data(i):
#     train_X = pd.read_csv("data/cross_val_split/{0}_train_X_{1}.csv".format(dataset,i))
#     train_y = pd.read_csv("data/cross_val_split/{0}_train_y_{1}.csv".format(dataset,i)).squeeze()
    
#     test_X = pd.read_csv("data/cross_val_split/{0}_test_X_{1}.csv".format(dataset,i))
#     test_y = pd.read_csv("data/cross_val_split/{0}_test_y_{1}.csv".format(dataset,i)).squeeze()

#     return train_X, test_X, train_y, test_y

def specificity_score(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred).ravel()
    tn = cm[0]
    fp = cm[1] if len(cm) > 1 else -1
    fn = cm[2] if len(cm) > 2 else -1
    tp = cm[3] if len(cm) > 3 else -1
    specificity = tn / (tn+fp) if len(cm) > 1 else -1
    return specificity

# def compute_metrics(probs, trues, threshold=0.5):
#     """Computes accuracy, f1, precision, recall, AUC
#         Hello :)
#     Args:
#         probs (1-d array; Real): probability of class 1
#         trues (1-d discrete): classes
#     """
#     results = {}
#     avg = "binary" #f dataset != "mnist" else "weighted"
#     pred_class = [int(p >= threshold) for p in probs]
#     results["accuracy"] = accuracy_score(y_true = trues, y_pred=pred_class) 
#     results["f1"] = f1_score(y_true = trues, y_pred=pred_class, average=avg)
#     # results["precision"] = precision_score(y_true = trues, y_pred=pred_class, average=avg)
#     # results["recall/sensitivity"] = recall_score(y_true = trues, y_pred=pred_class, average=avg)
#     # results["specificity"] = specificity_score(y_true = trues, y_pred=pred_class) if dataset != "mnist" else -1
#     try:
#         results["auc roc"] = roc_auc_score(y_true = trues, y_score=probs, average = avg)
#     except:
#         results["auc roc"] = -1
#     return results



def get_metrics(eval_df):
    print(eval_df)
    results = compute_metrics(probs=eval_df["p(positive class)"], 
                             trues=eval_df["truth"], threshold=0.5)
    return results


def generate_rate_corrected_predicted(y_cls_pro, truth):
    y = []
    for i in range(truth.shape[0]):
        truth = truth.values if isinstance(truth, pd.DataFrame) or isinstance(truth, pd.Series) else truth
        val = 1 - y_cls_pro[i] if truth[i] == 0 else y_cls_pro[i]
        y.append(val)
    return y

def generate_rejection_classification(mdl, X, y, method, n_trees, t, X_untransformed):
    """Creates rejection classification data. Returns as dataframe

    Args:
        mdl (XGBoost): pre-trained model to use
        X (Pandas DataFrame): input features
        y (Pandas DataFrame/Series): true labels
        n_trees (int): number of trees in ensemble
    """
    method_fn = method[0]
    print(method[1])

    unc,cls_unc,cls_pro = method_fn(X, mdl, t=t, n_trees=n_trees, filename="raw_uncertainty_{0}_{1}".format(method[1], dataset))
    preds_prob = mdl.predict_proba(X) if method[1][0:2] != "NN" else mdl(X.to_numpy(), training=True)
    preds_prob = preds_prob[:, 1]
    rate_correct = generate_rate_corrected_predicted(y_cls_pro=cls_pro, truth=y)

    # print(unc, preds_prob)
    # if method[1] == "catboost-ve":
    #     df = pd.read_csv("input/uncertainty-info_{0}_{1}.csv".format(method[1], dataset), index_col="Unnamed: 0")
    #     df["truth"] = y
    #     df.to_csv("input/uncertainty-info_{0}_{1}.csv".format(method[1], dataset))
    # else:
    print("unc.shape: ", unc.shape)
    print("cls_pro.shape: ", cls_pro.shape)
    print("len(rate_correct): ", len(rate_correct))
    print("cls_unc.shape: ", cls_unc.shape)
    print("preds_prob.shape: ", preds_prob.shape)
    print("y.shape: ", y.shape)
    print("type(y): ", type(y))
    df = pd.DataFrame({"uncertainty": unc, "ratio 1 predicted": cls_pro, "rate corrected predicted": rate_correct, "class uncertainty": cls_unc, "p(positive class)": preds_prob, "truth": y}, index=X.index)
    # pass the mapper and use inverse transform on X before creating df
    # if mapper is not None:
    #     X = mapper.inverse_transform(X)
    df = pd.concat([df, X_untransformed], axis=1)
    df.to_csv("input/uncertainty-info_{0}_{1}.csv".format(method[1], dataset))
    
    
    df.sort_values(by="uncertainty", ascending=True, inplace=True)
    
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     display(df)

    unc_results_lst = []
    rnd_results_lst = []

    #remove the most uncertain, calculate 
    for i in range(0, 100, 5):
        proportion_to_keep = 1 - (i/100.)
        index_to_keep = int(proportion_to_keep*df.shape[0])
        
        unc_results = get_metrics(df.iloc[0:index_to_keep])
        unc_results["% most uncertain removed"] = i
        unc_results_lst.append(unc_results)

        rnd_results = get_metrics(df.sample(n=index_to_keep))
        rnd_results["% most uncertain removed"] = i
        rnd_results_lst.append(rnd_results)
        
    unc_results = pd.DataFrame(unc_results_lst)
    rnd_results = pd.DataFrame(rnd_results_lst)

    return unc_results, rnd_results

def generate_data(mdl, X, y, method, n_trees=100, t=50, X_untransformed=None):
    args = {"mdl": mdl, "X":X, "y":y, "method":method, "n_trees": n_trees, "t":t, "X_untransformed":X_untransformed}
    unc_results, rnd_results = generate_rejection_classification(**args)
    unc_results.to_csv("results/rejection-classification_{0}_{1}.csv".format(method[1], dataset))
    rnd_results.to_csv("results/rejection-classification_{0}_rnd-cntl_{1}.csv".format(method[1], dataset))

def plot_results(mdl, X, y, method, n_trees=100, t=50, X_untransformed=None):
    unc_results, rnd_results = generate_rejection_classification(mdl, X, y, method, n_trees, t, X_untransformed)
    unc_results.to_csv("results/rejection-classification_{0}_{1}.csv".format(method[1], dataset))
    rnd_results.to_csv("results/rejection-classification_{0}_rnd-cntl_{1}.csv".format(method[1], dataset))
    plt.plot(unc_results["% most uncertain removed"], unc_results["accuracy"], label="Removed by Uncertainty")
    plt.plot(rnd_results["% most uncertain removed"], rnd_results["accuracy"], label="Randomly Removed")
    plt.legend()
    plt.xlabel("% Most Uncertain Removed")
    plt.ylabel("Accuracy")
    plt.savefig("results/rejection-classification_{0}_{1}.png".format(method[1], dataset))
    plt.clf()

    print("Area Under Curve = ", sp.integrate.simps(y=unc_results["accuracy"], x=unc_results["% most uncertain removed"]))
    print("Area Under Curve (control)= ", sp.integrate.simps(y=rnd_results["accuracy"], x=rnd_results["% most uncertain removed"]))

def calculate_uncertainty_ve(x, mdl, t=500, n_trees = -1, use_dropout=False, filename=""):
    max_trees = mdl.tree_count_
    # t = t if t <= max_trees else 15
    print("max_trees: {0} | t = {1}".format(max_trees, t))
    try:
        p_class = mdl.virtual_ensembles_predict(x, virtual_ensembles_count=t, prediction_type='VirtEnsembles')
        p = mdl.virtual_ensembles_predict(x, virtual_ensembles_count=t, prediction_type='TotalUncertainty')
    except: 
        p_class = mdl.virtual_ensembles_predict(x, virtual_ensembles_count=15, prediction_type='VirtEnsembles')
        p = mdl.virtual_ensembles_predict(x, virtual_ensembles_count=15, prediction_type='TotalUncertainty')
        t = 15
    p_class = np.reshape(p_class, (t, x.shape[0]))
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    p_class = sigmoid(p_class)
    np.savetxt("input/{0}.csv".format(filename), p_class, delimiter=",")
    mean_preds = np.mean(p_class, axis=0)
    cls_var, cls_percent_correct = calculate_class_uncertainty(p_class)
    ku = p[:,1]
    uq_info = {}
    uq_info["p(positive class)"] = mean_preds
    uq_info["uncertainty"] = ku
    uq_info["class uncertainty"] = cls_var
    uq_info["ratio 1 predicted"] = cls_percent_correct


    df = pd.DataFrame(uq_info, index=x.index)
    # print(df.shape)
    df = pd.concat([df, x], axis=1)
    # df.to_csv("input/uncertainty-info_{0}_{1}.csv".format("catboost-ve", dataset))

    return ku, cls_var, cls_percent_correct

def calculate_uncertainty_naive(x, mdl, t=500, n_trees = 100, p=0.3, filename=""):
    import warnings
    """Calculates epistemic uq from tree

    Args:
        x (Pandas DataFrame): Input Features; must have > 1 data point (TODO: Fix to not require this)
        mdl (sklearn RandomForestClassifier): Sklearn random forest classifier
        //t (int, optional): number of ensembles. Defaults to 5. Must be greater than 1; ignored in this method
        n_trees (int, optional): Number of trees in ensemble; need to change
        p (float, optional): ignored. 

    Returns:
        numpy array: uncertainties lining up to the datapoints
    """
    # t is number of samples to gen uncertainty; must be <= num trees in ensemble

    individual_preds = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for tree_id in range(n_trees):
            pred = mdl.estimators_[tree_id].predict_proba(x)[:, 1]
            print(type(mdl.estimators_[tree_id]))
            print(pred)
            individual_preds.append(
                pred
            )

    individual_preds_complete = np.vstack(individual_preds)
    np.savetxt("input/{0}.csv".format(filename), individual_preds_complete, delimiter=",")
    print(individual_preds_complete.shape)
    cls_uncertainty, cls_percent_correct = calculate_class_uncertainty(individual_preds_complete)

    uncertainty = np.std(individual_preds_complete, axis=0)
    print(uncertainty.shape)
    print("Uncertainty (naive): ", uncertainty)
    print("Class Uncertainty (naive): ", cls_uncertainty)
    return uncertainty, cls_uncertainty, cls_percent_correct


def calculate_uncertainty_dropout_rf(x, mdl, t = 50, n_trees = 100, p=0.3, filename=""):
    import warnings
    """Calculates epistemic uq from tree

    Args:
        x (Pandas DataFrame): Input Features; must have > 1 data point (TODO: Fix to not require this)
        mdl (sklearn RandomForestClassifier): Sklearn random forest classifier
        t (int, optional): number of ensembles. Defaults to 5. Must be greater than 1
        n_trees (int, optional): Number of trees in ensemble; need to change
        p (float, optional): probability that ensemble prediction is removed

    Returns:
        numpy array: uncertainties lining up to the datapoints
    """
    # t is number of samples to gen uncertainty; must be <= num trees in ensemble

    individual_preds = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for tree_id in range(n_trees):
            individual_preds.append(
                mdl.estimators_[tree_id].predict_proba(x)[:, 1]
            )

    individual_preds_complete = np.vstack(individual_preds)
    print(individual_preds_complete.shape)


    unc_preds = []
    for i in range(t):
        ens_pred = []
        for pred in individual_preds_complete.transpose():
            # print(pred.shape)
            mask = np.random.random_sample(len(pred))
            pred = pred[mask >= p]
            # print(pred.shape)
            prediction_ = np.mean(pred)
            ens_pred.append(prediction_)
        unc_preds.append(ens_pred)


    unc_preds = np.vstack(unc_preds)
    np.savetxt("input/{0}.csv".format(filename), unc_preds, delimiter=",")
    uncertainty = np.std(unc_preds, axis=0)
    class_uncertainty, cls_percent_correct = calculate_class_uncertainty(unc_preds)
    print(uncertainty.shape)
    # print("Uncertainty (dropout-sim): ", uncertainty)
    # print("Class Uncertainty (dropout-sim): ", class_uncertainty)
    return uncertainty, class_uncertainty, cls_percent_correct


def calculate_uncertainty_dropout_nn(x, mdl, t = 50, n_trees = 100, p=0.3, filename=""):
    # some parameters are for compatiblity
    print(x)
    # note this will not work for multi-class classification
    mdl_ = mdl #remove_softmax(mdl, dataset)
    y_samples = np.stack([mdl_(x.to_numpy(), training=True)[:,1] for sample in range(t)])
    np.savetxt("input/{0}.csv".format(filename), y_samples, delimiter=",")
    y_samples_mean = y_samples.mean(axis=0)
    y_samples_std = y_samples.std(axis=0)

    y_samples_cls_unc, cls_percent_correct = calculate_class_uncertainty(y_samples)
    return y_samples_std, y_samples_cls_unc, cls_percent_correct

def model_metrics(X, y, mdl):
    pred = mdl.predict(X)
    print(pred.shape)
    if len(pred.shape) > 1 and pred.shape[1] == 2:
        pred = pred[:, 1]
    res = compute_metrics(pred, y)
    print(res)

def CatBoostUQ_Analysis(X_train, X_test, y_train, y_test, t=50):
    # mdl,_ = tune_catboost(X_train, y_train) # CatBoostClassifier()
    print("CATBOOST")
    mdl = CatBoostClassifier()
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    mapper = preprocessing_pipeline(dataset, X_train.columns)  # Will need this
    X_train, X_test = preprocess_data(dataset, X_train, X_test)
    X_train_before_val_split = X_train.copy()
    y_train_before_val_split = y_train.copy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
    mdl = CatBoostClassifier()      # parameters not required.
    mdl.load_model("models/{dataset}_{iteration_no}_catboost".format(dataset=dataset, iteration_no=i))
    # X_train, y_train = oversample(dataset, X_train, y_train)
    # mdl.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=False)
    # print("===== Cat Boost Cross val iteration = {0}, dataset = {1} =====".format(i, dataset))
    # mdl.save_model("models/{dataset}_{iteration_no}_catboost".format(dataset=dataset, iteration_no=iteration_no))
    model_metrics(X_test, y_test, mdl)
    # print("y_test.shape = ", y_test.shape)
    # print("y_train.shape = ", y_train.shape)
    plot_results(mdl=mdl, X = X_test, y = y_test, t=t, method = (calculate_uncertainty_ve, "catboost-ve"), X_untransformed=X_test_)
    plot_results(mdl=mdl, X = X_train_before_val_split, y = y_train_before_val_split, t=t, method = (calculate_uncertainty_ve, "catboost-ve-train"), X_untransformed=X_train_)


def RandomForestUQ_Analysis(X_train, X_test, y_train, y_test):
    print("RANDOM FOREST")
    mdl = RandomForestClassifier()
    mapper = preprocessing_pipeline(dataset, X_train.columns)  # Will need this
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    # X_train, y_train = oversample(dataset, X_train, y_train)
    X_train, X_test = preprocess_data(dataset, X_train, X_test)
    mdl.fit(X_train, y_train)
    print("===== Random Forest Cross val iteration = {0}, dataset = {1} =====".format(i, dataset))
    model_metrics(X_test, y_test, mdl)
    plot_results(mdl, X=X_train, y=y_train, n_trees=100, method=(calculate_uncertainty_dropout_rf, "RF-dropout-train"), X_untransformed=X_train_)
    plot_results(mdl, X=X_train, y=y_train, n_trees=100, method=(calculate_uncertainty_naive, "RF-naive-train"), X_untransformed=X_train_)
    plot_results(mdl, X=X_test, y=y_test, n_trees=100, method=(calculate_uncertainty_dropout_rf, "RF-dropout"), X_untransformed=X_test_)
    plot_results(mdl, X=X_test, y=y_test, n_trees=100, method=(calculate_uncertainty_naive, "RF-naive"), X_untransformed=X_test_)


def NeuralNetUQ_Analysis(X_train, X_test, y_train, y_test, t=50):
    print("KERAS")
    num_classes = 10 if dataset == "mnist" else 2
    X_train_untransformed = X_train.copy()
    X_test_untransformed = X_test.copy()
    X_train_, X_test_ = preprocess_data(dataset, X_train, X_test)
    mdl_name = "models/{dataset}_{iteration_no}_KerasNN".format(dataset=dataset, iteration_no=i)
    mdl = tf.keras.models.load_model(mdl_name)
    # mdl = fit_classification_NN(X_train, X_test, y_train, dataset, num_classes, iteration_no, preprocess_data)
    # print("saving weights")
    # print("===== Neural Net Cross val iteration = {0}, dataset = {1} =====".format(i, dataset))
    # X_test_.to_csv("x_test2.csv")
    model_metrics(X_test_, y_test, mdl)
    
    
    
    plot_results(mdl, X=X_train_, y=y_train, t=t, method=(calculate_uncertainty_dropout_nn, "NN-dropout-train"), X_untransformed=X_train_untransformed)
    plot_results(mdl, X=X_test_, y=y_test, t=t, method=(calculate_uncertainty_dropout_nn, "NN-dropout"), X_untransformed=X_test_untransformed)

def calculate_class_uncertainty(p_class):
    p_class = p_class >= 0.5
    p_class_unc = np.std(p_class, axis=0)
    print(p_class_unc.shape)
    p_class_rate = np.sum(p_class, axis=0)/p_class.shape[0] #rate that 1 is predicted
    return p_class_unc, p_class_rate

def main(cross_val=False, t=50):    
    rnd = random.randint(0,1000)
    X,y = read_data(dataset, rnd)
    if cross_val:
        cross_validation(X, y, mdl_=RandomForestClassifier, mdl_string="RandomForest")
        cross_validation(X, y, mdl_=CatBoostClassifier, mdl_string="CatBoost")
        cross_validation(X, y, load_keras(dataset), mdl_string="NeuralNet")

    X_train, X_test, y_train, y_test = read_cv_data(i, dataset)
    # X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.33, random_state=rnd)
    NeuralNetUQ_Analysis(X_train, X_test, y_train, y_test, t=t)
    CatBoostUQ_Analysis(X_train, X_test, y_train, y_test, t=t)
    
    
    # RandomForestUQ_Analysis(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    # datasets = [sys.argv[1]]
<<<<<<< HEAD
    datasets = ["mnist"] #["trauma_uk", "diabetes", "critical_outcome", "critical_triage", "ED_3day_readmit", "hospitalization_prediction"]
=======
    datasets = ["trauma_uk", "diabetes", "critical_outcome", "critical_triage", "ED_3day_readmit", "hospitalization_prediction"]
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
    # datasets = ["trauma_uk"]
    # trials = [0] 
    trials = [0,1,2,3,4,5,6,7,8,9] 
    combos = itertools.product(trials, datasets)
    
    for combo in combos:
        global dataset
        global iteration_no 
        i, dataset = combo
        iteration_no = i
        
        print("==================== {0} - {1} ====================".format(i, dataset))
        try:
            main(False, 50)
        except:
            traceback.print_exc()
        
        try:
            os.system("mkdir results/{0} input/{0}".format(i))
            os.system("mkdir results/{0}/{1} input/{0}/{1}".format(dataset, i))
        except: pass
        os.system("mv results/*.csv results/*.png results/{0}".format(i))
        os.system("mv input/*.csv input/{0}".format(i))
