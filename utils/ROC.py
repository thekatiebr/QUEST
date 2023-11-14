from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    # v 3.3.2
import seaborn as sns              # v 0.11.0

# from collections import set
def _roc_point_at_threshold(df, threshold):
    """
    Helper function to pull FPR/TPR given a threshold

    Parameters
    ----------
    roc_curve: pandas df with TPR, FPR, Threshold columns
    threshold: threshold to access

    Returns
    -------
    FPR, TPR
    """
    i=0
    npts = df.shape[0]
    while i <= npts and df.iloc[i]["Threshold"] > threshold:
        i += 1
    row = df.iloc[i]

    return row["FPR"], row["TPR"]

def _calculate_statistics(values, z=0.95):
    """

    Parameters
    ----------
    values - np array (1-d) or list of values

    Returns
    -------
    average, std. dev, 95% CI of values

    """
    average = np.mean(values)
    std_dev = np.std(values)
    n = len(values)
    # print(z, std_dev, np.sqrt(n))
    ci = (z * (std_dev/np.sqrt(n)))
    ci_low = average - ci
    ci_high = average + ci
    return average, std_dev, ci_low, ci_high, ci

def threshold_averaging(y_trues, y_scores, class_=1, z=0.95):
    """
    Parameters
    ----------
    y_true: list of numpy 1-d array or python lists of true labels
    y_score: list of numpy arrays or python lists of predicted scores
    Assumes binary classification and uses positive class probability
    Returns
    -------
    Python dataframe of averaging results with columns
    threshold, FPR average, std. dev, 95%CI, TPR average, std. dev, 95%CI
    """
    assert(len(y_trues) == len(y_scores)) #make sure that we have the same number of scores & true arrays
    roc_curves = []
    # create dataframe of ROC information for each provided classifier scores

    thresholds_to_search = []
    for i in range(len(y_trues)):
        y_true = y_trues[i]
        y_score = y_scores[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:,class_], drop_intermediate=True)

        # display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        # display.plot()

        roc_curves.append(
            pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds}).sort_values(by="Threshold", ascending=False)
        )
        thresholds_to_search.append(thresholds)
    # thresholds = set.intersection(*map(set, thresholds_to_search))
    thresholds = [i/100 for i in range(0,100,5)]
    average_roc = []
    for threshold in thresholds:
        tprs = []
        fprs = []
        for df in roc_curves:
            fpr, tpr = _roc_point_at_threshold(df, threshold)
            tprs.append(tpr)
            fprs.append(fpr)
        average_fpr, std_dev_fpr, ci_low_fpr, ci_high_fpr, ci_fpr = _calculate_statistics(fprs, z=0.95)
        average_tpr, std_dev_tpr, ci_low_tpr, ci_high_tpr, ci_tpr = _calculate_statistics(tprs, z=0.95)
        average_roc.append(
            {
                "Threshold": threshold,
                "FPR": average_fpr,
                "Standard Dev. FPR": std_dev_fpr,
                "{0}% Lower CI FPR".format(z*100): ci_low_fpr,
                "{0}% Upper CI FPR".format(z * 100): ci_high_fpr,
                "ci_{0}_fpr".format(int(z*100)): ci_fpr,
                "ci_{0}_tpr".format(int(z*100)): ci_tpr,
                "TPR": average_tpr,
                "Standard Dev. TPR": std_dev_tpr,
                "{0}% Lower CI TPR".format(z * 100): ci_low_tpr,
                "{0}% Upper CI TPR".format(z * 100): ci_high_tpr
            }
        )
    return pd.DataFrame(average_roc)

def _interpolate_tpr(p1, p2, x):
    slope = (p2["TPR"] - p1["TPR"])/(p2["FPR"] - p1["FPR"])
    return p1["TPR"] + slope * (x-p1["FPR"])

def _tpr_for_fpr(fpr_sample, ROC, eps=0.0001):
    i = 0
    n_pts = ROC.shape[0]
    while i <= n_pts and ROC.iloc[i+1]["FPR"] <= fpr_sample:
        i += 1

    if np.abs(ROC.iloc[i]["FPR"] - fpr_sample) <= eps:
        return ROC.iloc[i]["FPR"], ROC.iloc[i]["TPR"]
    else:
        tpr = _interpolate_tpr(ROC.iloc[i], ROC.iloc[i+1], fpr_sample)
        return fpr_sample, tpr

def vertical_averaging(y_trues, y_scores, class_=1, z=0.95):
    assert (len(y_trues) == len(y_scores))  # make sure that we have the same number of scores & true arrays
    roc_curves = []
    # create dataframe of ROC information for each provided classifier scores

    for i in range(len(y_trues)):
        y_true = y_trues[i]
        y_score = y_scores[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:,class_], drop_intermediate=False)
        roc_ = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds}).sort_values(by="FPR", ascending=True)
        roc_.to_csv("roc_{0}.csv".format(i))
        roc_curves.append(
            roc_
        )
    to_return = []

    for fpr in range(0,100,5):
        fpr /= 100
        tprs = []
        for df in roc_curves:
            fpr, tpr = _tpr_for_fpr(fpr, df)
            tprs.append(tpr)

        avg, stdev, ci_low, ci_high, ci = _calculate_statistics(tprs, z=0.95)
        to_return.append({
            "FPR": fpr,
            "TPR": avg,
            "Standard Dev. TPR": stdev,
            "ci_{0}_tpr".format(int(z*100)): ci,
            "{0}% Lower CI TPR".format(z * 100): ci_low,
            "{0}% Upper CI TPR".format(z * 100): ci_high
        })
    return pd.DataFrame(to_return)

def _interpolate_fpr(p1, p2, x):
    slope = (p2["FPR"] - p1["FPR"])/(p2["TPR"] - p1["TPR"])
    return p1["FPR"] + slope * (x-p1["TPR"])

def _fpr_for_tpr(tpr_sample, ROC, eps=0.0001):
    i = 0
    n_pts = ROC.shape[0]
    while i <= n_pts and ROC.iloc[i+1]["TPR"] <= tpr_sample:
        i += 1
        ROC.iloc[i + 1]["TPR"]

    if np.abs(ROC.iloc[i]["TPR"] - tpr_sample) <= eps:
        return ROC.iloc[i]["FPR"], ROC.iloc[i]["TPR"]
    else:
        fpr = _interpolate_fpr(ROC.iloc[i], ROC.iloc[i+1], tpr_sample)
        return fpr, tpr_sample

def horizontal_averaging(y_trues, y_scores, class_=1, z=0.95):
    assert (len(y_trues) == len(y_scores))  # make sure that we have the same number of scores & true arrays
    roc_curves = []
    # create dataframe of ROC information for each provided classifier scores

    for i in range(len(y_trues)):
        y_true = y_trues[i]
        y_score = y_scores[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:,class_], drop_intermediate=False)
        roc_curves.append(
            pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds}).sort_values(by="TPR", ascending=True)
        )
    to_return = []

    for tpr in range(0,100,5):
        tpr /= 100
        fprs = []
        for df in roc_curves:
            fpr, tpr = _fpr_for_tpr(tpr, df)
            fprs.append(fpr)
        print(tpr, fprs)
        avg, stdev, ci_low, ci_high, ci = _calculate_statistics(fprs, z=0.95)
        to_return.append({
            "TPR": tpr,
            "FPR": avg,
            "Standard Dev. FPR": stdev,
            "ci_{0}_fpr".format(int(z*100)): ci,
            "{0}% Lower CI FPR".format(z * 100): ci_low,
            "{0}% Upper CI FPR".format(z * 100): ci_high
        })
    return pd.DataFrame(to_return)


def _old_plot(fpr, tpr, xerr, yerr):
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot()
    
    plt.errorbar(x=fpr, y=tpr, xerr=xerr, yerr=yerr)
    print(type(display))
    plt.show()



def plot_roc(fpr, tpr, xerr=None, yerr=None):
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(fpr, tpr, label='ROC')
    
    plt.errorbar(x = fpr, y = tpr,
            xerr=xerr, yerr=yerr, fmt='none', c= 'black', capsize = 2)
    # print(list(zip(fpr - xerr, fpr + xerr)))
    # lower = 
    # upper = 
    # print(lower.shape)
    # print(upper.shape)
    # ax.plot(fpr, lower, color='tab:blue', alpha=0.1)
    # ax.plot(fpr, upper, color='tab:blue', alpha=0.1)
    # ax.fill_between(fpr, lower, upper, alpha=0.2)
    # lower_x = xerr[:,0]
    # upper_x = xerr[:,1]
    
    plt.show()



def plot(df, name=""):
    df.sort_values(by="FPR", inplace=True)
    fpr = df["FPR"].values
    tpr = df["TPR"].values
    df.to_csv("{0}.csv".format(name))

    if name == "horizontal_averaging":       
        xerr = df["ci_95_fpr"].values
        # xerr = df[["95.0% Lower CI FPR", "95.0% Upper CI FPR"]].values
        # xerr = np.reshape(xerr, (xerr.shape[1], xerr.shape[0]))
        yerr=None
        
    elif name == "vertical_averaging":
        xerr = None
        yerr = df["ci_95_tpr"].values
        # yerr = df[["95.0% Lower CI TPR", "95.0% Upper CI TPR"]].values
        # yerr = np.reshape(yerr, (yerr.shape[1], yerr.shape[0]))
        # plt.title("Vertical Average")
        # plt.plot(df["FPR"].values, df["Average TPR"].values)
    elif name == "threshold_averaging":
        xerr = df["ci_95_fpr"].values
        yerr = df["ci_95_tpr"].values
        # xerr = df[["95.0% Lower CI FPR", "95.0% Upper CI FPR"]].values
        # xerr = np.reshape(xerr, (xerr.shape[1], xerr.shape[0]))
        # yerr = df[["95.0% Lower CI TPR", "95.0% Upper CI TPR"]].values
        # yerr = np.reshape(yerr, (yerr.shape[1], yerr.shape[0]))
        # plt.title("Threshold Average")
        # plt.plot(df["Average FPR"].values, df["Average TPR"].values)
    else:
        xerr = None
        yerr = None

    plot_roc(fpr, tpr, xerr, yerr)

def _test_fn(fn, name, y_trues, y_scores):
    df = fn(y_trues, y_scores)
    
    plot(df, name)
    

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from data import *
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    cls = RandomForestClassifier()
    X,y = read_data("trauma_uk")
    y_scores = []
    y_trues = []
    for i in range(30):
        X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.33)
        cls.fit(X_train, y_train)
        y_score = cls.predict_proba(X_test)
        y_trues.append(y_test)
        y_scores.append(y_score)

    _test_fn(threshold_averaging, "threshold_averaging", y_trues, y_scores)
    _test_fn(vertical_averaging, "vertical_averaging", y_trues, y_scores)
    _test_fn(horizontal_averaging, "horizontal_averaging", y_trues, y_scores)