import pandas as pd
import numpy as np
from sklearn import *
import pysubgroup as ps
import matplotlib.pyplot as plt

def create_bins(n_bins, df):
    df = df.sort_values(by="uncertainty", ascending=True)
    # lbs = ["uncertainty group {0}".format(i) for i in range(n_bins)]
    lbs = [i for i in range(n_bins)]
    bins = pd.qcut(df['uncertainty'], q=n_bins, labels=lbs)
    bins.rename("uncertainty group", inplace=True)
    # bins = pd.get_dummies(bins)
    # bins = preprocessing.OrdinalEncoder().fit_transform(np.reshape(bins, (-1,1)))
    # bins = pd.DataFrame(bins, columns=["uncertainty group {0}".format(i) for i in range(n_bins)])
    df_new = df # df.drop("uncertainty", axis=1)
    df_new = pd.concat([df_new, bins], axis=1)
    return df_new


# target = "uncertainty group 0"
def subgroup_disc_discrete(n_bins, df):
    ignore_lst =  ["uncertainty", "uncertainty group"]
    # ignore_lst.extend(["uncertainty group {0}".format(i) for i in range(n_bins)])
    results = []
    # for i in range(n_bins):
    target = "uncertainty group"
    target = ps.BinaryTarget(target, True)
    searchspace = ps.create_selectors(df, ignore=ignore_lst)
    task = ps.SubgroupDiscoveryTask (
        df,
        target,
        search_space= searchspace,
        # result_set_size=1,
        # depth=2,
        qf=ps.WRAccQF())
    result = ps.BeamSearch().execute(task)
    results.extend(result.to_descriptions())
    return results




def create_subgroup_rule(cn, df):
    cn = cn[1]
    cv = cn.covers(df)
    return df.iloc[cv]

if __name__ == "__main__":
    # read in data
    iteration=0
    train = False
    model = "catboost-ve"
    model_str = model if not train else "{0}-train".format(model)
    dataset = "trauma_uk" 
    n_bins = 2
    #uncertainty-info_catboost-ve_trauma_uk
    #uncertainty-info_catboost-ve-train_trauma_uk
    # fn = "../../input/{0}/uncertainty-info_{1}_{2}.csv".format(iteration, model_str, dataset)
    fn = "input/VarianceMapperResults_raw_trauma_uk_catboost-ve.csv"
    df = pd.read_csv(fn, index_col="Unnamed: 0")
    #drop truth column
    df = df.drop(["truth", "correct", "p(positive class)", "ratio 1 predicted", "rate corrected predicted", "class uncertainty", "pred_likelihood"], axis=1)
    print(df.shape)
    df_new = create_bins(n_bins, df)
    print(df_new)
    d = subgroup_disc_discrete(n_bins=n_bins, df=df_new)
    # # print(d)
    print(len(d))
    # # subgroup_disc_discrete(target, n_bins, df)
    for i in range(n_bins):
        print(d[i])
        s = create_subgroup_rule(d[i], df_new)
        # print(s)
        print(s.shape)
        unc = s["uncertainty"].values
        print(np.min(unc), np.max(unc))