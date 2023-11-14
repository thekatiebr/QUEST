import os, sys
sys.path.append("..")


from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from metrics import compute_metrics

def read_data(dataset, seed=None):
# dataset = "trauma_uk"
    df = pd.read_csv("data/{0}.csv".format(dataset)).sample(frac=1, random_state=seed)
    print("Total No. Columns in File: ", len(df.columns))
    y_col="class"
    x_cols = list(df.columns)
    while y_col in x_cols: x_cols.remove(y_col)
    print("No. Features: ", len(x_cols))
    assert(len(x_cols) + 1 == len(df.columns)) #ensure we split the data correctly
    X = df[x_cols]
    y = df[y_col]
    og_counts = y.value_counts()
    if dataset in ["trauma_uk"]:
        y = y.map({"T": 1, "F": 0})
    return X,y

def split_data(X, y, test_size=0.33, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train = pd.DataFrame(y_train).reset_index(drop=True).squeeze()
    y_test = pd.DataFrame(y_test).reset_index(drop=True).squeeze()
    # print(y_train)
    # sys.exit()
    return X_train, X_test, y_train, y_test

def tune_catboost(X,y):

    model = CatBoostClassifier()

    grid = {'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]}

    grid_search_result = model.grid_search(grid,
                                        X=X,
                                        y=y,
                                        plot=False)
    return model, grid_search_result


if __name__ == "__main__":
    dataset = sys.argv[1]
    X,y = read_data(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.33)
    mdl, result = tune_catboost(X_train,y_train)

    probs = mdl.predict_proba(X_test)
    print(compute_metrics(probs[:,1], y_test))