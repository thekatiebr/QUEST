import sys
sys.path.append("RQ 3/")
from categorical_rsd import run_sgd
import numpy as np
import pandas as pd
from tensorflow import keras
from functools import reduce
from catboost import CatBoostClassifier
from imblearn.over_sampling import * 
from data import *
from KerasNNs import *
from metrics import *
from numpy.random import seed 
from GenerateUncertaintyData import *
seed(1) # keras seed fixing import

import tensorflow
tensorflow.random.set_seed(2) # tensorflow seed fixing

def _calculate_metrics(model, X, y, model_type, type_=""):
    prediction_fn = {
        "NN-dropout": model.predict,
        "catboost-ve": model.predict_proba
    }
    
    to_return = {}
    y_prob = prediction_fn[model_type](X)
    y_prob = [p[1] for p in y_prob]
    metrics = compute_metrics(y_prob, y)
    for key in metrics:
        to_return["{1} {0}".format(type_, key)] = metrics[key]
    
    if model_type == "catboost-ve":
        uncertainty, _, _ = calculate_uncertainty_ve(X, model, t=50, n_trees = -1, use_dropout=False, filename="")
    elif model_type == "NN-dropout":
        uncertainty, _, _ = calculate_uncertainty_dropout_nn(X, model, t=50, n_trees = -1, filename="")
        
    to_return["uncertainty - {0}".format(type_)] = np.mean(uncertainty)
    
    return to_return

def _fit(model, X, y, model_type):
    # new_X, _ = preprocess_data(dataset, X, X)
    if model_type == "NN-dropout":
        y_train_ = tf.keras.utils.to_categorical(y, 2)
        cb = [tf.keras.callbacks.EarlyStopping(patience=5)]
        model.fit(X, y_train_, epochs=100, callbacks=cb, validation_split=0.4)
    elif model_type == "catboost-ve":
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model

def _save_model(m, dataset, iteration_no, model_type):
    if model_type == "NN-dropout":
        mdl_name = "models/{dataset}_{iteration_no}_KerasNN".format(dataset=dataset, iteration_no=iteration_no)
        m.save(mdl_name)
    if model_type == "catboost-ve":    # parameters not required.
        m.save_model("models/{dataset}_{iteration_no}_catboost".format(dataset=dataset, iteration_no=iteration_no))

def _instantiate_model(dataset, model):
    if model == "NN-dropout":
        m = load_keras(dataset)
    if model == "catboost-ve":
        m = CatBoostClassifier()      # parameters not required.
    return m

def cross_val(dataset, model):
    results = []
    for i in range(10):
        train_X_, test_X_, train_y_, test_y_ = read_cv_data(i, dataset)
        train_X, test_X = preprocess_data(dataset, train_X_, test_X_)
        m = _instantiate_model(dataset, model)
        m = _fit(m, train_X, train_y_, model)
        res = _calculate_metrics(m, test_X, test_y_, model, type_="")
        res["iteration"] = i
        results.append(res)
        _save_model(m, dataset, i, model)
    return pd.DataFrame(results)

if __name__ == "__main__":
    for model in ["catboost-ve", "NN-dropout"]:
<<<<<<< HEAD
        for dataset in ["adult_income"]: #["diabetes", "trauma_uk", "critical_outcome", "critical_triage", "ED_3day_readmit", "hospitalization_prediction"]:
=======
        for dataset in ["diabetes", "trauma_uk", "critical_outcome", "critical_triage", "ED_3day_readmit", "hospitalization_prediction"]:
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
            print("============ {0} {1} ============".format(dataset, model))
            results = cross_val(dataset, model)
            print(results)
            results.to_csv("results/cv results/{0}_{1}_crossval_new.csv".format(dataset, model))