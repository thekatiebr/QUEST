from metrics import *
from KerasNNs import load_keras
import sys
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from data import *

dataset = sys.argv[1]
model = sys.argv[2]
X,y = read_data(dataset)
model_types = {
    "RandomForest": RandomForestClassifier,
    "CatBoost": CatBoostClassifier,
    "NeuralNet": load_keras(dataset)
}

thresholds = {
    "trauma_uk": 0.5,
    "diabetes": 0.5,
    "critical_outcome": 0.05,
    "critical_triage": 0.05,
    "ED_3day_readmit": 0.04,
    "hospitalization_prediction": 0.455,
}


cross_validation(X, y, mdl_=model_types[model], mdl_string=model, dataset=dataset, threshold=thresholds[dataset])
# cross_validation(X, y, mdl_=model_types[model], mdl_string=model, dataset=dataset)
# cross_validation(X, y, model_types[model], mdl_string=model, dataset=dataset)

