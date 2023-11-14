import os, sys
sys.path.append("..")
from utils.metrics import compute_metrics

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import tensorflow as tf
import pandas as pd
import numpy as np
# from autokeras.keras_layers import MultiCategoryEncoding
from KerasNNs import load_keras
from sklearn import preprocessing



dataset = sys.argv[1]
df = pd.read_csv("../data/{0}.csv".format(dataset)).sample(frac=1)
X = df.copy()
X.drop("class", axis=1, inplace=True)
y = df["class"]
if dataset == "trauma_uk":
    y = y.map({"T": 1, "F": 0})

# y_ = tf.keras.utils.to_categorical(y, 2)
y_ = y
# y_train_ = tf.keras.utils.to_categorical(y_train, 2)
# y_test_ = tf.keras.utils.to_categorical(y_test, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




model = load_keras(dataset)
model.summary()
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = []
for train, test in kfold.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

    sc = preprocessing.StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)


    y_train_ = tf.keras.utils.to_categorical(y_train, 2)
    model.fit(X_train, y_train_, epochs=10)
    y_pred = model.predict(X_test)
    
    predicted_preds = compute_metrics(y_pred[:,1], y_test)
    # predicted_preds = compute_metrics(y_pred[:,1], y_test)
    print(predicted_preds)
    results.append(predicted_preds)
    # results.append(compute_metrics(y_pred, y_test))
    model=load_keras(dataset)

results_df_2 = pd.DataFrame(results)
results_df_2.loc["mean"] = results_df_2.mean(axis=0)
print(results_df_2)


model_json = model.to_json()
with open("saved_models/model_{0}.json".format(dataset), "w") as json_file:
    json_file.write(model_json)


# res = model(X_test.to_numpy(), training=True)
# print(res)
