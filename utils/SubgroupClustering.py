import pandas as pd
import numpy as np
import pysubgroup as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import sklearn.cluster as cl
from RSD.rulelist_class import MDLRuleList
from RSD.measures.subgroup_measures import numeric_discovery_measures
import pysubgroup as ps

# read in data
model_type="RF-naive"
dataset = "trauma_uk"
iteration=0
fn = "../../input/{0}/{1}/uncertainty-info_{2}_{0}.csv".format(dataset, iteration, model_type)

df = pd.read_csv(fn, index_col="Unnamed: 0")
#drop truth column
df = df.drop(["truth", "p(positive class)"], axis=1)
display(df)

y_col = "uncertainty"
x_cols = list(df.columns)
while y_col in x_cols: 
    x_cols.remove(y_col)


y = df[y_col]
X = df[x_cols]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

y_train.plot.kde()

# Import library
from clusteval import clusteval
X_ = train.to_numpy()
cls = "kmeans"
eval = "silhouette"
# Set parameters, as an example dbscan
ce = clusteval(cluster=cls, evaluate=eval)

# Fit to find optimal number of clusters using dbscan
results= ce.fit(X_)

# Make plot of the cluster evaluation
ce.plot()

# Make scatter plot. Note that the first two coordinates are used for plotting.
ce.scatter(X_)

# results is a dict with various output statistics. One of them are the labels.
cluster_labels = results['labx']
print(ce)

# subgroup disc
target_model = 'gaussian'
task = "discovery"

# user configuration
delim = ','
disc_type = "static"
max_len = 5
beamsize = 100
ncutpoints = 5
# load data
model = MDLRuleList(task = task, target_model = target_model, max_rules=10, n_cutpoints=ncutpoints)
model.fit(X_train, y_train)



print(model)
X_t = X
y_t = pd.DataFrame(y)
display(y_t)
measures = numeric_discovery_measures(model._rulelist, X_t, y_t)
pd.DataFrame(measures, index=[0])