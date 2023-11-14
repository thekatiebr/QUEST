import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper
from imblearn.over_sampling import *
<<<<<<< HEAD
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

def read_image_data(dataset):
    dataset_dir = f"data/{dataset}/train"
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(dataset_dir,
                                                                    image_size=(224,224),
                                                                    validation_split=0.3,
                                                                    seed=42,
                                                                    subset = "both",
                                                                    label_mode='categorical',
                                                                    shuffle=True)
    dataset_dir = f"data/{dataset}/test"
    test_ds = tf.keras.utils.image_dataset_from_directory(dataset_dir,
                                                         image_size=(224,224),
                                                          label_mode='categorical',
                                                        shuffle=True)
   
    return train_ds, val_ds, test_ds
    

=======
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30

def read_cv_data(i, dataset):
    train_X = pd.read_csv("data/cross_val_split/{0}_train_X_{1}.csv".format(dataset,i))
    train_y = pd.read_csv("data/cross_val_split/{0}_train_y_{1}.csv".format(dataset,i)).squeeze()
    
    test_X = pd.read_csv("data/cross_val_split/{0}_test_X_{1}.csv".format(dataset,i))
    test_y = pd.read_csv("data/cross_val_split/{0}_test_y_{1}.csv".format(dataset,i)).squeeze()

    return train_X, test_X, train_y, test_y


<<<<<<< HEAD
def cross_val_split(X,y, dataset,):
=======
def cross_val_split(X,y):
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X,y)):
        train_X, test_X, train_y, test_y = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        
        train_X.to_csv("data/cross_val_split/{0}_train_X_{1}.csv".format(dataset,i), index=False)
        train_y.to_csv("data/cross_val_split/{0}_train_y_{1}.csv".format(dataset,i), index=False)
        
        test_X.to_csv("data/cross_val_split/{0}_test_X_{1}.csv".format(dataset,i), index=False)
        test_y.to_csv("data/cross_val_split/{0}_test_y_{1}.csv".format(dataset,i), index=False)

def read_data(dataset: str, rand_sd=None):
    df = pd.read_csv("data/{0}.csv".format(dataset)).sample(frac=1, random_state=rand_sd)
    
    if dataset in ["critical_outcome", "critical_triage", "ED_3day_readmit", "hospitalization_prediction"]:
        df["gender"] = df["gender"].map({"M": 0, "F": 1})
    if dataset == "trauma_uk":
        df['class'] = df['class'].map({"T": 1, "F": 0})
    
<<<<<<< HEAD
    if dataset not in ["diabetic_retinopathy"]:
        df = df.astype(float)
=======
    df = df.astype(float)
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
    X = df.copy()
    
    X.drop("class", axis=1, inplace=True)
    y = df["class"]
    
    
    return X,y

def get_columns(dataset: str):
    categorical_columns = {
        "trauma_uk": [],
        "diabetes": [],
        "critical_outcome": [],
        "critical_triage": [],
        "ED_3day_readmit": [],
        "hospitalization_prediction": [],
<<<<<<< HEAD
        "adult_income": [],
        "mnist": []
=======
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
    }

    continuous_columns = {
        "trauma_uk": ["SYSBP","RR","GCS","SI","AGE","RTS"],
        "diabetes": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
<<<<<<< HEAD
        "mnist": [f"pixel{i}" for i in range(784)],
=======
        "mnist": ["pixel_{0}".format(i) for i in range(784)],
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
        "critical_outcome": ["age", "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity"],
        "critical_triage": ["age", "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_pain", "triage_acuity"],
        "ED_3day_readmit": ["age", "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_pain", "triage_acuity", "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last", "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_los", "n_med", "n_medrecon"],
        "hospitalization_prediction": ["age", "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_pain", "triage_acuity"],
<<<<<<< HEAD
        "adult_income": ["age", "hours_per_week"]
=======
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
    }
    categorical_column_lst = categorical_columns[dataset]
    continuous_column_lst = continuous_columns[dataset]
    return categorical_column_lst, continuous_column_lst

def preprocessing_pipeline(dataset: str, all_cols=[]):
    cat_columns, continuous_columns = get_columns(dataset)
    transformer = preprocessing.FunctionTransformer(lambda x:x, validate=True)
    remaining_cols = list(set(all_cols).difference(set(cat_columns+continuous_columns)))
    mapper = DataFrameMapper(
        [([continuous_col], preprocessing.StandardScaler()) for continuous_col in continuous_columns] +
        [([cat_col], preprocessing.LabelBinarizer()) for cat_col in cat_columns] +
        [([c], transformer) for c in remaining_cols], df_out=True
    )
    
    return mapper

def preprocess_data(dataset: str, X_train, X_test):
    mapper = preprocessing_pipeline(dataset, X_train.columns)
    all_columns = list(X_train)
    all_columns.sort()
    X_train = mapper.fit_transform(X_train)
    X_test = mapper.transform(X_test)
    return X_train[all_columns], X_test[all_columns]

def oversample(dataset, X_train, y_train):
    status = {
        "trauma_uk": False,
        "diabetes": False,
        "critical_outcome": False,
        "critical_triage": False,
        "ED_3day_readmit": False,
        "hospitalization_prediction": False
    }

    if status[dataset]:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

<<<<<<< HEAD
    return X_train, y_train

if __name__ == "__main__":
    import sys
    # d = sys.argv[1]
    # X, y = read_data(d)
    # cross_val_split(X,y,d)
    read_image_data(dataset="diabetic_retinopathy")
=======
    return X_train, y_train
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
