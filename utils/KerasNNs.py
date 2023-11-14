from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
import pandas as pd
import os
# from utils.data import preprocess_data

p=0.3

def mnist_model(n_inputs=784, output_activation="softmax"):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, input_shape=(n_inputs,), activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
<<<<<<< HEAD
            tf.keras.layers.Dense(2, activation=output_activation)
=======
            tf.keras.layers.Dense(10, activation=output_activation)
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy")
    return model

def trauma_model(n_inputs=32, output_activation="softmax"):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, input_shape=(n_inputs,), activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(2, activation=output_activation)
        ]
    )
    # opt = tf.keras.optimizers.RMSprop()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy")
    return model

def diabetes_model(n_inputs=8, output_activation="softmax"):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(n_inputs,)),
            # tf.keras.layers.experimental.preprocessing.Normalization(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(2, activation=output_activation)
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss="binary_crossentropy")
    return model

def load_mimic(n_inputs=64, output_activation="softmax"):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(n_inputs,)),
            # tf.keras.layers.experimental.preprocessing.Normalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(2, activation=output_activation)
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss="binary_crossentropy")
    return model

<<<<<<< HEAD
def load_adult_income(n_inputs=8, output_activation="softmax"):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(n_inputs,)),
            # tf.keras.layers.experimental.preprocessing.Normalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(rate=p),
            tf.keras.layers.Dense(2, activation=output_activation)
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss="binary_crossentropy")
    return model

def load_DR_CNN(input_size=(224,224,3), output_activation="softmax"):
    mdl = tf.keras.applications.VGG16(
                                        include_top=True,
                                        weights=None,
                                        input_shape=input_size,
                                        pooling=None,
                                        classes=2,
                                        classifier_activation=output_activation,
                                        )
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.keras.optimizers.RMSprop()
    mdl.compile(optimizer=opt, loss="binary_crossentropy")
    return mdl

=======
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
def remove_softmax(mdl, id):
    mdl_copy = load_keras(id, output_activation=None)
    weights = mdl.get_weights()
    mdl_copy.set_weights(weights)
    return mdl_copy

def load_keras(id="", output_activation="softmax"):
    nns = {
            "trauma_uk": trauma_model,
            "diabetes": diabetes_model,
            "mnist": mnist_model,
            "critical_outcome": load_mimic,
            "critical_triage": load_mimic,
            "ED_3day_readmit": load_mimic,
            "hospitalization_prediction": load_mimic,
<<<<<<< HEAD
            "adult_income": load_adult_income,
            "diabetic_retinopathy": load_DR_CNN
=======
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
            }
    n_inputs = {
        "trauma_uk": 32,
        "diabetes": 8,
        "mnist": 784,
        "critical_outcome": 64,
        "critical_triage": 67,
        "ED_3day_readmit": 67,
        "hospitalization_prediction": 64,
<<<<<<< HEAD
        "adult_income": 8,
        "diabetic_retinopathy": (224,224,3)
=======
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
    }
    m = nns[id](n_inputs[id], output_activation)
    return m

def fit_classification_NN(X_train, X_test, y_train, dataset, num_classes, iteration_no, preprocess_data):
    mdl = load_keras(dataset)
    X_train_untransformed = X_train.copy()
    X_test_untransformed = X_test.copy()
    
    
    X_train_, X_test_ = preprocess_data(dataset, X_train_untransformed, X_test_untransformed)
    
    print(X_test_)
    print(type(X_test_))
    
    y_train_ = tf.keras.utils.to_categorical(y_train, num_classes)
    cb = [tf.keras.callbacks.EarlyStopping(patience=5)]
    # X_train_.to_csv("x_train2.csv")
    mdl.fit(X_train_, y_train_, epochs=100, callbacks=cb, validation_split=0.4)
    # mdl.save("models/{dataset}_{iteration_no}_KerasNN".format(dataset=dataset, iteration_no=iteration_no))
    mdl_name = "models/{dataset}_{iteration_no}_KerasNN".format(dataset=dataset, iteration_no=iteration_no)
    print(mdl_name)
   
    mdl.save(mdl_name)
<<<<<<< HEAD
    return mdl

def fit_image_CNN(train_ds, val_ds, dataset):
    
    mdl = load_keras(dataset)
    
    cb = [tf.keras.callbacks.EarlyStopping(patience=5)]
    # X_train_.to_csv("x_train2.csv")
    mdl.fit(train_ds, epochs=1, validation_data = val_ds, callbacks=cb)
    # mdl.save("models/{dataset}_{iteration_no}_KerasNN".format(dataset=dataset, iteration_no=iteration_no))
    mdl_name = "models/{dataset}_{iteration_no}_KerasNN".format(dataset=dataset, iteration_no=iteration_no)
    print(mdl_name)
   
    mdl.save(mdl_name)
    return mdl


if __name__ == "__main__":
    import data
    train, test = data.read_image_data(dataset="diabetic_retinopathy")
    cnn = fit_image_CNN(train_ds, "diabetic_retinopathy")
    print(cnn.evaluate(test))
=======
    return mdl
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
