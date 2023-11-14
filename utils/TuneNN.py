import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import sys
from data import *
dataset = sys.argv[1]
X, y = read_data(dataset)

# y = tf.keras.utils.to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test = preprocess_data(dataset, X_train, X_test)

# print(X_train, X_test, y_train, y_test)

# input_node = ak.StructuredDataInput()
# output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
# output_node = ak.ClassificationHead(num_classes=2, loss="binary_crossentropy")(output_node)
# clf = ak.AutoModel(
#     inputs=input_node, outputs=output_node, overwrite=True, max_trials=3
# )

clf = ak.StructuredDataClassifier(
    overwrite=True, max_trials=10, project_name=""
)  # It tries 3 different models.

original_stdout = sys.stdout # Save a reference to the original standard output

clf.fit(X_train, y_train, epochs=10)


# Predict with the best model.
# predicted_y = clf.predict(test_file_path)
# Evaluate the best model with testing data.
print(clf.evaluate(X_test, y_test))

model = clf.export_model()
model.save("saved_models/model_autokeras_{0}".format(dataset), save_format="tf")

model.summary()
model_json = model.to_json()
with open("saved_models/model_{0}.json".format(dataset), "w") as json_file:
    json_file.write(model_json)

print(model.get_config())