import tensorflow as tf
import time
import os

def create_model(input_shape, no_classes, loss_function, optimizer, metrics):
    LAYERS = [tf.keras.layers.Flatten(input_shape=input_shape, name="inputLayer"),
              tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer_1"),
              tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer_2"),
              tf.keras.layers.Dense(no_classes, activation="sigmoid", name="outputLayer")
             ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    LOSS_Function = loss_function
    OPTIMIZER = optimizer
    Metrics = metrics
    model_clf.compile(loss=LOSS_Function, optimizer=OPTIMIZER, metrics=Metrics) 
    return model_clf

def get_unique_filename(model_name):
    unique_filename=time.strftime(f"%Y-%m-%m_%H-%M_{model_name}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename=get_unique_filename(model_name)
    path_to_model=os.path.join(model_dir, unique_filename)
    model.save(path_to_model)