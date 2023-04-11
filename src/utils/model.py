import tensorflow as tf

def create_model(input_shape, no_classes, loss_function, optimizer, metrics):
    LAYERS = [tf.keras.Flatten(input_shape=input_shape, name="inputLayer"),
              tf.keras.Dense(300, activation="relu", name="hiddenLayer_1"),
              tf.keras.Dense(100, activation="relu", name="hiddenLayer_2"),
              tf.keras.Dense(no_classes, activation="sigmoid", name="outputLayer")
             ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    LOSS_Function = loss_function
    OPTIMIZER = optimizer
    Metrics = metrics
    model_clf.compile(loss=LOSS_Function, optimizer=OPTIMIZER, metrics=Metrics) 
    return model_clf