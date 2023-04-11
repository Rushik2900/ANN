import tensorflow as tf

def get_data(valid_datasize):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    X_valid, X_train = X_train_full[:valid_datasize]/255, X_train_full[valid_datasize:]/255
    y_valid, y_train = y_train_full[:valid_datasize], y_train_full[valid_datasize:]

    X_test = X_test/255

    return(X_train, y_train, X_test, y_test, X_valid, y_valid)