import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers, models
import tensorflow as tf


def cnn(df, model_path):
    X = np.array(df.iloc[:, 0:11].values)
    y = np.array(df.iloc[:, 11].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))

    model = models.Sequential()
    model.add(layers.Conv2D(
        32, (1, 1), activation='relu', input_shape=(11, 1, 1)))
    model.add(layers.MaxPooling2D((1, 1)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.MaxPooling2D((1, 1)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.load_weights(model_path)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    return test_acc
