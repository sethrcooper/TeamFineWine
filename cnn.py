import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt


def cnn():
    ORIGINAL_DATASET = "WineQT.xlsx"

    ORIGINAL_DATA_COLUMNS = {"fixed acidity": float,	"volatile acidity": float,	"citric acid": float,	"residual sugar": float,
                             "chlorides": float,	"free sulfur dioxide": float,	"total sulfur dioxide": float,	"density": float,	"pH": float,
                             "sulphates": float,	"alcohol": float,	"quality": float, "Id": float}

    df = pd.read_excel(ORIGINAL_DATASET, dtype=ORIGINAL_DATA_COLUMNS)

    X = np.array(df.iloc[:, 0:11].values)
    y = np.array(df.iloc[:, 11].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
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
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_test, y_test))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    return test_acc
