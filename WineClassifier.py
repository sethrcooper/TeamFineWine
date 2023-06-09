# Team Fine Wine
# Wine Quality Predictor

# library imports
import os
import math
import statistics
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cnn.cnn import cnn
from naive_bayes import *
from MLR import *



# constants
ORIGINAL_DATASET = "WineQT.xlsx"

ORIGINAL_DATA_COLUMNS = {"fixed acidity": float,	"volatile acidity": float,	"citric acid": float,	"residual sugar": float,
                         "chlorides": float,	"free sulfur dioxide": float,	"total sulfur dioxide": float,	"density": float,	"pH": float,
                         "sulphates": float,	"alcohol": float,	"quality": int, "Id": int}

# Function to load dataframe from excel


def getData(path=ORIGINAL_DATASET):

    if not os.path.exists(path):
        print("[ERROR] COIN COLOR DATA FILE NOT FOUND")
        return None

    return pd.read_excel(path, dtype=ORIGINAL_DATA_COLUMNS)



# main program execution
def main():
    print("\n[INFO] BEGINNING MAIN FUNCTION\n")    # program status update

    # Load data from excel
    df = getData()
    # split into features and results
    X = np.array(df.iloc[:, 0:11].values)
    y = np.array(df.iloc[:, 11].values)
    # split the dataset into a training and testing portion for validation purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # TODO ## Implement ML Algorithms in their own functions as seen with getData() above and call them below
    model_path = os.getcwd() + '\\cnn\\cnn.h5'
    print("Final Testing Accuracy: ", cnn(X_test, y_test, model_path))

    with open('save_nb_model.txt', 'rb') as f:
        nb_model = pickle.load(f)
    nb_test_predictions = naive_bayes_predict(X_test.T, nb_model)
    nb_test_accuracy = np.mean(nb_test_predictions == (y_test - 3))
    print("Naive Bayes testing accuracy: %f" % nb_test_accuracy)


    # This function trains new weights based on this random train/test split
    # multi_test_predictions, J, theta = multi_logistic_regression(X_train, y_train, X_test)
    # this function uses constant weights as requested for the final project
    multi_test_predictions = predictPreset(X_test)

    lr_test_accuracy = np.mean(np.round(multi_test_predictions) == y_test)
    print("MLR testing accuracy: %f" % lr_test_accuracy)
    # print(theta)
    # print(confusionMatrix(y_test, multi_test_predictions))
    # plotCost(J)




    print("\n[INFO] ENDING MAIN FUNCTION\n")    # program status update


# program entry point
if __name__ == "__main__":
    main()
