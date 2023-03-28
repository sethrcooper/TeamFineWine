# Team Fine Wine
# Wine Quality Predictor

# library imports
import os
import math
import statistics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# constants
ORIGINAL_DATASET = "WineQT.xlsx"

ORIGINAL_DATA_COLUMNS = { "fixed acidity": float,	"volatile acidity": float,	"citric acid": float,	"residual sugar": float,
                    "chlorides": float,	"free sulfur dioxide": float,	"total sulfur dioxide": float,	"density": float,	"pH": float,
                    "sulphates": float,	"alcohol": float,	"quality": float, "Id": float}

# Function to load dataframe from excel
def getData(path = ORIGINAL_DATASET):

    if not os.path.exists(path):
        print("[ERROR] COIN COLOR DATA FILE NOT FOUND")
        return None

    return pd.read_excel(path, dtype=ORIGINAL_DATA_COLUMNS)


# main program execution
def main():
    print("\n[INFO] BEGINNING MAIN FUNCTION\n")    # program status update

    # Load data from excel
    df = getData()
    #split into features and results
    X = np.array(df.iloc[:, 0:11].values)
    y = np.array(df.iloc[:, 11].values)
    # split the dataset into a training and testing portion for validation purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    ## TODO ## Implement ML Algorithms in their own functions as seen with getData() above and call them below


    print("\n[INFO] ENDING MAIN FUNCTION\n")    # program status update


# program entry point
if __name__ == "__main__":
    main()