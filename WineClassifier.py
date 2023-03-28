# Team Fine Wine
# Wine Quality Predictor

# library imports
import os
import math
import statistics
import numpy as np
import pandas as pd


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

    return pd.read_excel(path, index_col=0, dtype=ORIGINAL_DATA_COLUMNS)


# main program execution
def main():
    print("\n[INFO] BEGINNING MAIN FUNCTION\n")    # program status update

    df = getData()
    X = np.array(df.iloc[:, 0:11].values)
    Y = np.array(df.iloc[:, 11].values)

    

    #print(outputData)  

    print("\n[INFO] ENDING MAIN FUNCTION\n")    # program status update


# program entry point
if __name__ == "__main__":
    main()