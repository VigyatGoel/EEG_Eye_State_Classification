import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    data = pd.read_csv("EEG_Eye_State_Classification.csv")
    X = data.drop("eyeDetection", axis=1)
    Y = data["eyeDetection"]

    Z_scores = np.abs(stats.zscore(X))
    outliers = (Z_scores > 3)
    outlier_indices = X[outliers].index

    X_new = X[(Z_scores < 3).all(axis=1)]
    Y = Y[X_new.index]

    return X_new, Y, outlier_indices


def split_data(X, Y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
