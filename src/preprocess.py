import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data):
    X = data.drop('Class', axis=1)
    Y = data['Class']
    return X, Y

def split_and_resample(X, Y, test_size=0.2, random_state=42):
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    sm = SMOTE(random_state=random_state)
    xTrain_res, yTrain_res = sm.fit_resample(xTrain, yTrain)
    return xTrain_res, xTest, yTrain_res, yTest
