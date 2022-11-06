import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from util import Compare, getError, getdata


def simpleLinearRegression(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    Y_test = np.array(Y_test)
    Compare(predictions, Y_test)
    return


def KfoldLinearRegression(X, Y, kfold):
    error = [0, 0, 0, 0]
    for train_index, test_index in kfold.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        model = LinearRegression()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        Y_test = np.array(Y_test)
        error += getError(predictions*10000, Y_test*10000)
    return error/5


df1 = getdata("UseData.csv")


X = df1.drop(["Day","Value"], axis=1)  # 选择特征值和标签值
Y = df1.Value

kfolds_regresssion = KFold(n_splits=5, random_state=1, shuffle=True)

simpleLinearRegression(X, Y)
error = KfoldLinearRegression(X, Y, kfold=kfolds_regresssion)
print("###############Error of LinearRegression###########")
print("MAE  = ", error[0])
print("RMSE = ", error[1])
print("MPE  = ", error[2])
print("MAPE = ", error[3])
