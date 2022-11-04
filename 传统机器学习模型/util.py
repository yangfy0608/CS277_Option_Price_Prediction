import pandas as pd
import numpy as np


def MAE(predictions, test):
    error = 1/len(predictions) * np.sum(abs(predictions-test))
    return error


def RMSE(predictions, test):
    error = np.sqrt(1/len(predictions) * np.sum(np.square(predictions-test)))
    return error


def MPE(predictions, test):
    error = 1/len(predictions) * np.sum(predictions-test) / predictions[-1]
    return error


def MAPE(predictions, test):
    error = 1/len(predictions) * \
        np.sum(abs(predictions-test)) / predictions[-1]
    return error


def Compare(predictions, test):
    print(abs(MAE(predictions, test)*100), '%')
    print(abs(RMSE(predictions, test)*100), '%')
    print(abs(MPE(predictions, test)), '%')
    print(abs(MAPE(predictions, test)), '%')
    return


def getError(predictions, test):
    return np.abs(np.array([MAE(predictions, test)*100, RMSE(predictions, test)*100, MPE(predictions, test), MAPE(predictions, test)]))


def getdata(dataname):
    df = pd.read_csv(dataname)
    df = df.drop(["Day"], axis=1)
    df = df.dropna()  # 删去无效值
    df.drop(df[(df.rate >= 1.15) | (df.rate < 0.85)].index, inplace=True)
    return df
