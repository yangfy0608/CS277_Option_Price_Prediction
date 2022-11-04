import pandas as pd
import numpy as np


def MAE(predictions, test):  # 平均误差
    error = 1/len(predictions) * np.sum(abs(predictions-test))
    return error


def RMSE(predictions, test):  # 均方误差
    error = np.sqrt(1/len(predictions) * np.sum(np.square(predictions-test)))
    return error


def MPE(predictions, test):  # 均百分比误差
    error = 1/len(predictions) * np.sum(predictions-test) / predictions[-1]
    return error


def MAPE(predictions, test):  # 均绝对值百分比误差
    error = 1/len(predictions) * \
        np.sum(abs(predictions-test)) / predictions[-1]
    return error


def Compare(predictions, test):  # 返回误差
    print("MAE  = ", abs(MAE(predictions, test)))
    print("RMSE = ", abs(RMSE(predictions, test)))
    print("MPE  = ", abs(MPE(predictions, test)))
    print("MAPE = ", abs(MAPE(predictions, test)))
    return


def getError(predictions, test):  # 得到误差
    return np.abs(np.array([MAE(predictions, test), RMSE(predictions, test), MPE(predictions, test), MAPE(predictions, test)]))


def getdata(dataname):  # 获取数据
    df = pd.read_csv(dataname)
    df = df.drop(["Day"], axis=1)
    df = df.dropna()  # 删去无效值
    df.drop(df[(df.rate >= 1.15) | (df.rate < 0.85)].index, inplace=True)
    return df
