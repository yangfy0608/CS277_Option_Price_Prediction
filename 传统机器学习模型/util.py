import pandas as pd
import numpy as np
import math


def MAE(predictions, test):  # 平均误差
    # error = 1/len(predictions) * np.sum(abs(predictions-test))
    error = np.mean(np.abs(predictions-test))
    return error


def RMSE(predictions, test):  # 均方误差
    # error = np.sqrt(1/len(predictions) * np.sum(np.square(predictions-test))
    error = np.sqrt(np.mean(np.square(predictions-test)))
    return error


def MPE(predictions, test):  # 均百分比误差
    # error = 1/len(predictions) * np.sum((predictions-test) / predictions)
    error = np.mean((predictions-test)/test)
    return error


def MAPE(predictions, test):  # 均绝对值百分比误差
    # error = 1/len(predictions) * \
    #     np.sum(abs((predictions-test)/predictions))
    error = np.mean(np.abs((predictions-test)/test))
    return error


def Compare(predictions, test):  # 返回误差
    print("MAE  = ", (MAE(predictions, test)))
    print("RMSE = ", (RMSE(predictions, test)))
    print("MPE  = ", (MPE(predictions, test)))
    print("MAPE = ", (MAPE(predictions, test)))
    return


def getError(predictions, test):  # 得到误差
    return np.abs(np.array([MAE(predictions, test), RMSE(predictions, test), MPE(predictions, test), MAPE(predictions, test)]))


def getdata(dataname):  # 获取数据
    df = pd.read_csv(dataname)
    df = df.dropna()  # 删去无效值
    df.drop(df[(df.rate >= 1.15) | (df.rate < 0.85) | (df.Value<=0)].index, inplace=True)
    return df
