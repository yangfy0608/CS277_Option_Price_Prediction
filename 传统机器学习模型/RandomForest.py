import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from util import Compare, getError, getdata


def KfoldRandomForest(X, Y, kfold):
    error = [0, 0, 0, 0]
    for train_index, test_index in kfold.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        model = RandomForestRegressor(
            n_estimators=190,
            max_depth=12,
            max_features='log2',
            random_state=42,
            min_samples_leaf=1
        )
        model.fit(X_train, Y_train)
        print(model.coef_)
        predictions = model.predict(X_test)
        Y_test = np.array(Y_test)
        error += getError(predictions, Y_test)

    return error/5


if __name__ == "__main__":
    df1 = getdata("./data/UseData.csv")

    X = df1.drop(["Day", "Value"], axis=1)  # 选择特征值和标签值
    Y = df1.Value
    kfolds_regresssion = KFold(n_splits=5, random_state=42, shuffle=True)

    error = KfoldRandomForest(X, Y, kfolds_regresssion)
    print("#############Error of Random Forest################")
    print("MAE  = ", error[0])
    print("RMSE = ", error[1])
    print("MPE  = ", error[2])
    print("MAPE = ", error[3])
