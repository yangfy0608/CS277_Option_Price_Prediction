# CS277 Option Price Prediction
This code is for the project "Option Price Prediction" in CS277, ShanghaiTech University.<br>
Authors: Chenhao Jiang, Jiarui Kou, Fuyi Yang, Jiawen Yang

## Introduction
```
Option is the right to buy or sell a certain amount of basic commodities at a time allowed in the future obtained by the purchaser after paying a certain option fee. Option price is the only variable in the option contract that changes with market supply and demand. Its level directly affects the profit and loss of both buyers and sellers, and is the core issue of option trading. 
```

## What tasks do our code solve?
```
We plan to build a new option pricing model based on the traditional parametric option pricing model by introducing a combination of non-parametric machine learning models to obtain higher prediction accuracy of option pricing. Based on the call option data of Shanghai 50ETF, this paper comprehensively compares the prediction results of option pricing by parametric models and non-parametric machine learning models, in order to study the advantages and disadvantages of each option pricing model.
```

## Code Structure
```
|- 传统机器学习模型/
    |- LinearRegression.py/      	      # 线性回归模型
    |- RandomForest.py/        		      # 随机森林模型
    |- RandomForestParameter.ipynb/       # 随机森林模型调参
    |- Comparison.ipynb/   				  # 模型比较结果以及绘图
    |- util.py/                 		  # 工具包
    |- Data/              			 	  # 数据集
    	|- 行权价预测图/
    	|- plot1/						  # MAE、RMSE
    	|- plot2/						  # MPE、MAPE
    	|- Prediction.csv/
    	|- Usedata.csv/
|- GA模型/
    |- GA.py/      	                      # 基因遗传算法
    |- util.py/                 		  # 遗传算法工具包
    |- Data/              			 	  # 数据集
    	|- 日行情1/ 
        |- 日行情2/ 
    	|- 看涨期权/ 
    	|- test.csv/ 
    	|- train.csv/ 

```

## Conclusion
```
Ultimately, the best performing parametric and nonparametric models are combined to achieve optimal results. Due to the different types of models, the combination here cannot perform model fusion in the traditional sense, so we use the output of the parameter model as the input of the machine learning model to improve the prediction accuracy. 
```
