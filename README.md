# CS277 Option Price Prediction

This code is for the project "Option Price Prediction" in CS277, ShanghaiTech University.<br>
Authors: Chenhao Jiang, Jiarui Kou, Fuyi Yang, Jiawen Yang

## Introduction

The option is the right to buy or sell a certain amount of basic commodities at a time allowed in the future obtained by the purchaser after paying a certain option fee. The option price is the only variable in the option contract that changes with market supply and demand. Its level directly affects the profit and loss of both buyers and sellers and is the core issue of options trading. 

## What task does our code solve?

We plan to build a new option pricing model based on the traditional parametric option pricing model by introducing a combination of non-parametric machine learning models to obtain higher prediction accuracy of option pricing. Based on the call option data of Shanghai 50ETF,  we compare the prediction results of option pricing by parametric models and non-parametric machine learning models, tobest-performing study the advantages and disadvantages of each option pricing model.

## Structure

##### Parametric Model

```
|- 参数模型/
	|- BlackSholes.py/					   # B-S模型 												(method-1)
	|- MonteCarlo.py/					   # Monte-Carlo模型 										(method-2)
	|- Merton.py/						   # Merton模型(包含参数校准) 							    (method-3)
	|- 看涨期权.xlsx/              		    # Input(包含数据处理和初始参数计算，例如历史波动率)
	|- Error.xlsx/						   # 参数模型Error结果汇总及输出结果汇总
```

##### Non-Parametric Model

```
|- 传统机器学习模型/
    |- LinearRegression.py/      	      # 线性回归模型											(method-1)
    |- RandomForest.py/        		      # 随机森林模型											(method-2)
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
    |- GA.py/      	                      # 基因遗传算法											(method-3)
    |- util.py/                 		  # 遗传算法工具包
    |- Data/              			 	  # 数据集
    	|- 日行情1/ 
        |- 日行情2/ 
    	|- 看涨期权/ 
    	|- test.csv/ 
    	|- train.csv/ 
```

##### Combination Model

```
Ultimately, the best performing parametric and nonparametric models are combined to achieve optimal results. Due to the different types of models, the combination here cannot perform model fusion in the traditional sense, so we use the output of the parameter model as the input of the machine learning model to improve the prediction accuracy. 
```


### Parametric Model
+ Method1: BlackSholes
> We implemented the B-S model, and obtained the theoretical
value.
<br> Run the code by:
```
python BlackSholes.py
```

+ Method2: MonteCarlo
> The basic idea is to simulate multiple paths of the
price of the underlying asset in a risk-neutral world as much
as possible, calculate the average option return under each
path, and then discount the option price. 
<br> Run the code by:
```
python MonteCarlo.py
```

+ Method3: Merton
> TWe completed the Merton model which could also be called
jump-diffusion model. This approach is based on the fact that
in market transactions, the price of the underlying asset tends
to jump suddenly due to unexpected events 
<br> Run the code by:
```
python Merton.py
```

### Non-Parametric Model
+ Method1:  LinearRegression
>A linear regression model is used to predict the theoretical value of option prices, and a loss function is used to control the linear regression model.
<br>Run the code by:
```
python LinearRegression.py
```
+ Method2: RandomForest
> A random sample is aggregated from the original set to generate a set of data, each set is predicted, and the final prediction is determined by the mean method. 
<br>Run the code by:
```
python RandomForest.py
```
+ Method3: LGBM with Genetic Algorithm
> Boosting prediction model is a type of serial integrated learning, where multiple individual learners with strong depen-
dencies are generated serially and then combined into a single
module to complete learning. 
<br>Run the code by:
```
python GA.py
```

### Evaluations
We introduced several pricing error criteria commonly used
to evaluate model predictions:
+ Mean Absolute Deviation(MAE)
  
    $MAE =\frac{1}{n}\sum_{i=1}^n\lvert
    C_i-C_i^{real}\rvert$

+ Root Mean Square Error(RMSE):

	$RMSE =\sqrt{\frac{1}{n}\sum_{i=1}^n{( C_i-C_i^{real})}^2}$

+ Mean ABSolute Percentage Error(MAPE):

	$MAPE =\frac{1}{n}\sum_{i=1}^n(\frac{\lvert C_i-C_i^{real}\rvert}{C_i^{real}})$

Where the $C_n$ represents the call option price predicted by the model, and the $C_n^{real}$ represents the real call option price.

### Outcomes
We have implemented three classical parametric and three non-parametric models and the results are presented in the table bellow:
|Evaluation Methods |B-S model |Monte Carlo |Merton model|LR| RFR |Genetic lgbm|
|-------------------|----------|------------|------------|---|---|---|
|MAE|0.015816 |0.015876 |0.022146|0.033376| 0.013984| 0.029087|
|RMSE|0.024384 |0.024446| 0.025446|0.043294 |0.020674 |0.042116|
|MAPE|0.307365 |0.308817 |0.341510|9.876618 |0.581362|0.567493|


