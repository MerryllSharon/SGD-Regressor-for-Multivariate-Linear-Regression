# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start
2.Data preparation
3.Hypothesis Definition
4.Cost Function 
5.Parameter Update Rule 
6.Iterative Training 
7.Model evaluation 
8.End 

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Merryll S Sharon
RegisterNumber:  212222080033
*/

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


data = fetch_california_housing()


x= data.data[:,:3]


y=np.column_stack((data.target,data.data[:,6]))

x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state =42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.fit_transform(y_test)

sgd = SGDRegressor(max_iter=1000, tol = 1e-3)

multi_output_sgd= MultiOutputRegressor(sgd)

multi_output_sgd.fit(x_train, y_train)

y_pred =multi_output_sgd.predict(x_test)

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
print(y_pred)
[[ 1.04860312 35.69231257]
 [ 1.49909521 35.72530255]
 [ 2.35760015 35.50646978]
 ...
 [ 4.47157887 35.06594388]
 [ 1.70991815 35.75406191]
 [ 1.79884624 35.34680017]]

mse = mean_squared_error(y_test,y_pred)

print("Mean Squared Error:",mse)

Mean Squared Error: 2.560165984862198

print("\nPredictions:\n",y_pred[:5])
Predictions:
 [[ 1.04860312 35.69231257]
 [ 1.49909521 35.72530255]
 [ 2.35760015 35.50646978]
 [ 2.73967825 35.37568192]
 [ 2.10914107 35.63894336]]
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](https://private-user-images.githubusercontent.com/124332243/366126406-1e6c626f-8096-49d4-93b8-97ca749326d3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjYzMTAzMTMsIm5iZiI6MTcyNjMxMDAxMywicGF0aCI6Ii8xMjQzMzIyNDMvMzY2MTI2NDA2LTFlNmM2MjZmLTgwOTYtNDlkNC05M2I4LTk3Y2E3NDkzMjZkMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwOTE0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDkxNFQxMDMzMzNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00NmM4N2JhMzNiMGFlNDZmNjQ3MjQyZWEwMzFlYWFhODU3OTc3ZGJkY2VkZWJjZTk1NGEzMzQwNjI3YTc1OThhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.VZNCOfD7pl05d0sA3EnQpTQ3QEeoYg5NFPI1uNnDKII)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
