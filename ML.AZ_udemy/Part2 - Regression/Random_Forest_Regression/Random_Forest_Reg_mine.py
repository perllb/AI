#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:15:01 2019

@author: med-pvb
"""

# Random Forest Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Skip splitting due to small dataset
"""# Split data to Train and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
sc_y = Standard_scaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X = X, y = y)

# Predicting a new result 
y_pred = regressor.predict(6)
y[np.where(X==6)]

# Visualizing Regression results
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X = X),color="blue")
plt.title("Predictions Random Forest Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualization for higher resolution
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X = X_grid),color="blue")
plt.title("Predictions Random Forest Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
