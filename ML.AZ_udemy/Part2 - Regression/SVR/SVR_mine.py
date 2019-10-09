#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:03:49 2019

@author: med-pvb
"""

## SVR 

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# exp
sc_dataset = StandardScaler()
xy = dataset.iloc[:,1:3].values
scdata = sc_dataset.fit_transform(xy)

# Fitting SVRegression model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

# Predicting a new result 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


# Visualizing Regression results
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X = X),color="blue")
plt.title("Predictions SVR Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualization for higher resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X = X_grid),color="blue")
plt.title("Predictions SVR Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()