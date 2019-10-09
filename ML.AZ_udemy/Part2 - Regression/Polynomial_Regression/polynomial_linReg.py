#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:05:00 2019

@author: med-pvb
"""

# Polynomial regression

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
X_test= sc_X.transform(X_test)"""

# Fitting Linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X = X, y = y)

# Fitting Polynomial regression modal
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X = X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing Linear Reg results 
plt.scatter(X,y,color="red")
plt.plot(X,linreg.predict(X),color="blue")
plt.title("Linreg predictions")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing Polynomial reg results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X = X)),color="blue")
plt.title("Polynomial predictions")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing Polynomial reg results smooth
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X = X_grid)),color="blue")
plt.title("Polynomial predictions")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Lin Reg
linreg.predict(6.5)

# Predicting a new result with Pol LR
lin_reg2.predict(poly_reg.fit_transform(X = 6.5))