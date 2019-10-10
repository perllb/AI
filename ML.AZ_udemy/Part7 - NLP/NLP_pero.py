#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:12:08 2019

@author: med-pvb
"""

# NLP

# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
dataset = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t', quoting =  3)

# cleaning text
import re
import nltk 
nltk.download('stopwords') # stopwords, noninformative words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # remove all non-letter 
    review = review.lower()
    review = review.split() 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # take only informative words
    review = ' '.join(review)
    corpus.append(review)
    
# creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked'].values

# classification model

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cmdf = pd.DataFrame(cm,columns = ['0','1'],index = ['0','1'])
print(cmdf)
tn, fp, fn, tp = cm.ravel()

# Metrics
accuracy = (tp + tn) / cmdf.values.sum() # TP + TN / all
precision = tp / (tp + tn)  # TP / (TP + TN)
recall = tp / (tp + fn) # TP / (TP + FN) 
f1 = 2 * precision * recall / (precision + recall)

################################################
# Evalueate several classifiers

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# Function to evaluate model
def evaluator(classifier, clas_type):
    

    print("\n\n######################\n##Classifier:\t", clas_type, "\n######################")
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    cmdf = pd.DataFrame(cm,columns = ['0','1'],index = ['0','1'])
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics
    accuracy = (tp + tn) / cmdf.values.sum() # TP + TN / all
    precision = tp / (tp + fp)  # TP / (TP + FP)
    recall = tp / (tp + fn) # TP / (TP + FN) 
    f1 = 2 * precision * recall / (precision + recall)
    
    print("Acc:\t\t", "%.2f" % accuracy)
    print("Precision:\t", "%.2f" % precision)
    print("Recall:\t\t", "%.2f" % recall)
    print("F1:\t\t", "%.2f" % f1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier.fit(X_train, y_train)
evaluator(classifier, "Bayes")

# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', weights = 'uniform')
classifier.fit(X_train, y_train)
evaluator(classifier, "KNN")

# kernel svm (SVC) 
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 'scale', C = 5, random_state = 0)
classifier.fit(X_train, y_train)
evaluator(classifier, "SVC")

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 'scale', random_state = 0)
classifier.fit(X_train, y_train)
evaluator(classifier, "SVM")

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
evaluator(classifier, "Random Forest")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
evaluator(classifier, "Decision Tree")

# Logistic Reg
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
evaluator(classifier, "Logistic Reg")