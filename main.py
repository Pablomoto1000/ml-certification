# -*- coding: utf-8 -*-

# AI Trainer 
import sys
sys.path.append('supervised/')
sys.path.append('preprocessing/')


import preprocessing, linear_regression, logistic_regression

# Call Preprocess class to format the data
preprocess = preprocessing.Preprocess()
preprocess.preprocess()

# Supervised learning methods
# Linear Regression
linear_regression = linear_regression.LinearReg()
linear_regression.linear_regression()

# Logistic Regression
logistic_regression = logistic_regression.LogisticReg()
logistic_regression.logistic_regression()

# Decision Tree
