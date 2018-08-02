# -*- coding: utf-8 -*-

# AI Trainer 
import sys
import re
from functools import reduce

sys.path.append('preprocessing/')
sys.path.append('supervised/')
sys.path.append('unsupervised/')

import preprocessing, linear_regression, logistic_regression, decision_tree, svm, k_means


best_score=[]
counter=0.0

# Call Preprocess class to format the data
preprocess = preprocessing.Preprocess()
# Uncomment the line below if need to preprocess the dataframe on run
#preprocess.preprocess()

# Supervised learning methods
# Linear Regression
linear_regression = linear_regression.LinearReg()
best_score.append(linear_regression.linear_regression())

# Logistic Regression
logistic_regression = logistic_regression.LogisticReg()
best_score.append(logistic_regression.logistic_regression())

# Decision Tree
decision_tree = decision_tree.DeciTree()
best_score.append(decision_tree.decision_tree())

# Unsupervised learning methods
# Support Vector Machines
svm = svm.Svm()
best_score.append(svm.svmachines())

# K-means
k_means = k_means.Kmeans()
best_score.append(k_means.k_means())

def highest(x):
    global counter
    highest = float(re.findall("\d+\.\d+", x)[0])
    if (highest > counter):
        counter = highest
        return x

max_score = list(map(highest, best_score))
max_score = list(filter(None.__ne__, max_score))[-1]

print("\n\nThe best model is " + str(max_score))