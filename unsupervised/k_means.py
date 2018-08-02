#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Logistic Regression

from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import pandas as pd

class Kmeans:
    def __init__(self):
        self.dataset = "dataset/preprocessed_data.csv"
        self.y = "dataset/objective_data.csv"

    def k_means(self):
        X = pd.read_csv(self.dataset)
        y = pd.read_csv(self.y)

        df_unprocessed = X

         # 6: Feature Selection and Model Building
        from sklearn.cross_validation import train_test_split
        
        # Use train_test_split in sklearn.cross_validation to split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)

        # Such a large set of features can cause overfitting and also slow computing
        # Use feature selection to select the most important features
        import sklearn.feature_selection
        
        select = sklearn.feature_selection.SelectKBest(k=35)
        selected_features = select.fit(X_train, y_train)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [X.columns[i] for i in indices_selected]
        
        X_train_selected = X_train[colnames_selected]
        X_test_selected = X_test[colnames_selected]
        

        def find_model_perf(X_train, y_train, X_test, y_test):
            model = KMeans(n_clusters=2,random_state=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
        
            return auc
        
        auc_processed = find_model_perf(X_train_selected, y_train, X_test_selected, y_test)
        
     
        # Building a model using unprocessed data to compare
        
        # Drop missing values so model does not throw any error
        df_unprocessed = df_unprocessed.dropna(axis=0, how='any')
        
        
        # Remove non-numeric columns so model does not throw an error
        for col_name in df_unprocessed.columns:
            if df_unprocessed[col_name].dtypes not in ['int32', 'int64', 'float32', 'float64']:
                df_unprocessed = df_unprocessed.drop(col_name, 1)
        
        X_unprocessed = df_unprocessed
        y_unprocessed = y
        
        # Split unprocessed data into train and test set
        # Build model and assess performance
        X_train_unprocessed, X_test_unprocessed, y_train, y_test = train_test_split(
            X_unprocessed, y_unprocessed, train_size=0.70, random_state=1)
        
        auc_unprocessed = find_model_perf(X_train_unprocessed, y_train, X_test_unprocessed, y_test)
        
        # Compare model performance
        print('\nKMeans:')
        print('AUC of model with data preprocessing: {auc}'.format(auc=auc_processed))
        print('AUC of model with data without preprocessing: {auc}'.format(auc=auc_unprocessed))
        per_improve = ((auc_processed-auc_unprocessed)/auc_unprocessed)*100
        print('Model improvement of preprocessing: {per_improve}%'.format(per_improve = per_improve))
                
        return "KMeans: " + str(auc_processed)