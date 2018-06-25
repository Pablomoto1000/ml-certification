# 1: Modeling Overview

# Task: Binary classification model (Given a set of questions predict whether a candidate will attend interviews or not)

# 2: Introduce the Data

# Import data and take a look
import numpy as np
import pandas as pd

df = pd.read_csv(
    '/Users/pablomoto/Documents/Machine Learning/Preprocessing/Week 2 (Preprocess)/MLChallenge/Interview.csv')

# Format data
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
     print(df)
     
print(df['Observed Attendance'].value_counts())

def format_data(feature):
    for x in range(len(feature)):
        value = feature[x]
        if type(value) is str:
            value = value.lower()
            value = value.replace(' ', '')
            feature[x] = value
    return feature


# Format data of feature 'Observed Attendance'
df['Observed Attendance'] = format_data(df['Observed Attendance'])

# Assign outcome as 0 if Observed attendance is NO and as 1 if YES (Binarize)
df['Observed Attendance'] = [0 if x == 'no' else 1 for x in df['Observed Attendance']]

# Assign X as a DataFrame of features and y as a Series of the outcome variable
X = df.drop('Observed Attendance', 1)
y = df['Observed Attendance']
     
     
# 3: Basic Data Cleaning
# A. Dealing with data types
print("Data without cleaning")
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
     print(X)
    
     
# In this case Expected Attendance category, set as 'yes, no and uncertain' to later create dummies of it
X['Expected Attendance'] = format_data(df['Expected Attendance'])

print("After formatting expected attendance")
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
     print(X)
print(X['Expected Attendance'].value_counts())

for x in range(len(X['Expected Attendance'])):
    value = X['Expected Attendance'][x]
    if type(value) is str:
        if value == 'no':
            X['Expected Attendance'][x] = value
        elif value == 'uncertain':
            X['Expected Attendance'][x] = value
        else:
            X['Expected Attendance'][x] = 'yes'

print("After formatting expected attendance and standarize it")
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
     print(X)
print(X['Expected Attendance'].value_counts())


# Create a list of features to dummy
todummy_list = ['Client name', 'Industry', 'Location', 'Position to be closed', 'Nature of Skillset', 'Interview Type', 'Gender',
                'Candidate Current Location', 'Candidate Job Location', 'Interview Venue', 'Candidate Native location', 'Marital Status', 'Expected Attendance']

# Function to dummy all the categorial variables used for modeling


def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


X = dummy_df(X, todummy_list)

print("After dummying features")
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
     print(X)


# Convert the next features to binary
tobinary_list = ['Have you obtained the necessary permission to start at the required time', 'Hope there will be no unscheduled meetings', 'Can I Call you three hours before the interview and follow up on your attendance for the interview',
                 'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much', 'Have you taken a printout of your updated resume. Have you read the JD and understood the same', 'Are you clear with the venue details and the landmark.', 'Has the call letter been shared']

for x in tobinary_list:
    X[x] = format_data(X[x])
    print("Feature '{col_name}' has \n{unique_cat} ".format(
        col_name=x, unique_cat=X[x].value_counts()))
    X[x] = [1 if x == 'yes' else 0 for x in X[x]]
    print("Feature '{col_name}' has \n{unique_cat} ".format(
        col_name=x, unique_cat=X[x].value_counts()))

print("After binarize features")
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
     print(X)

# Display all the features on the data set and 5 rows of each one
# with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
#     print(X)
     
# B. Handling missing data
# How much of you data is missing?
X.isnull().sum().sort_values(ascending=False).head()

X = X.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Date of Interview', 'Name(Cand ID)'], 1)
print("Removing useless features")
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
     print(X)
X.isnull().sum().sort_values(ascending=False).head()
df_unprocessed = X

# 4: More Data Exploration
# A. Outlier detection
# B. Plotting distributions
# 5: Feature Engineering
# A. Interactions between features
# B. Dimensionality reduction using PCA

# 6: Feature Selection and Model Building
from sklearn.cross_validation import train_test_split

# Use train_test_split in sklearn.cross_validation to split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)

# The total number of features have grown substantially after dummyung and adding interaction terms
print(df.shape)
print(X.shape)

# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=27)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]

print("Features selected")
colnames_selected

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def find_model_perf(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat = [x[1] for x in model.predict_proba(X_test)]
    auc = roc_auc_score(y_test, y_hat)
    
    return auc

auc_processed = find_model_perf(X_train_selected, y_train, X_test_selected, y_test)
print(auc_processed)

# Building a model using unprocessed data to compare

# Drop missing values so model does not throw any error
df_unprocessed = df_unprocessed.dropna(axis=0, how='any')

print(df.shape)
print(df_unprocessed.shape)

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
print(auc_unprocessed)

# Compare model performance
print('AUC of model with data preprocessing: {auc}'.format(auc=auc_processed))
print('AUC of model with data without preprocessing: {auc}'.format(auc=auc_unprocessed))
per_improve = ((auc_processed-auc_unprocessed)/auc_unprocessed)*100
print('Model improvement of preprocessing: {per_improve}%'.format(per_improve = per_improve))