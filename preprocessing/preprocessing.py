# 1: Modeling Overview

# Task: Binary classification model (Given a set of questions predict whether a candidate will attend interviews or not)

# 2: Introduce the Data

# Import data and take a look
import pandas as pd


# Define a class
class Preprocess:
    def __init__(self):
        self.dataset = "dataset/interview.csv"

    def preprocess(self):
        
        df = pd.read_csv(self.dataset)
        
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
        

        X.isnull().sum().sort_values(ascending=False).head()
        
        X = X.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Date of Interview', 'Name(Cand ID)'], 1)
        print("Removing useless features")
        with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
             print(X)
        X.isnull().sum().sort_values(ascending=False).head()
        
        X.to_csv('dataset/preprocessed_data.csv', encoding='utf-8', header=False, index=False)
        y.to_csv('dataset/objective_data.csv', encoding='utf-8', header=False, index=False)
