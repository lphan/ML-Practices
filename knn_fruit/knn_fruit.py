# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:30:49 2017

@author: lphan
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset.csv')
print (dataset.columns)
# replace 0-value by numpy NaN
dataset = dataset.replace(0, np.NaN)

# importing the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Replace the missing data (value 0.0) with mean value of all other existing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data (this case: color in column 0)
# encoding the independent variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()

# encoding the dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the dataset into the training set and test set
# from sklearn.cross_validation import train_test_split (deprecated)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Feature scaling - Normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting to the training set using k-NN
from sklearn.neighbors import KNeighborsClassifier
classifier_k3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean', p=2)
classifier_k3.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier_k3.predict(X_test)

# find total errors/ correct by using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print ("Confusion matrix")
print (cm)
print ("Predicted value y_pred", y_pred)
print ("Actual value y_test", y_test)

# Classify the inputdata & visualize them 
inputdata = pd.read_csv('classify_data.txt', sep='\s+', 
                        names=['Color', 'Radius (cm)', 'Weight (grams)'])

# importing data into object
X_input = inputdata.iloc[:, [0,1,2]].values

# preprocessing data
labelencoder_X_input = LabelEncoder()
X_input[:, 0] = labelencoder_X_input.fit_transform(X_input[:, 0])

onehotencoder = OneHotEncoder(categorical_features= [0])
X_input = onehotencoder.fit_transform(X_input).toarray()

# feature scaling 
X_input = sc_X.transform(X_input)

# Predicting the input set results
y_pred_input_k3 = classifier_k3.predict(X_input)

# ------------- Predicting the input set with k=1 -------------------------
# classifier_k1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean', p=2)
# classifier_k1.fit(X_train, y_train)
# y_pred_input_k1 = classifier_k1.predict(X_input)

# TODO: visualization
