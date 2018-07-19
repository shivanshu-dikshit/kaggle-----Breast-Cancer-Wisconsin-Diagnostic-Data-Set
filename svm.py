# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:17:15 2018

@author: shivanshu_dikshit
"""

###importing the necessary libraries to be used in the process of analyzing the data and creation of model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


###importing data
data = pd.read_csv("data.csv")


"""
now as we have imported the data, it's time to better understand the data like what is the shape,
size,features etc in our data
"""
print(data.head(5))
print("----------------------______________________________________---------------------------------")
print(data.info())
print("----------------------______________________________________---------------------------------")
print(data.shape)


"""
as we can understand from the above findings that there are 569 enteries in the data and a total of
33 columns
there are not any missing values, just a column "unnamed:32" with  "Nan" values
there are some of the columns like "unnamed:32", "id" which are not needed for the prediction so we
are going to drop them
"""
data.drop("id", axis = 1, inplace = True)
print(data.columns)
print(data.info())


data.drop("Unnamed: 32", axis = 1, inplace = True)
print(data.columns)
print(data.shape)

###now as we can see diagnosis is of object type so we can convert in into numeric type
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})


print(data.diagnosis)


print(data.head())

y = data['diagnosis']
data.drop('diagnosis', axis = 1, inplace = True)

print(y.shape)
print(data.shape)

#now splitting our dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(data, y, random_state = 0, test_size = 0.3)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
##svm
mac = SVC()
model = mac.fit(X_train, y_train)

prediction = model.predict(X_test)

svm_accuracy = accuracy_score(y_test, prediction)
print(svm_accuracy)
# accuracy of 0.9766081871345029   with no scaling 0.631578947368421


svm_matrix = confusion_matrix(y_test, prediction)
print(svm_matrix)

# matrix[[107   1]                           no scaling  [[108   0]
#       [  3  60]]                                       [ 63   0]]
