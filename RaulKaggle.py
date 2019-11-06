# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:17:54 2019

@author: ryanl
"""

#Rauls Kaggle

# Kaggle Regression

# Import Libraries

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#train.describe()
#test.describe()

# Assign IDs and fill null values
def processing(dat):
    
    ID = dat['Id']
    del dat['Id']
    
    dat = dat.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('None'))
    
    return ID, dat

# Clean Data
Id_train,train = processing(train)
Id_test,test = processing(test)

# Assign KPI of dataset
x_train = train.drop(columns = 'SalePrice')
y_train = train['SalePrice']

# Create Dummy variables and standardize test data
sc = StandardScaler()
x_train = pd.get_dummies(x_train)
a = np.array(x_train.columns.values)
sc.fit_transform(x_train)
x_train = sc.transform(x_train)
x_train = pd.DataFrame(data = x_train,columns = a)

# Create Random forest Regression model
regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_train)

# Performance on training dataset
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# Run Random Forest on 
temp_test = pd.get_dummies(test)
x_test = pd.DataFrame(np.zeros((temp_test.shape[0],len(a))),columns = a)
x_test.update(temp_test)
x_test = pd.DataFrame(data = sc.transform(x_test),columns = a)

# Create Output
Out = pd.DataFrame(Id_test)
Out['SalePrice'] = regressor.predict(x_test)
Out.to_csv('raulkaggle.csv', index=False)