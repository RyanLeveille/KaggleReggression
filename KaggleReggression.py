# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:37:36 2019

@author: ryanl
"""

# Bubble sorting #overfitting, early stopping (practice later)


###########################################################################

# Kaggle Regression

# Import Libraries

import numpy as np
import pandas as pd

# Import Data

train = pd.read_csv("train.csv")

# Descriptive Stats
train.describe()
#                Id   MSSubClass  ...       YrSold      SalePrice
#count  1460.000000  1460.000000  ...  1460.000000    1460.000000
#mean    730.500000    56.897260  ...  2007.815753  180921.195890
#std     421.610009    42.300571  ...     1.328095   79442.502883
#min       1.000000    20.000000  ...  2006.000000   34900.000000
#25%     365.750000    20.000000  ...  2007.000000  129975.000000
#50%     730.500000    50.000000  ...  2008.000000  163000.000000
#75%    1095.250000    70.000000  ...  2009.000000  214000.000000
#max    1460.000000   190.000000  ...  2010.000000  755000.000000

# Set up Model

X = train.iloc[:, 0:79].values
y = train.iloc[:, 80].values

# Finally, let's divide the data into training and testing sets:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Model

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluate the Model

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

