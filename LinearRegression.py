# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Importing the dataset
#dataset = pd.read_csv('data/50_Startups-Data.csv')
dataset = pd.read_csv('data/bobblehead-01.csv' )
print(dataset.describe())

X = dataset.iloc[:,3:4 ].values
y = dataset.iloc[:,5].values

print(X)
print(y)

print("=====")

# Encoding categorical data
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3,4])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
X = X.reshape(-1,2)
y = y.reshape(-1,2)
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X, y)


# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#print(y_pred)

#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print('Coefficients: {}'.format(regressor.coef_))
print("Intercept: .{}".format(regressor.intercept_))
print("R2Score: {}".format(r2_score(y_test, y_pred) ))
