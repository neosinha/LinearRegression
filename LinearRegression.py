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

X = dataset.iloc[:,3:4 ].values #X variables, data is in col3 and 4
y = dataset.iloc[:,5].values #Y var

xvars = dataset['AdSpend']
yvar = dataset['Unit Sales']

print("XVARS")
print(xvars)    # prints X data
print(yvar)     # prints y Data

print("=====")

# Encoding categorical data
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3,4])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
X = X.reshape(-1,2) # do array size mathching, 1x2 , 2 coz there are 2 variables
y = y.reshape(-1,2) # do array size matching , 1x2

print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

regressor = LinearRegression() #instnatiate the Linear Regressor class
#regressor = LogisticRegression()
regressor.fit(X_train, y_train) #fit the model using the complete data


# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#print(y_pred)

#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("=====")
print('Coefficients: {}'.format(regressor.coef_[0])) #print coefficients
print('Coefficients: {}'.format(regressor.coef_[1]))
print("Intercept: {}".format(regressor.intercept_[0])) #print intercept
print("R2Score: {}".format(r2_score(y_test, y_pred) )) #print R2

