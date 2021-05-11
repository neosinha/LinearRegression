# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Importing the dataset
dataset = pd.read_csv('data/bobblehead-01.csv' )
print(dataset.describe())

#X = dataset.iloc[:,3:4 ].values #X variables, data is in col3 and 4
#y = dataset.iloc[:,5].values #Y var

ydata = dataset['Unit Sales']
xdata = dataset[['AdSpend']]

print(xdata.head())    # prints X data
print(ydata.head())     # prints y Data


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
#xscaler = MinMaxScaler()

#xdata = xdata.reshape(-1,1)
#ydata = ydata.reshape(-1,1)

#xscaled = scaler.fit_transform(xdata)
#yscaled = scaler.fit_transform(ydata)

regr = LinearRegression(fit_intercept=True, ) #instnatiate the Linear Regressor class
regr.fit(xdata, ydata) #fit the model using the complete data


# Predicting the Test set results
print("=====")
print("Reg Score: {}".format(regr.score(xdata, ydata)))
print("Intercept: {}".format(regr.intercept_)) #print intercept
print("Coefficients")
for idx, col in enumerate(xdata.columns):
    print('\t{}: {}'.format(col, regr.coef_[idx])) #print coefficients

