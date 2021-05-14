
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Importing the dataset
dataset = pd.read_excel('data/swactivity-0510.xlsx',
                        engine='openpyxl')

print(dataset.describe())
#X = dataset.iloc[:,3:4 ].values #X variables, data is in col3 and 4
#y = dataset.iloc[:,5].values #Y var
print(dataset.columns)


ydata = dataset['USAGEW']
xdata = dataset[['MORNING', 'AFTERNOON', 'EVENING']]

print("XVARS")
print(xdata.head())    # prints X data
print(ydata.head())     # prints y Data

regressor = LinearRegression(fit_intercept=True) #instnatiate the Linear Regressor class
regressor.fit(xdata, ydata) #fit the model using the complete data
#regressor.fit(xscaled, yscaled) #fit the model using the complete data
#print(model.summary())

# Predicting the Test set results
print("=====")
print("Reg Score: {}".format(regressor.score(xdata, ydata)))
print("Intercept: {}".format(regressor.intercept_)) #print intercept
print('Coefficients==') #print coefficients
for idx, ctrlpt in enumerate(xdata.columns):
    beta = "{}".format(regressor.coef_[idx], '000.3f')
    print("{} : {}".format(ctrlpt, beta ))


print(dataset.columns)
controlpoints = list(dataset.columns[10:])
print(controlpoints)
#xdata=dataset[[]]
xdata=dataset[controlpoints]

print(xdata.head())
regressor = LinearRegression(fit_intercept=True) #instnatiate the Linear Regressor class
regressor.fit(xdata, ydata) #fit the model using the complete data
print("=====")
print("Reg Score: {}".format(regressor.score(xdata, ydata)))
print("Intercept: {}".format(regressor.intercept_)) #print intercept

print('Coefficients') #print coefficients
for idx, ctrlpt in enumerate(controlpoints):
    beta = "{}".format(regressor.coef_[idx], '000.3f')
    print("{} : {}".format(ctrlpt, beta ))



## Algorithm
print(dataset.columns)
controlpoints = list(dataset.columns[9:])
print(controlpoints)
#xdata=dataset[[]]
xdata=dataset[controlpoints]

print(xdata.head())
regressor = LinearRegression(fit_intercept=True) #instnatiate the Linear Regressor class
regressor.fit(xdata, ydata) #fit the model using the complete data
print("=====")
print("Reg Score: {}".format(regressor.score(xdata, ydata)))
print("Intercept: {}".format(regressor.intercept_)) #print intercept

print('Coefficients') #print coefficients
for idx, ctrlpt in enumerate(controlpoints):
    beta = "{}".format(regressor.coef_[idx], '000.3f')
    print("{} : {}".format(ctrlpt, beta ))

