# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Training the Multiple Linear Regression model on the Training set
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
ydata = dataset['USAGEW'].values
xdata = dataset[['MORNING', 'AFTERNOON', 'EVENING']].values

print("XVARS")
print(xdata)    # prints X data
print(ydata)     # prints y Data

# Encoding categorical data
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3,4])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
#X = X.reshape(-1, 4)# do array size mathching, 1x2 , 2 coz there are 2 variables
#y = y.reshape(-1,4) # do array size matching , 1x2
#print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
scaler = MinMaxScaler()

xdata = xdata.reshape(-1,4)
ydata = ydata.reshape(-1,4)

xscaled = scaler.fit_transform(xdata)
yscaled = scaler.fit_transform(ydata)

#model = sm.OLS(ydata, xdata).fit()
regressor = LinearRegression() #instnatiate the Linear Regressor class
#regressor = LogisticRegression()
#regressor.fit(Xtrain, y_train) #fit the model using the complete data
regressor.fit(ydata, xdata) #fit the model using the complete data
#regressor.fit(xscaled, yscaled) #fit the model using the complete data
#print(model.summary())

# Predicting the Test set results
print("=====")
print("Reg Score: {}".format(regressor.score(xdata, ydata)))
print('Coefficients: {}'.format(regressor.coef_)) #print coefficients
print("Intercept: {}".format(regressor.intercept_)) #print intercept

