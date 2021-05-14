
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
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
print("XVARS")
print(ydata.head())     # prints y Data

hrcol = dataset['hour']
windows = { 'window1' : {'min' : 6, 'max': 11},
            'window2' : {'min' : 12,'max': 17},
            'window3' : {'min' : 18,'max': 22} }

wcol = []
windowlist = []
for hr in hrcol:
    wdict = {}
    for wd in windows.keys():
        wdict[wd] = int(0)

    for seg, range in windows.items():
        if (hr > range['min']) & (hr <= range['max']):
            wdict[seg] = int(1)
            print("Hr: {} / Seg: {}/ \n\t{}".format(hr, seg, wdict))
            break
    windowlist.append(wdict)

print(json.dumps(windowlist, indent=2))
dataset = dataset.append(windowlist)
print(dataset.head(24))

