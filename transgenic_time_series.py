import pandas as pd 
import numpy as np 
from pandas import read_csv 
from statsmodels.tsa.stattools import adfuller 
from pmdarima import auto_arima 

import warnings 
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_squared_error 
from math import sqrt 


df = read_csv('main.csv', index_col='DATE') 
df2 = read_csv('imports-oil-smp-bdi.csv', index_col='DATE') 
df3 = pd.concat([df, df2.reindex(df.index)], axis=1) 
df3.to_csv('Main2.csv') 

 

# Reforms the IMF dataset to only include total import data for every country. 

 
df = pd.read_csv('bigboi.csv') 
df[:] = df[:].fillna(0) 
df = df.drop(columns=['WEO Country Code','ISO','WEO Subject Code', 'Subject Notes', 'Units', 'Scale','Country/Series-specific Notes','2022','2023','2024','2025','2026','Estimates Start After']) 
df = df.drop([8775,8776]) 
df2 = df.Country 
df['Subject Descriptor'][1:20] 
df.loc[df['Subject Descriptor'] != ('Volume of Imports of goods' or 'Volume of Exports of goods'), :] = np.nan 
df.head(40) 
df = df.dropna() 
df.to_csv('bigboi2.csv') 

# Runs the ADF test for the time-series, auto-ARIMA  and the algorithm that creates the transgenic time-series. 

 
 
def ad_test(dataset): 
    dftest = adfuller(dataset, autolag = 'AIC') 
    print("1. ADF : ",dftest[0]) 
    print("2. P-value : ", dftest[1]) 
    print("3. Num Of Lags : ", dftest[2]) 
    print("4. Num of Observations Used For ADF regression and Critical Values Calculation : ", dftest[3]) 
    print("5. Critical Values : ", dftest[4]) 
    for key, val in dftest[4].items(): 
        print("\t",key, ": ", val) 
 
 
oil = pandas.read_csv('Crude Oil Monthly.csv', header=None, names=['Date', 'Value'], index_col='Date', parse_dates=True) 
 
a = adfuller(oil, autolag = 'AIC') 
ad_test(oil) 
 

 
print([oil.describe()]) 
trainoil = oil[:190] 
testoil = oil[190:203] 
rmse = 40 
pick = 40 
start = len(trainoil) 
end = len(trainoil) + len(testoil) - 1 
warnings.filterwarnings("ignore") 
step_fit_oil = auto_arima(trainoil['Value'], trace=True, suppress_warnins=True) 
 
for i in range(185): 
    for k in range(5+i, 185): 
        train1 = trainoil[0:i] 
        train2 = trainoil[k:-1] 
        drame = [train1, train2] 
        train3 = pd.concat(drame) 
        modeloil = ARIMA(train3['Value'], order=(0, 1, 1)) 
        modeloil = modeloil.fit() 
        pred = modeloil.predict(start=start, end=end, type='levels') 
        pred.index = oil.index[start:end+1] 
        rmse = sqrt(mean_squared_error(pred, testoil['Value'])) 
        # print(i,k) 
        if rmse < pick: 
            starting = i 
            ending = k 
            pick = rmse 
            print('got 1!') 
            print('First cut at', starting,'Second cut at', ending,'RMSE = ', pick) 
