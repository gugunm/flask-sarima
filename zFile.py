from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics  
from pandas import datetime
from datetime import date
from math import sqrt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import itertools
import warnings
import dateutil
import pickle
import math
import json

def showDf(df):
  print(df)

def readData(pathData):
    df = pd.read_csv(pathData)
    df = df.iloc[:,:4]
    df = df.fillna(method='ffill')
    df['Date'] = df['Date'].apply(dateutil.parser.parse).dt.date
    df = df.iloc[:,:-1]
    return df

def coutMenu(df):
  uniqueDate = df['Date'].unique()
  arr_date = []
  for d in uniqueDate:
    arr_date.append(d)

  for i, dt in enumerate(arr_date):
    dt2 = df[df['Date'] == dt]
    dt2 = dt2.iloc[:,1:2]  
    dt2 = dt2['Menu Name'].value_counts()

    dfDate = pd.DataFrame(dt, index=range(len(dt2.values)), columns=range(1))
    dt2 = dt2.reset_index()
    dt2 = dt2.rename(columns={'Menu Name': 'total', 'index': 'menu'})
    dt2 = pd.concat([dfDate, dt2], axis=1)
    dt2 = dt2.rename(columns={0: 'Date'})
    if i == 0:
      df2 = dt2
    else:
      df2 = pd.concat([df2, dt2], ignore_index=True) #.append(dt2) #pd.concat([df2, dt2], axis=0)

  df2 = df2
  df2.head()
  return df2

def transformDf(df):
  listMenu = list(df['menu'].unique())
  listDate = list(df['Date'].unique())

  dfMenu = pd.DataFrame(listMenu)
  dfDate = pd.DataFrame(0, index=np.arange(len(listMenu)), columns=listDate)
  df2 = pd.concat([dfMenu, dfDate], axis=1)
  df2 = df2.rename(columns={0: 'menu'}).set_index('menu')

  for column in df2:
    oneDate = df[df['Date'] == column]
    oneDate = oneDate.iloc[:,1:].set_index('menu').T

    oneDateColumn = list(oneDate.columns.values)

    for i_menu, row in df2[column].iteritems():
      if i_menu in oneDateColumn:
        df2.loc[i_menu].at[column] = oneDate.iloc[0][i_menu]

  return df2.T

def min_pdq(train, pdq):
  params = {}
  for param in pdq:
    try:
      model_arima = ARIMA(train, order=param)
      model_arima_fit = model_arima.fit()
      if param not in params and model_arima_fit.aic:
        params[param] = model_arima_fit.aic
    except:
      continue
      
  try:
    minimum_pdq = min(params, key=params.get)
  except:
    minimum_pdq = (0,0,0)

  return minimum_pdq

def arimaModel(df2):
  models = {}
  # df2 = df.copy()
  # df2 = df2.T
  # df2 = df2.iloc[:,:3]

  for num, column in enumerate(df2):
    print('No. Product : ', num, column, type(column))
    sales_diff = df2[column].diff(periods=1)  # integreted order 1
    sales_diff = sales_diff[1:]
    
    X = df2[column].values
    train = X[0:len(df2[column].values)-7]
    predictions = []
    
    # p=d=q=range(0,5)
    # pdq = list(itertools.product(p,d,q))
    # use_pdq = min_pdq(train, pdq)
    # print(use_pdq)
    
    # ========= S A R I M A X ============
    mod = sm.tsa.statespace.SARIMAX(
      train, 
      order=(0,1,0), 
      seasonal_order=(1,1,0,7),
      enforce_stationarity=False,
      enforce_invertibility=False)
    model_fit = mod.fit(disp=0)
    # ====================================
    
    predictions = model_fit.predict(start=len(df2[column].values)-7, end=len(df2[column].values)-1)
    rmse = sqrt(mean_squared_error(df2[column][len(df2[column].values)-7:].values, predictions))

    # if column not in result:
    #   result[column] = ', '.join(map(str, predictions)) 
    # print(predictions)

    fileName = str(date.today())+'('+column+')'+'.pkl'
    pickle.dump(model_fit, open('models/'+fileName,'wb'))
    models[column] = fileName
  return predictions, rmse, models
  # return predictions, rmse

def main(pathData):
  warnings.filterwarnings('ignore')
  df = readData(pathData)
  df = coutMenu(df)
  df2 = transformDf(df)
  predictions, rmse, models = arimaModel(df2)
  predictions = predictions.tolist()
  # print(type(predictions))
  data = {
    "predictions" : predictions,
    "rmse"        : rmse
  }

  jsonData = json.dumps(data)
  return jsonData

def forecast(model):
  # model = pickle.load(open('model.pkl','rb'))
  # predictions = model.predict(start=len(df2[column].values), end=len(df2[column].values))
  predictions = model.forecast(7)
  predictions = predictions.tolist()
  # print(type(predictions))
  data = {
    "predictions" : predictions,
    # "rmse"        : rmse
  }

  jsonData = json.dumps(data)
  return jsonData

pathData = 'data/dataset.csv'
hasil = main(pathData)

print(hasil)