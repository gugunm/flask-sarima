from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import json
import math
import os

def predict(dictModels, df2):
  listResult = []
  for column in df2:
    modelName = dictModels[column]
    model = pickle.load(open('models/'+modelName,'rb'))
    predictions = model.predict(start=len(df2[column].values)-7, end=len(df2[column].values)-1)
    rmse = sqrt(mean_squared_error(df2[column][len(df2[column].values)-7:].values, predictions))
    if math.isinf(rmse):
      rmse = 0

    predictions = predictions.tolist()
    predictions = [0 if i < 0 else i for i in predictions]
    data = {
      "menu"        : column,
      "predictions" : predictions,
      "rmse"        : rmse
    }
    listResult.append(data)
  allData = {
    "data" : listResult
  }
  jsonData = json.dumps(allData)
  return jsonData

def forecast(dictModels, df2):
  listResult = []
  for column in df2:
    modelName = dictModels[column]
    model = pickle.load(open('models/'+modelName,'rb'))
    predictions = model.forecast(7)

    predictions = predictions.tolist()
    predictions = [0 if i < 0 else i for i in predictions]
    predictions = [math.floor(i) if i-math.floor(i) < 0.5 else math.ceil(i) for i in predictions]
    data = {
      "menu"        : column,
      "predictions" : predictions,
      "rmse"        : "There is no rmse"
    }
    listResult.append(data)
  allData = {
    "data" : listResult
  }
  jsonData = json.dumps(allData)
  return jsonData