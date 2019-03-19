from sklearn.metrics import mean_squared_error
from math import sqrt
import json

def predict(model, df2):
  print(df2)
  predictions = model.predict(start=len(df2.columns)-7, end=len(df2.columns)-1)
  # rmse = sqrt(mean_squared_error(df2['menu'][len(df2.column)-7:].values, predictions))

  predictions = predictions.tolist()
  data = {
    "predictions" : predictions,
    "rmse"        : "ppp"
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
    "rmse"        : "There is no rmse"
  }

  jsonData = json.dumps(data)
  return jsonData