from flask import Flask, render_template, request
import predict as pr
import model as md
import requests
import pickle
import json
import os
from pandas import DataFrame

app = Flask(__name__)

r = requests.get('http://dummy.restapiexample.com/api/v1/employees')
dataApi = json.loads(r.text)

pathData = 'data/dataset.csv'
df = md.proccessData(pathData)
if not os.path.isfile('model.pkl'):
    md.modeling(df) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # df = DataFrame(dataApi)
    # pr.showDf(df)
    sarimaModel = pickle.load(open('model.pkl','rb'))
    data = pr.predict(sarimaModel, df)
    print(data)
    
    data = json.loads(data)
    return render_template('predict.html', rmse=data["rmse"], preds=data["predictions"])

@app.route('/forecast')
def forecast():
    sarimaModel = pickle.load(open('model.pkl','rb'))
    data = pr.forecast(sarimaModel)
    
    data = json.loads(data)
    return render_template('forecast.html', rmse=data["rmse"], preds=data["predictions"])

if __name__ == '__main__':
    app.run(port=5000, debug=True)