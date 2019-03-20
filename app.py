from flask import Flask, render_template, request
from datetime import date
import predict as pr
import model as md
import requests
import pickle
import json
import os
from pandas import DataFrame

app = Flask(__name__)

# r = requests.get('http://dummy.restapiexample.com/api/v1/employees')
# dataApi = json.loads(r.text)
# df = DataFrame(dataApi)
# pr.showDf(df)

pathData = 'data/dataset.csv'
df = md.proccessData(pathData)

listModels = os.listdir('models/') 

if (len(listModels) == 0 or listModels[0][:10] != str(date.today())):
    md.arimaModel(df) 

dictModels = pickle.load(open('models/dictModels.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    data = pr.predict(dictModels, df)
    data = json.loads(data)
    return render_template('predict.html', data=data)

@app.route('/forecast')
def forecast():
    data = pr.forecast(dictModels, df)
    data = json.loads(data)
    return render_template('forecast.html', data=data)

if __name__ == '__main__':
    app.run(port=5000, debug=True)