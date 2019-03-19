from flask import Flask, render_template, request
import predict as pr
import json

app = Flask(__name__)

pathData = 'data/dataset.csv'
data = pr.main(pathData)
data = json.loads(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html', rmse=data["rmse"], preds=data["predictions"])

if __name__ == '__main__':
    app.run(port=5000, debug=True)