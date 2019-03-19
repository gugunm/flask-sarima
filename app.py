from flask import Flask, render_template, request
import predict as pr

app = Flask(__name__)

pathData = 'data/dataset.csv'
predictResult, rmse = pr.main(pathData)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html', rmse=rmse, preds=predictResult)

if __name__ == '__main__':
    app.run(port=8080, debug=True)