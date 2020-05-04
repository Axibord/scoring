# imports
import pickle
import pandas as pd
import joblib
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash

# load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.joblib',mmap_mode='r+')

# ALLOWED_EXTENSIONS = {'json'}
# app
app = Flask(__name__)

# routes

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # get data
        data = request.json

        # convert data into dataframe
        data = pd.DataFrame(data)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)
        
        # scale data
       
        data_scaled = scaler.transform(data)

        # predictions
        result = model.predict(data_scaled)
        result = result.astype(float)

        # transform it to dict and send back to browser
        result_dict_output = dict(enumerate(result))

    # return data
    return jsonify(results=result_dict_output)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
     
      data = request.files['file'] # receive data
      f = data.save('upload.json') 
      f = open('upload.json')
      data = json.load(f)
      
      
    # convert data into dataframe
      data = pd.DataFrame(data)
      data.replace([np.inf, -np.inf], np.nan, inplace=True)
      data.fillna(0,inplace=True)

      # scale data
      data_scaled = scaler.transform(data)

      # predictions
      result = model.predict(data_scaled)
      result = result.astype(float)

      # transform it to dict and send back to browser
      result_dict_output = dict(enumerate(result))

      return jsonify(results=result_dict_output)


if __name__ == '__main__':
    app.run(port = 5000, debug=True)


