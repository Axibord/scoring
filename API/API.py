#imports
import pickle
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash

# load model
model = pickle.load(open('model.pkl', 'rb'))

#ALLOWED_EXTENSIONS = {'json'}
# app
app = Flask(__name__)

# routes

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # get data
        #data = request.files['file'] #we use this method with <form>html tag
        #data = request.get_json(force=True)
        data = request.json
       
        # convert data into dataframe
        # data = json.loads(data)
        data = pd.DataFrame(data)
        data.reset_index(level=0, inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0,inplace=True)
        #scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        # predictions
        result = model.predict(data_scaled)
        
    
        # transform it to dict and send back to browser
        result_dict_output = dict(enumerate(result))
     
    # return data
    return jsonify(results = result_dict_output)



# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(f.filename)
#       return redirect(url_for())


if __name__ == '__main__':
    app.run(port = 5000, debug=True)


