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


@app.route('/api', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get data
       # data = request.files['test.json'] #we use this method with <form>html tag
        data = request.get_json(force=True)
       #data = request.json
       
        # convert data into dataframe
        #data.update((x, [y]) for x, y in data.items())
        
        # data_df = pd.DataFrame.from_dict(data, orient='index')
        # data_df.reset_index(level=0, inplace=True)
        data = json.loads(data)
        data = np.array(data)
        data = data.reshape(-1,1)
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


