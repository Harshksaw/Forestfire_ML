import pickle

from flask import Flask,request , jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#import ridge regressor model and standard scler pickle
ridge_model = pickle.load(open('./models/ridge.pkl' , 'rb'))
standard_scaler = pickle.load(open('./models/scaler.pkl' , 'rb'))

#route for homepage

# @app.route('/')
# def index():
#     return render_template('index.html')


@app.route('/' ,methods = ['GET' , 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = request.form.get('Classes')
        region = request.form.get('Region')
        
        new_data_scaled = standard_scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html' , result = result[0])



    else:
        return render_template('home.html')

















if __name__=="__main__":
    app.run(host="0.0.0.0")
