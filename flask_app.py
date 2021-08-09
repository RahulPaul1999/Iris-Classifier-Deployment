# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 13:53:55 2021

@author: PRIYANKA
"""

from flask import Flask, request, render_template
import numpy as np
import pickle

from sklearn.datasets import load_iris
iris = load_iris()
X= iris.data
from sklearn.preprocessing import StandardScaler # import the scaler
scaler = StandardScaler() # initiate it
scaled_X = scaler.fit(X)


app = Flask(__name__)

model= pickle.load(open('iris_classifier.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("AppPage.html")
    

@app.route('/predict', methods=["POST"])
def predict_iris_species():  
    if request.method == "POST":
        sepal_length= request.form['sepal_length']
        print(sepal_length) 
        sepal_width = request.form['sepal_width']
        print(sepal_width) 
        petal_length = request.form['petal_length']
        print(petal_length)
        petal_width = request.form['petal_width']
        print(petal_width)
        array = np.array([[sepal_length , sepal_width , petal_length , petal_width]])
        array = scaler.transform(array)
        # array = np.array([[-4.26739366e-01,  2.72483973e+00, -1.34658875e+00, -1.30814890e+00]])
        prediction = model.predict(array)
        print("Prediction: ", int(prediction))
        # return "Predicted value is "+ str(prediction)
        return render_template("PredictPage.html", data = prediction)


if __name__ == '__main__':
    app.run()