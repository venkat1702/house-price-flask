from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    df = pd.read_csv('dataset2.csv')
    x = df[['total_sqft','bhk','balcony']]
    y = df[["price"]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
    #Naive Bayes Classifier
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    if request.method == 'POST':
        area=int(request.form['area'])
        bhk=int(request.form['bhk'])
        balcony=int(request.form['balcony'])
        data=([[area,bhk,balcony]])
        my_prediction = lm.predict(data)
        result=float(my_prediction)
    return render_template('index.html',prediction = result)

if __name__ == '__main__':
	app.run(debug=True,port=50001)