from ctypes.wintypes import HICON
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def heartTotal(l1):
    
    def listExtender(l3):
        if l3[2] == 0:
            l3.extend([0, 0, 0])
        elif l3[2] == 1:
            l3.extend([1, 0, 0])
        elif l3[2] == 2:
            l3.extend([0, 1, 0])
        else:
            l3.extend([0, 0, 1])
        if l3[5] == 0:
            l3.extend([0, 0])
        elif l3[5] == 1:
            l3.extend([1, 0])
        else:
            l3.extend([0, 1])
        if l3[9] == 0:
            l3.extend([0, 0])
        elif l3[9] == 1:
            l3.extend([1, 0])
        else:
            l3.extend([0, 1])
        if l3[10] == 0:
            l3.extend([0, 0, 0, 0])
        elif l3[10] == 1:
            l3.extend([1, 0, 0, 0])
        elif l3[10] == 2:
            l3.extend([0, 1, 0, 0])
        elif l3[10] == 3:
            l3.extend([0, 0, 1, 0])
        else:
            l3.extend([0, 0, 0, 1])
        if l3[11] == 0:
            l3.extend([0, 0, 0])
        elif l3[11] == 1:
            l3.extend([1, 0, 0])
        elif l3[11] == 2:
            l3.extend([0, 1, 0])
        else:
            l3.extend([0, 0, 1])
        l3.pop(11)
        l3.pop(10)
        l3.pop(9)
        l3.pop(5)
        l3.pop(2)
        return l3
    
    def listFinalForm(l1):
        return [l1[0], l1[3], l1[4], l1[6], l1[8]]
    
    def listTransition(l1, l2):
        l2[0] = l1[0][0]
        l2[3] = l1[0][1]
        l2[4] = l1[0][2]
        l2[6] = l1[0][3]
        l2[8] = l1[0][4]
        return l2
    
    def scaleAndChange(l1):
        df = pd.read_csv('heart.csv')
        df.drop_duplicates(inplace = True)
        df.drop("fbs", axis = 1, inplace = True)
        dataset = pd.get_dummies(df, columns = ['cp', 'restecg', 'slope', 'ca', 'thal'], drop_first=True)
        sc = StandardScaler()
        columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        dataset[columns_to_scale] = sc.fit_transform(dataset[columns_to_scale])
        l2 = listFinalForm(l1)
        l2 = sc.transform([l2])
        l1 = listTransition(l2, l1)
        l1 = listExtender(l1)
        return l1

    return scaleAndChange(l1)

def diabetesTotal(l1):
    column_names = ["Pregnancies", "Glucose", "BPressure", "Skinfold", "Insulin", "BMI", "Pedigree", "Age", "Class"]
    df = pd.read_csv("data.csv", names = column_names)
    df[['Glucose','BPressure','Skinfold','Insulin','BMI']] = df[['Glucose','BPressure','Skinfold','Insulin','BMI']].replace(0,np.NaN)
    df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
    df['BPressure'].fillna(df['BPressure'].mean(), inplace = True)
    df['Skinfold'].fillna(df['Skinfold'].median(), inplace = True)
    df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
    df['BMI'].fillna(df['BMI'].median(), inplace = True)
    y = df.pop('Class')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    l1 = sc.transform([l1])
    return l1

def decide_model(values):
    if len(values) == 8:
        loaded_model = pickle.load(open("models/diabetes.pkl", "rb"))
        values = diabetesTotal(values)
        result = loaded_model.predict(values)
        if result[0]:
            return "Hmmm, Looks uneasy Suggesting a doctor's consultation"
        else:
            return "You are Healthy"

    elif len(values) == 12:
        loaded_model = pickle.load(open("models/heart.pkl", "rb"))
        values = heartTotal(values)
        result = loaded_model.predict([values])
        if result[0]:
            return "You are Healthy"
        else:
            return "Hmmm, Looks uneasy Suggesting a doctor's consultation"
        
    elif len(values) == 10:
        loaded_model = pickle.load(open("models/kidney.pkl", "rb"))
        print(values)
        result = loaded_model.predict([values])
        print(values)
        if result[0]:
            return "Hmmm, Looks uneasy Suggesting a doctor's consultation"
        else:
            return "You are Healthy"

@ app.route("/")
def home():
    return render_template('home.html')

@ app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@ app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@ app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@ app.route("/predict", methods=['GET', 'POST'])
def predictPage():
    if request.method == 'POST':
        features = [x for x in request.form.values()]
        pred = decide_model(features)
        return render_template('output.html', pred=pred)

if __name__ == '__main__':
       app.run(host='localhost', port=5000, debug=True)