import pandas as pd
import numpy as np

from flask import Flask, request, render_template

import pickle

app = Flask(__name__)
app.secret_key = 'ML Project'


def get_values():
    v = {'manufacturer': ['acura', 'alfa-romeo', 'aston-martin', 'audi', 'bmw', 'buick', 'cadillac', 'chevrolet',
                             'chrysler', 'dodge', 'fiat', 'ford', 'gmc', 'harley-davidson', 'hennessey', 'honda',
                             'hyundai', 'infiniti', 'jaguar', 'jeep', 'kia', 'land rover', 'lexus', 'lincoln', 'mazda',
                             'mercedes-benz', 'mercury', 'mini', 'mitsubishi', 'nissan', 'pontiac', 'porsche', 'ram',
                             'rover', 'saturn', 'subaru', 'tesla', 'toyota', 'unknown', 'volkswagen', 'volvo'],
            'condition': ['excellent', 'fair', 'good', 'like new', 'salvage'],
            'cylinders': ['3', '4', '5', '6', '8', '10',
                          '12 cylinders', 'unknown'],
            'fuel': ['diesel', 'electric', 'gas', 'hybrid', 'other'],
            'title_status': ['clean', 'rebuilt', 'salvage'],
            'transmission': ['automatic', 'manual', 'other'],
            'type': ['SUV', 'bus', 'convertible', 'coupe', 'hatchback', 'mini-van', 'offroad', 'other', 'pickup',
                     'sedan', 'truck', 'van', 'wagon'],
            'state': ['midwest', 'northeast', 'south', 'west'],
            'drive': ['2', '4']}
    return v


def get_submitted_values():
    submitted_values = {"year": int(request.form["year"]), "manufacturer": request.form["manufacturer"],
                        "condition": request.form["condition"], "cylinders": int(request.form["cylinders"]),
                        "fuel": request.form["fuel"], "odometer": int(request.form["odometer"]),
                        "title_status": request.form["title_status"], "transmission": request.form["transmission"],
                        "drive": request.form["drive"], "type": request.form["type"],
                        "state": request.form["state"]}
    submitted_values1 = submitted_values.copy()
    submitted_values["year"] = ((submitted_values["year"] - 1900) / (2020 - 1900))
    submitted_values["odometer"] = ((submitted_values["odometer"] - 0) / (10000000 - 0))
    submitted_values["cylinders"] = ((submitted_values["cylinders"] - 0) / (6 - 0))
    df = pickle.load(open(r'D:\All_Docs\Masters\CS 584 Machine Learning\Project\Code\df.pkl', 'rb'))
    numbers = ['year', 'cylinders', 'odometer', 'drive']
    for i in list(submitted_values.keys()):
        if i in numbers:
            df[i] = submitted_values[i]
        else:
            df[f'{i}_{submitted_values[i]}'] = 1
    return df, submitted_values1


def prediction(df):
    model = pickle.load(
        open(r'D:\All_Docs\Masters\CS 584 Machine Learning\Project\Code\GradienBoostingModel.pkl', 'rb'))
    result = model.predict(df)
    return np.expm1(result)


@app.route('/')
def home():
    return render_template('index.html', values=get_values(), run=False)


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        car_details, submitted_values = get_submitted_values()
        result = prediction(car_details)
        return render_template('index.html', prediction_text='The best price is $ {}'.format(round(result[0])),
                               values=get_values(), run=True, submitted_values=submitted_values)


if __name__ == "__main__":
    app.run(debug=True)
