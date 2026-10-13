from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


# Must match the notebook exactly, and be defined BEFORE pickle.load
def add_features(X):
    X = X.copy()
    X["Glucose_BMI"]  = X["Glucose"] * X["BMI"]
    X["Glucose_Age"]  = X["Glucose"] * X["Age"]
    X["Age_Preg"]     = X["Age"] * X["Pregnancies"]
    X["Insulin_miss"] = X["Insulin"].isna().astype(int)
    X["Skin_miss"]    = X["SkinThickness"].isna().astype(int)
    return X


with open("diabetes_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
threshold = bundle["threshold"]
columns = bundle["columns"]

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[c]) for c in columns]

        row = pd.DataFrame([values], columns=columns)
        row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)

        probability = model.predict_proba(row)[0, 1]
        prediction = int(probability >= threshold)

        if prediction == 1:
            result = f"High chance of Diabetes ({probability * 100:.2f}%)"
        else:
            result = f"Low chance of Diabetes ({probability * 100:.2f}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)