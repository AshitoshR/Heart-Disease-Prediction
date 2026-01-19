from flask import Flask, render_template, request
import numpy as np
import joblib as b
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = b.load(MODEL_DIR / "logistic_model.pkl")
scaler = b.load(MODEL_DIR / "scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        features = [
            float(request.form[x]) for x in [
                "age","sex","cp","bp","chol","fbs","ekg","max_hr",
                "ex_angina","st_dep","slope","vessels","thallium"              
            ]
        ]

        data = scaler.transform([features])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
    
    return render_template("index.html", 
                           prediction=prediction,
                           probability=probability)

if __name__ == "__main__":
    app.run(debug=True)