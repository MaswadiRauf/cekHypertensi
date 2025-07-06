from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model dan scaler
log_model = joblib.load("models/logistic.pkl")
dt_model = joblib.load("models/decision_tree.pkl")
rf_model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    voting = None
    if request.method == 'POST':
        # Ambil input user
        inputs = [
            float(request.form['male']),
            float(request.form['age']),
            float(request.form['currentSmoker']),
            float(request.form['cigsPerDay']),
            float(request.form['BPMeds']),
            float(request.form['diabetes']),
            float(request.form['totChol']),
            float(request.form['sysBP']),
            float(request.form['diaBP']),
            float(request.form['BMI']),
            float(request.form['heartRate']),
            float(request.form['glucose'])
        ]
        
        inputs_scaled = scaler.transform([inputs])

        # Prediksi dari 3 model
        pred_log = log_model.predict(inputs_scaled)[0]
        pred_dt = dt_model.predict(inputs_scaled)[0]
        pred_rf = rf_model.predict(inputs_scaled)[0]

        # Voting sederhana
        total_pos = sum([pred_log, pred_dt, pred_rf])
        if total_pos >= 2:
            voting = "POSITIF (Berisiko Hipertensi)"
        else:
            voting = "NEGATIF (Tidak Berisiko)"

        result = {
            'Logistic Regression': pred_log,
            'Decision Tree': pred_dt,
            'Random Forest': pred_rf
        }
        
    return render_template('index.html', result=result, voting=voting)

if __name__ == '__main__':
    app.run(debug=True)
