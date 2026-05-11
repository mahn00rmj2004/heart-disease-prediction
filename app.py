import os
import sys
import numpy as np
import joblib
from flask import Flask, request, render_template

# 1. Look in the current folder for imports (Removed '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, template_folder='templates')

# 2. Setup correct directory paths (Removed '..')
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 3. Load model artifacts from the local models folder
try:
    model         = joblib.load(os.path.join(MODELS_DIR, 'trained_model.pkl'))
    scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    print("[INFO] All model artifacts loaded successfully")
except FileNotFoundError as e:
    print(f"[ERROR] Model file not found: {e}")
    model = scaler = feature_names = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="ERROR", message="Model not loaded.")

    try:
        # Collect data from form
        data = [float(request.form[name]) for name in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        
        features = np.array([data])
        features_scaled = scaler.transform(features)

        prediction_label = model.predict(features_scaled)[0]
        probability      = model.predict_proba(features_scaled)[0]

        if prediction_label == 1:
            result, message = "HIGH RISK", "High Risk detected. Consult a doctor."
        else:
            result, message = "LOW RISK", "Low Risk. Keep it up!"

        return render_template('index.html',
                               prediction=result,
                               message=message,
                               risk_pct=round(probability[1] * 100, 1),
                               form_data=request.form)
    except Exception as e:
        return render_template('index.html', prediction="ERROR", message=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)