from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    age = float(request.form['age'])
    hypertension = 1 if 'hypertension' in request.form else 0
    heart_disease = 1 if 'heart_disease' in request.form else 0
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    gender = request.form['gender']
    smoking_history = request.form['smoking_history']

    # One-hot encode categorical features
    gender_encoded = [1 if gender == 'Female' else 0, 1 if gender == 'Male' else 0, 1 if gender == 'Other' else 0]
    smoking_history_encoded = [1 if smoking_history == 'No Info' else 0,
                                1 if smoking_history == 'current' else 0,
                                1 if smoking_history == 'ever' else 0,
                                1 if smoking_history == 'former' else 0,
                                1 if smoking_history == 'never' else 0,
                                1 if smoking_history == 'not current' else 0]

    # Prepare the input data
    input_data = np.array([[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level] + gender_encoded + smoking_history_encoded])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 0:
        result_text = "Likely Absent"
    else:
        result_text = "Likely Present"

    return render_template('index.html', result=result_text, probability=round(probability, 2))

if __name__ == '__main__':
    app.run(debug=True)