from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and expected columns
model = joblib.load("gradient_boosting_model.pkl")
expected_columns = joblib.load("model_columns.pkl")

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        form_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['senior']),
            'Partner': request.form['partner'],
            'Dependents': request.form['dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['phone'],
            'MultipleLines': request.form['multiple'],
            'InternetService': request.form['internet'],
            'OnlineSecurity': request.form['security'],
            'OnlineBackup': request.form['backup'],
            'DeviceProtection': request.form['protection'],
            'TechSupport': request.form['support'],
            'StreamingTV': request.form['tv'],
            'StreamingMovies': request.form['movies'],
            'Contract': request.form['contract'],
            'PaperlessBilling': request.form['paperless'],
            'PaymentMethod': request.form['payment'],
            'MonthlyCharges': float(request.form['charges']),
            'TotalCharges': float(request.form['total']),
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])

        # One-hot encode and align with training columns
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][int(prediction)]

        label = "Customer will churn" if prediction == 1 else "Customer will stay"

        return render_template('index.html', result=label, confidence=f"{probability * 100:.2f}%")

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
