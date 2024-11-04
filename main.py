from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

models_dir = "Insurance_Cost_Predictor/insurance_model.pkl"
data_file = "Insurance_Cost_Predictor/insurance.csv"

# Load dataset
df = pd.read_csv(data_file)
print(df.columns)

try:
    with open(models_dir, "rb") as model_file:
        insurance_model = pickle.load(model_file)
except FileNotFoundError:
    insurance_model = None

# Function to save new data entries in lowercase
def save_data(inputs, predicted_cost):
    # Convert 'sex', 'smoker', 'region' values to lowercase before saving
    new_data = pd.DataFrame([inputs + [predicted_cost]], 
                            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'])
    new_data['sex'] = new_data['sex'].str.lower()
    new_data['smoker'] = new_data['smoker'].str.lower()
    new_data['region'] = new_data['region'].str.lower()

    # Append to the CSV file
    with open(data_file, mode='a', newline='') as file:
        new_data.to_csv(file, header=False, index=False)

# Function to retrain the model with updated data
def retrain_model():
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']

    # Preprocessing (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train RandomForestRegressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    with open("insurance_model.pkl", "wb") as model_file:
        pickle.dump(rf_model, model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_insurance():
    if request.method == 'POST':
        try:
            # Collect input data from the form
            age = float(request.form.get('age'))

            # Convert sex to numerical value for prediction
            sex = request.form.get('sex').lower()
            sex_value = 0 if sex == 'male' else 1  # 0 for male, 1 for female

            bmi = float(request.form.get('bmi'))
            children = int(request.form.get('children'))

            # Convert smoker status to numerical value for prediction
            smoker = request.form.get('smoker').lower()
            smoker_value = 0 if smoker == 'yes' else 1  # 0 for yes, 1 for no

            # Convert region to numerical value for prediction
            region = request.form.get('region').lower()
            region_dict = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
            region_value = region_dict.get(region, -1)  # Convert region to numerical value

            if region_value == -1:
                return "Error: Invalid region"

            # Prepare the input list with numerical values
            inputs = [age, sex_value, bmi, children, smoker_value, region_value]

            # Predict the insurance cost using the model
            insurance_prediction = insurance_model.predict([inputs])[0] if insurance_model else None

            # Save the data (original string values) and prediction to the CSV file
            save_data([age, sex, bmi, children, smoker, region], insurance_prediction)
            retrain_model()

            # Return the prediction as a string
            return str(round(insurance_prediction, 2)) if insurance_prediction is not None else "Error: Model not loaded."
        except Exception as e:
            return f"An error occurred: {e}"
    return render_template('predict.html')



if __name__ == "__main__":
    app.run(debug=True)
