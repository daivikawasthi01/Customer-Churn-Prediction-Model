import streamlit as st
import numpy as np
import xgboost as xgb
import joblib

# Load model and scaler exported from kaggle notebook
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')
config = joblib.load('model_config.save')
scaler = joblib.load('scaler.save')

st.title("Customer Churn Prediction Model for Credit Card Users")

# Input fields
credit_score = st.number_input("Credit Score (300-900)", min_value=300, max_value=900, value=650)
age = st.number_input("Age (18-100)", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_prods = st.number_input("Number of prods (1-4)", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card (Yes=1, No=0)", [0, 1])
is_active_member = st.selectbox("Active Member (Yes=1, No=0)", [0, 1])
est_salary = st.number_input("est Salary", min_value=0.0, value=100000.0)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])

#Label encoding
if geography == "France":
    geo_enco = 0
elif geography == "Germany":
    geo_enco = 1
elif geography == "Spain":
    geo_enco = 2

if gender == "Male":
    gender_encoded = 1
elif gender == "Female":
    gender_encoded = 0

input_features = [
    credit_score,
    geo_enco,  
    gender_encoded,     
    age,
    tenure,
    balance,
    num_of_prods,
    has_cr_card,
    is_active_member,
    est_salary
]

# Convert to numpy array
input_array = np.array([input_features])

# Make prediction on button click
if st.button("Predict"):
    # Scale input data
    input_scaled = scaler.transform(input_array)
    
    # Make Predictions on scaled input
    prediction = model.predict_proba(input_scaled)
    
    # Get the probability(0-1)
    raw_probability = prediction[0][1]
    
    # Convert to percentage for display
    churn_percentage = raw_probability * 100

     # Optimal threshold print
    threshold = config['optimal_threshold']
    
    # Show results
    st.write(f"Churn Probability: {churn_percentage:.2f}%")
    
    if raw_probability > threshold:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")

