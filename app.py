import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

print(tf.__version__)

# Load model and scaler from kaggle
model = load_model('c_model.h5')
scaler = joblib.load('scaler_fix.save')

st.title("Customer Churn Prediction Model for Credit Card Users")

# Input fields
credit_score = st.number_input("Credit Score (300-900)", min_value=300, max_value=900, value=650)
age = st.number_input("Age (18-100)", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Number of Products (1-4)", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card (Yes=1, No=0)", [0, 1])
is_active_member = st.selectbox("Active Member (Yes=1, No=0)", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])

#one hot encoding
if geography == "Spain":
    g_spain = 1
    g_germany = 0
elif geography == "Germany":
    g_germany = 1
    g_spain = 0
else:  # This handles France
    g_spain = 0
    g_germany = 0

if gender == "Male":
    male_gender = 1
else:
    male_gender = 0

input_features = [
    credit_score,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    g_germany,
    g_spain,
    male_gender
]

# Convert to numpy array
input_array = np.array([input_features])

# Make prediction on button click
if st.button("Predict"):
    # Scale the input data
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Get the probability (this is between 0 and 1)
    raw_probability = prediction[0][0]
    
    # Convert to percentage for display
    churn_percentage = raw_probability * 100
    
    # Show the results
    st.write(f"Churn Probability: {churn_percentage:.2f}%")
    
    if raw_probability > 0.33:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
