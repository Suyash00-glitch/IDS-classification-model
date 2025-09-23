import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('marketing_model.pkl')

st.title("📊 Marketing Campaign Response Predictor")

st.write("Enter the customer details below to predict whether they will respond (Yes/No):")

# --- Input fields ---
age = st.number_input("Age", min_value=18, max_value=100, value=35)
annual_income = st.number_input("Annual Income (₹)", min_value=10000, max_value=200000, value=65000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720)

# Prepare the input for prediction
input_data = np.array([[age, annual_income, credit_score]])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = prediction[0]

    if result == "Yes":
        st.success("✅ This customer is likely to respond to the campaign.")
    else:
        st.error("❌ This customer is unlikely to respond to the campaign.")
