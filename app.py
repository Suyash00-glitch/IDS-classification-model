import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('marketing_model.pkl')

st.title("Marketing Campaign Response Predictor")
st.write("Enter customer details to predict if they will respond.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'annual_income': [income],
        'credit_score': [credit_score]
    })
    prediction = model.predict(input_data)
    st.success(f"Predicted Response: {prediction[0]}")
