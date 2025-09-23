import streamlit as st
import pandas as pd
import joblib

# Load the trained model
logreg = joblib.load('marketing_model.pkl')

st.title("Marketing Campaign Response Predictor")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=1000, max_value=1000000, value=60000)
credit = st.number_input("Credit Score", min_value=300, max_value=850, value=710)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
gender = st.selectbox("Gender (Male=1, Female=0)", [0,1])
employed = st.selectbox("Employed (Yes=1, No=0)", [0,1])
single = st.selectbox("Single? (Yes=1, No=0)", [0,1])
customer_id = st.number_input("Customer ID", min_value=1, max_value=1000, value=57)

# Predict button
if st.button("Predict"):
    new_data = pd.DataFrame({
        'customer_id':[customer_id],
        'age':[age],
        'annual_income':[income],
        'credit_score':[credit],
        'no_of_children':[children],
        'gender_Male':[gender],
        'employed_Yes':[employed],
        'marital_status_Single':[single]
    })
    
    prediction = logreg.predict(new_data)
    st.success(f"Predicted Response: {prediction[0]}")
