import streamlit as st
import joblib
import pandas as pd

# Load the retrained model
model = joblib.load('marketing_model.pkl')

# Streamlit app title
st.title("Marketing Campaign Response Prediction")

st.write("""
Predict whether a customer will respond positively to a marketing campaign.
Please enter the customer's details below.
""")

# User inputs: only the 3 features your new model uses
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=1000, max_value=200000, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'annual_income': [annual_income],
    'credit_score': [credit_score]
})

# Predict button
if st.button("Predict Response"):
    prediction = model.predict(input_data)
    st.write("Predicted Response:", prediction[0])
