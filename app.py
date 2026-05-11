import streamlit as st
import pandas as pd
import joblib

model = joblib.load("sales_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("📈 Walmart Sales Prediction")

store = st.number_input("Store", min_value=1)
temperature = st.number_input("Temperature")
fuel_price = st.number_input("Fuel Price")
cpi = st.number_input("CPI")
unemployment = st.number_input("Unemployment")
holiday_flag = st.selectbox("Holiday Flag", [0, 1])

year = st.number_input("Year", min_value=2010)
month = st.number_input("Month", min_value=1, max_value=12)
day = st.number_input("Day", min_value=1, max_value=31)
week = st.number_input("Week", min_value=1, max_value=53)

if st.button("Predict"):

    sample = pd.DataFrame({
        'Store': [store],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Holiday_Flag': [holiday_flag],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Week': [week]
    })

    # Exact training order
    sample = sample[feature_names]

    prediction = model.predict(sample)

    st.success(f"Predicted Sales = ₹ {prediction[0]:,.2f}")