import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("rain_model.pkl", "rb"))

# Page setup
st.set_page_config(page_title="Rainfall Prediction", page_icon="🌧️")
st.title("🌧️ Rainfall Prediction App")
st.write("Enter today's weather details to predict whether it will rain tomorrow.")

# Input fields
min_temp = st.number_input("Min Temperature (°C)")
max_temp = st.number_input("Max Temperature (°C)")
rainfall = st.number_input("Rainfall (mm)")
humidity9am = st.number_input("Humidity at 9AM (%)")
humidity3pm = st.number_input("Humidity at 3PM (%)")
pressure9am = st.number_input("Pressure at 9AM (hPa)")
pressure3pm = st.number_input("Pressure at 3PM (hPa)")
temp9am = st.number_input("Temperature at 9AM (°C)")
temp3pm = st.number_input("Temperature at 3PM (°C)")
rain_today = st.selectbox("Did it rain today?", ["No", "Yes"])

# Prepare input
if st.button("Predict Rainfall"):
    input_data = pd.DataFrame([[
        min_temp, max_temp, rainfall, 
        humidity9am, humidity3pm,
        pressure9am, pressure3pm,
        temp9am, temp3pm,
        1 if rain_today == "Yes" else 0
    ]], columns=[
        'MinTemp', 'MaxTemp', 'Rainfall',
        'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm',
        'Temp9am', 'Temp3pm', 'RainToday'
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Yes, it will rain tomorrow.")
    else:
        st.info("❌ No, it will not rain tomorrow.")
