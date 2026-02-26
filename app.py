import streamlit as st
import numpy as np
import joblib

# Load models
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")

diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("diabetes_scaler.pkl")

st.title("🩺 Disease Prediction System")

option = st.sidebar.selectbox(
    "Select Prediction Model",
    ("Heart Disease", "Diabetes")
)

# ================= HEART =================
if option == "Heart Disease":
    st.header(" Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
    restingBP = st.number_input("Resting BP")
    cholesterol = st.number_input("Cholesterol")
    fastingBS = st.selectbox("Fasting Blood Sugar (0/1)", [0,1])
    maxHR = st.number_input("Max Heart Rate")
    exerciseAngina = st.selectbox("Exercise Angina (0/1)", [0,1])
    oldpeak = st.number_input("Oldpeak")

    if st.button("Predict Heart Disease"):

        input_data = (age, sex, restingBP, cholesterol, fastingBS,
                      maxHR, exerciseAngina, oldpeak,
                      0,1,0,
                      1,0,
                      0,1)

        input_array = np.asarray(input_data).reshape(1,-1)
        scaled = heart_scaler.transform(input_array)
        prediction = heart_model.predict(scaled)

        if prediction[0] == 0:
            st.success("Low Risk of Heart Disease")
        else:
            st.error("High Risk of Heart Disease")

# ================= DIABETES =================
if option == "Diabetes":
    st.header(" Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):

        input_data = (pregnancies, glucose, bp, skin,
                      insulin, bmi, dpf, age)

        input_array = np.asarray(input_data).reshape(1,-1)
        scaled = diabetes_scaler.transform(input_array)
        prediction = diabetes_model.predict(scaled)

        if prediction[0] == 0:
            st.success("Not Diabetic")
        else:
            st.error("Diabetic")