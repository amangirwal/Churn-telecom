import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction")

# load tuple
final_model, preprocess = joblib.load("model/churn_best_model.joblib")

# INPUT UI
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("SeniorCitizen", [0,1])
partner = st.selectbox("Partner", ["Yes","No"])
dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.number_input("Tenure (months)",0,100,12)
internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
monthly = st.number_input("MonthlyCharges",0.0,500.0,50.0)
total = st.number_input("TotalCharges",0.0,10000.0,500.0)

data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "InternetService":[internet],
    "Contract":[contract],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
})

# PREDICT
if st.button("Predict"):
    X = preprocess.transform(data)
    prob = final_model.predict_proba(X)[0][1]
    st.subheader(f"Churn Probability: {prob*100:.2f}%")
    st.write("Prediction:", "Will Churn" if prob>0.5 else "Will NOT Churn")
