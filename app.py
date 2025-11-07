import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("/Users/amangirwal/Desktop/projects/churn/churn_best_model.joblib")

st.title("Customer Churn Prediction")

# input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("SeniorCitizen", [0,1])
partner = st.selectbox("Partner", ["Yes","No"])
dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly = st.number_input("MonthlyCharges", min_value=0.0, value=50.0)
total = st.number_input("TotalCharges", min_value=0.0, value=500.0)

# create input row
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

# fill missing columns if necessary
missing_cols = set(model.named_steps['pre'].feature_names_in_) - set(data.columns)
for c in missing_cols:
    data[c] = "No"

# predict
if st.button("Predict"):
    prob = model.predict_proba(data)[0][1]
    pred = "Will Churn" if prob>0.5 else "Will NOT Churn"
    st.subheader(f"Prediction: {pred}")
    st.text(f"Probability: {prob*100:.2f}%")
