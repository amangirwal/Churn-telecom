import streamlit as st
import pandas as pd
import joblib

# load fitted pipeline model
model = joblib.load("model/churn_best_model.joblib")

st.title("Customer Churn Prediction App")

# user input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 100, 12)
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# create dataframe from inputs
user_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "InternetService": internet,
    "Contract": contract,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

# predict button
if st.button("Predict Churn"):
    prob = model.predict_proba(user_data)[0][1]  # churn probability
    st.subheader(f"Churn Probability: {prob*100:.2f}%")

    if prob > 0.5:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer will NOT churn")
