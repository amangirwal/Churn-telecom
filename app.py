import streamlit as st
import pandas as pd

# Load dataset directly from GitHub
url = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# clean
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
df.drop(columns=["customerID"], inplace=True)

# split
from sklearn.model_selection import train_test_split
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)

# model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

num_cols = ["tenure","MonthlyCharges","TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

model = Pipeline([
    ("pre", preprocess),
    ("model", RandomForestClassifier(n_estimators=300,random_state=42))
])

model.fit(X_train, y_train)

st.title("Customer Churn Prediction App")

gender = st.selectbox("Gender",["Male","Female"])
senior = st.selectbox("SeniorCitizen",[0,1])
partner = st.selectbox("Partner",["Yes","No"])
dependents = st.selectbox("Dependents",["Yes","No"])
tenure = st.number_input("Tenure",0,100,12)
internet = st.selectbox("InternetService",["DSL","Fiber optic","No"])
contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
monthly = st.number_input("MonthlyCharges",0.0,500.0,50.0)
total = st.number_input("TotalCharges",0.0,10000.0,500.0)

input_data = pd.DataFrame([{
    "gender":gender,
    "SeniorCitizen":senior,
    "Partner":partner,
    "Dependents":dependents,
    "tenure":tenure,
    "InternetService":internet,
    "Contract":contract,
    "MonthlyCharges":monthly,
    "TotalCharges":total
}])

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    st.write("Churn Probability:", f"{prob*100:.2f}%")
    st.write("Prediction:", "Will Churn" if prob>0.5 else "Will NOT Churn")
