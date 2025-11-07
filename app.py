import streamlit as st
import pandas as pd
import kagglehub

# Download dataset
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")

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

st.title("Customer Churn Prediction")

# UI
data = {}
data["gender"] = st.selectbox("Gender",["Male","Female"])
data["SeniorCitizen"] = st.selectbox("SeniorCitizen",[0,1])
data["Partner"] = st.selectbox("Partner",["Yes","No"])
data["Dependents"] = st.selectbox("Dependents",["Yes","No"])
data["tenure"] = st.number_input("Tenure (months)",0,100,12)
data["InternetService"] = st.selectbox("InternetService",["DSL","Fiber optic","No"])
data["Contract"] = st.selectbox("Contract",["Month-to-month","One year","Two year"])
data["MonthlyCharges"] = st.number_input("MonthlyCharges",0.0,500.0,50.0)
data["TotalCharges"] = st.number_input("TotalCharges",0.0,10000.0,500.0)

df_input = pd.DataFrame([data])

if st.button("Predict"):
    prob = model.predict_proba(df_input)[0][1]
    st.write("Churn Probability:", round(prob*100,2),"%")
    st.write("Result:", "Will Churn" if prob>0.5 else "Will NOT Churn")
