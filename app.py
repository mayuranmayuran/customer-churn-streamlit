import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Load trained model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler (if used)
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
phone = st.selectbox("Phone Service", ["No", "Yes"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year"])
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

# Encoding (same as training)
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
phone = 1 if phone == "Yes" else 0
internet = 1 if internet == "Fiber optic" else 0
contract = 1 if contract == "One year" else 0

input_data = np.array([[gender, senior, partner, dependents,
                        tenure, phone, internet, contract,
                        monthly, total]])

# Scaling
if scaler:
    input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
