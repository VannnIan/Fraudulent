import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier

# === Load All Pretrained Components ===
model = XGBClassifier()
model.load_model("fraud_detection_model_pseudolabel.json")

scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

with open("selected_features.json") as f:
    selected_features = json.load(f)

# === Threshold from F1-Score Tuning ===
optimal_threshold = 0.81

# === Streamlit UI ===
st.title("💳 Fraud Detection – LOL Bank")
st.write("Masukkan data transaksi untuk memprediksi apakah transaksi tersebut berpotensi fraud.")

# === Input Form ===
with st.form("fraud_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    account_balance = st.number_input("Account Balance", min_value=0.0, value=1000.0, step=100.0)
    account_type = st.selectbox("Account Type", ["Savings", "Checking", "Credit", "Business"])
    transaction_type = st.selectbox("Transaction Type", ["Withdrawal", "Deposit", "Transfer", "Bill Payment", "Debit"])
    merchant_category = st.selectbox("Merchant Category", ["Retail", "Food", "Electronics", "Travel"])
    device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "ATM"])
    transaction_amount = st.number_input("Transaction Amount", min_value=1.0, value=100.0, step=1.0)

    submitted = st.form_submit_button("Prediksi")

# === Process Input ===
if submitted:
    # Raw input → DataFrame
    input_data = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Account_Balance": account_balance,
        "Account_Type": account_type,
        "Transaction_Type": transaction_type,
        "Merchant_Category": merchant_category,
        "Device_Type": device_type,
        "Transaction_Amount": transaction_amount
    }])

    # Categorical columns to encode
    cat_cols = ["Account_Type", "Transaction_Type", "Merchant_Category", "Device_Type"]
    encoded_cat = encoder.transform(input_data[cat_cols])
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

    # Combine numerical + encoded categorical
    final_df = pd.concat([
        input_data[["Gender", "Age", "Account_Balance", "Transaction_Amount"]].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ], axis=1)

    # Reorder columns based on training feature list
    final_df = final_df.reindex(columns=selected_features, fill_value=0)

    # Scaling
    final_scaled = scaler.transform(final_df)

    # Prediction
    proba = model.predict_proba(final_scaled)[0][1]

    # Output
    st.write("### 🔍 Hasil Prediksi:")

    if proba >= optimal_threshold:
        st.error("❌ Transaksi ini terindikasi **FRAUD**.")
    else:
        st.success("✅ Transaksi ini **NORMAL**.")

