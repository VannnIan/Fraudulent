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
optimal_threshold = 0.82  # Ganti sesuai hasil tuning

# === Streamlit UI ===
st.title("💳 Fraud Detection – LOL Bank")
st.write("Masukkan data transaksi untuk memprediksi apakah transaksi tersebut berpotensi fraud.")

# === Input Form ===
with st.form("fraud_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    account_type = st.selectbox("Account Type", ["Savings", "Checking", "Credit"])
    transaction_type = st.selectbox("Transaction Type", ["Withdrawal", "Deposit", "Transfer"])
    merchant_category = st.selectbox("Merchant Category", ["Retail", "Food", "Electronics", "Travel"])
    device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
    transaction_amount = st.number_input("Transaction Amount", min_value=1.0, value=100.0, step=1.0)

    submitted = st.form_submit_button("Prediksi")

# === Process Input ===
if submitted:
    # Raw input → DataFrame
    input_data = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Account_Type": account_type,
        "Transaction_Type": transaction_type,
        "Merchant_Category": merchant_category,
        "Device_Type": device_type,
        "Transaction_Amount": transaction_amount
    }])

    # Pisahkan kategorikal
    cat_cols = ["Account_Type", "Transaction_Type", "Merchant_Category", "Device_Type"]
    encoded_cat = encoder.transform(input_data[cat_cols])
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

    # Gabungkan numerik dan encoded
    final_df = pd.concat(
        [input_data[["Gender", "Transaction_Amount"]].reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Reorder kolom sesuai selected_features
    final_df = final_df.reindex(columns=selected_features, fill_value=0)

    # Scaling
    final_scaled = scaler.transform(final_df)

    # Prediksi
    proba = model.predict_proba(final_scaled)[0][1]
    prediction = int(proba >= optimal_threshold)

    # Output
    st.write("### 🔍 Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Transaksi ini terindikasi **FRAUD**.")
    else:
        st.success("✅ Transaksi ini **normal**.")
