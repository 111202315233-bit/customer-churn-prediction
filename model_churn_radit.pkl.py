import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Dashboard Churn Analytics - Radit", layout="wide")
st.title("Sales and Marketing - Customer Churn Dashboard")
st.write("Aplikasi berbasis Machine Learning untuk mendeteksi potensi churn pada pelanggan.")

@st.cache_resource
def load_model():
    return joblib.load('model_churn_radit.pkl')

try:
    model = load_model()
    if hasattr(model, 'feature_names_in_'):
        fitur_wajib = list(model.feature_names_in_)
    else:
        fitur_wajib = ['Age', 'Gender', 'Tenure', 'UsageFrequency', 'SupportCalls', 'PaymentDelay', 
                       'SubscriptionType', 'ContractLength', 'TotalCharges', 'LastInteraction']
except Exception as e:
    st.error(f"Gagal memuat file model_churn_radit.pkl. Error: {e}")
    fitur_wajib = []

st.markdown("---")

if fitur_wajib:
    st.subheader("Masukkan Data Pelanggan")
    
    input_dict = {}
    col1, col2 = st.columns(2)
    
    for i, col_name in enumerate(fitur_wajib):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if col_name in ['Gender', 'SubscriptionType', 'ContractLength']:
                if col_name == 'Gender': options = ['Male', 'Female']
                elif col_name == 'SubscriptionType': options = ['Basic', 'Standard', 'Premium']
                else: options = ['Monthly', 'Quarterly', 'Annual']
                input_dict[col_name] = st.selectbox(f"Pilih {col_name}", options)
            else:
                if col_name == 'Age': val, min_v, max_v = 30, 18, 100
                elif col_name == 'Tenure': val, min_v, max_v = 12, 0, 120
                elif col_name == 'TotalCharges': val, min_v, max_v = 500.0, 0.0, 10000.0
                else: val, min_v, max_v = 3, 0, 500
                input_dict[col_name] = st.number_input(f"Input {col_name}", min_value=min_v, value=val)

    st.markdown("---")
    if st.button("Hitung Probabilitas Churn", use_container_width=True):
        try:
            input_data = pd.DataFrame([input_dict])[fitur_wajib]
            
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.markdown("### Hasil Deteksi Sistem:")
            if prediction[0] == 1:
                st.error(f"STATUS FINAL: PELANGGAN AKAN CHURN (Risiko: {probability*100:.2f}%)")
            else:
                st.success(f"STATUS FINAL: PELANGGAN AMAN / LOYAL (Risiko Churn: {probability*100:.2f}%)")
        except Exception as e:
            st.error(f"Gagal memproses data. Detail error: {e}")