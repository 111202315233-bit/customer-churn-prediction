import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model_churn.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model : {e}")
    st.stop()

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "Dashboard Machine Learning untuk memprediksi kemungkinan pelanggan melakukan **Customer Churn**."
)

st.sidebar.header("Tentang Dashboard")
st.sidebar.info(
"""
Dashboard ini dibuat menggunakan:

- Random Forest
- Scikit-Learn
- Streamlit

Dataset:
Sales & Marketing Customer Dataset
"""
)

st.markdown("---")

input_dict = {}
st.header("👤 Demografi Pelanggan")

col1, col2 = st.columns(2)

with col1:

    input_dict["customer_id"] = st.number_input(
        "Customer ID",
        min_value=1,
        value=1
    )

    input_dict["gender"] = st.selectbox(
        "Gender",
        ["Female","Male","Other"]
    )

    input_dict["age"] = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=30.0
    )

with col2:

    input_dict["country"] = st.selectbox(
        "Country",
        [
            "Bangladesh",
            "Germany",
            "India",
            "UK",
            "USA"
        ]
    )

    input_dict["city"] = st.selectbox(
        "City",
        [
            "Berlin",
            "Delhi",
            "Dhaka",
            "Hamburg",
            "London",
            "Mumbai",
            "New York"
        ]
    )

st.markdown("---")

st.header("📱 Informasi Akun")

col1, col2 = st.columns(2)

with col1:

    input_dict["signup_date"] = st.text_input(
        "Signup Date",
        "2023-01-01"
    )

    input_dict["last_purchase_date"] = st.text_input(
        "Last Purchase Date",
        "2024-01-01"
    )

    input_dict["device_type"] = st.selectbox(
        "Device Type",
        ["Desktop","Mobile","Tablet"]
    )

with col2:

    input_dict["subscription_type"] = st.selectbox(
        "Subscription Type",
        ["Annual","Monthly"]
    )

    input_dict["is_premium_user"] = st.selectbox(
        "Premium User",
        [0,1]
    )

    input_dict["acquisition_channel"] = st.selectbox(
        "Acquisition Channel",
        [
            "Email",
            "Facebook Ads",
            "Google Ads",
            "Organic",
            "Referral"
        ]
    )
    st.markdown("---")

st.header("💳 Informasi Transaksi")

col1, col2 = st.columns(2)

with col1:

    input_dict["total_spent"] = st.number_input(
        "Total Spent",
        min_value=0.0,
        value=500.0
    )

    input_dict["avg_order_value"] = st.number_input(
        "Average Order Value",
        min_value=0.0,
        value=100.0
    )

    input_dict["payment_method"] = st.selectbox(
        "Payment Method",
        [
            "BKash",
            "Card",
            "PayPal",
            "SEPA",
            "UPI"
        ]
    )

with col2:

    input_dict["coupon_code"] = st.selectbox(
        "Coupon Code",
        [
            "NEW20",
            "REF10",
            "SALE15"
        ]
    )

    input_dict["discount_used"] = st.selectbox(
        "Discount Used",
        [0,1]
    )

    input_dict["delivery_delay_days"] = st.number_input(
        "Delivery Delay Days",
        min_value=0,
        value=0
    )

st.markdown("---")

st.header("😊 Kepuasan Pelanggan")

col1, col2 = st.columns(2)

with col1:

    input_dict["support_tickets"] = st.number_input(
        "Support Tickets",
        min_value=0,
        value=0
    )

    input_dict["refund_requested"] = st.selectbox(
        "Refund Requested",
        [0,1]
    )

with col2:

    input_dict["satisfaction_score"] = st.slider(
        "Satisfaction Score",
        1.0,
        10.0,
        8.0
    )

    input_dict["nps_score"] = st.slider(
        "NPS Score",
        0,
        100,
        50
    )

st.markdown("---")

st.header("📈 Aktivitas Pelanggan")

col1, col2 = st.columns(2)

with col1:

    input_dict["total_visits"] = st.number_input(
        "Total Visits",
        min_value=0,
        value=20
    )

    input_dict["avg_session_time"] = st.number_input(
        "Average Session Time",
        min_value=0.0,
        value=20.0
    )

    input_dict["pages_per_session"] = st.number_input(
        "Pages Per Session",
        min_value=0.0,
        value=5.0
    )

    input_dict["email_open_rate"] = st.slider(
        "Email Open Rate",
        0.0,
        1.0,
        0.5
    )

with col2:

    input_dict["email_click_rate"] = st.slider(
        "Email Click Rate",
        0.0,
        1.0,
        0.2
    )

    input_dict["marketing_spend_per_user"] = st.number_input(
        "Marketing Spend Per User",
        min_value=0.0,
        value=100.0
    )

    input_dict["lifetime_value"] = st.number_input(
        "Lifetime Value",
        min_value=0.0,
        value=1000.0
    )

    input_dict["last_3_month_purchase_freq"] = st.number_input(
        "Purchase Frequency (Last 3 Months)",
        min_value=0,
        value=5
    )
    st.markdown("---")

if st.button("🔍 Prediksi Customer Churn", use_container_width=True):

    try:

        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("## 📊 Hasil Prediksi")

        if prediction == 1:

            st.error("### ❌ Customer Diprediksi Akan Churn")

        else:

            st.success("### ✅ Customer Diprediksi Loyal")

        st.write(f"**Probabilitas Churn : {probability:.2%}**")

        st.progress(float(probability))

        st.markdown("---")

        st.subheader("📋 Ringkasan Data Pelanggan")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Gender** :", input_dict["gender"])
            st.write("**Age** :", input_dict["age"])
            st.write("**Country** :", input_dict["country"])
            st.write("**City** :", input_dict["city"])
            st.write("**Subscription** :", input_dict["subscription_type"])
            st.write("**Device** :", input_dict["device_type"])

        with col2:
            st.write("**Total Spent** :", input_dict["total_spent"])
            st.write("**Payment** :", input_dict["payment_method"])
            st.write("**Satisfaction** :", input_dict["satisfaction_score"])
            st.write("**NPS** :", input_dict["nps_score"])
            st.write("**Total Visits** :", input_dict["total_visits"])
            st.write("**Lifetime Value** :", input_dict["lifetime_value"])

        st.markdown("---")

        if probability >= 0.80:

            st.warning(
                """
### ⚠️ Rekomendasi

Pelanggan memiliki risiko churn yang sangat tinggi.

Disarankan:

- Berikan promo khusus
- Hubungi pelanggan
- Tingkatkan pelayanan
- Berikan loyalty reward
                """
            )

        elif probability >= 0.50:

            st.info(
                """
### 📌 Rekomendasi

Risiko churn sedang.

Perlu dilakukan monitoring pelanggan secara berkala.
                """
            )

        else:

            st.success(
                """
### 🎉 Rekomendasi

Pelanggan masih loyal.

Pertahankan kualitas pelayanan.
                """
            )

    except Exception as e:

        st.error(f"Terjadi error : {e}")
