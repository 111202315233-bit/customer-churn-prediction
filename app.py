import streamlit as st
import pandas as pd
import joblib
import sklearn
from pathlib import Path

st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    layout="wide"
)

st.title("📊 Dashboard Prediksi Customer Churn")
st.write("Aplikasi Machine Learning untuk memprediksi pelanggan yang berpotensi churn.")

st.write("scikit-learn version:", sklearn.__version__)

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model_churn.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


st.markdown("---")

st.header("Masukkan Data Pelanggan")

col1, col2 = st.columns(2)

input_data = {}

with col1:

    input_data["customer_id"] = st.number_input(
        "Customer ID", value=1, step=1
    )

    input_data["gender"] = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    input_data["age"] = st.number_input(
        "Age", value=30.0
    )

    input_data["country"] = st.selectbox(
        "Country",
        ["Bangladesh", "Germany", "India", "UK", "USA"]
    )

    input_data["city"] = st.selectbox(
        "City",
        ["Berlin", "Delhi", "Dhaka",
         "Hamburg", "London", "Mumbai", "New York"]
    )

    input_data["signup_date"] = st.text_input(
        "Signup Date",
        "2023-01-01"
    )

    input_data["last_purchase_date"] = st.text_input(
        "Last Purchase Date",
        "2024-01-01"
    )

    input_data["acquisition_channel"] = st.selectbox(
        "Acquisition Channel",
        ["Email",
         "Facebook Ads",
         "Google Ads",
         "Organic",
         "Referral"]
    )

    input_data["device_type"] = st.selectbox(
        "Device Type",
        ["Desktop", "Mobile", "Tablet"]
    )

    input_data["subscription_type"] = st.selectbox(
        "Subscription Type",
        ["Annual", "Monthly"]
    )

    input_data["is_premium_user"] = st.selectbox(
        "Premium User",
        [0, 1]
    )

    input_data["total_visits"] = st.number_input(
        "Total Visits", value=10
    )

    input_data["avg_session_time"] = st.number_input(
        "Average Session Time", value=15.0
    )

    input_data["pages_per_session"] = st.number_input(
        "Pages per Session", value=5.0
    )

    input_data["email_open_rate"] = st.number_input(
        "Email Open Rate", value=0.5
    )

with col2:

    input_data["email_click_rate"] = st.number_input(
        "Email Click Rate", value=0.2
    )

    input_data["total_spent"] = st.number_input(
        "Total Spent", value=500.0
    )

    input_data["avg_order_value"] = st.number_input(
        "Average Order Value", value=100.0
    )

    input_data["discount_used"] = st.selectbox(
        "Discount Used",
        [0, 1]
    )

    input_data["coupon_code"] = st.selectbox(
        "Coupon Code",
        ["NEW20", "REF10", "SALE15"]
    )

    input_data["support_tickets"] = st.number_input(
        "Support Tickets", value=1
    )

    input_data["refund_requested"] = st.selectbox(
        "Refund Requested",
        [0, 1]
    )

    input_data["delivery_delay_days"] = st.number_input(
        "Delivery Delay Days", value=0
    )

    input_data["payment_method"] = st.selectbox(
        "Payment Method",
        ["BKash", "Card", "PayPal", "SEPA", "UPI"]
    )

    input_data["satisfaction_score"] = st.number_input(
        "Satisfaction Score", value=8.0
    )

    input_data["nps_score"] = st.number_input(
        "NPS Score", value=50
    )

    input_data["marketing_spend_per_user"] = st.number_input(
        "Marketing Spend", value=100.0
    )

    input_data["lifetime_value"] = st.number_input(
        "Lifetime Value", value=1000.0
    )

    input_data["last_3_month_purchase_freq"] = st.number_input(
        "Last 3 Month Purchase Frequency",
        value=5
    )

st.markdown("---")

if st.button("🔍 Prediksi Churn", use_container_width=True):

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ Pelanggan diprediksi CHURN\n\nProbabilitas: {probability:.2%}")
    else:
        st.success(f"✅ Pelanggan TIDAK CHURN\n\nProbabilitas Churn: {probability:.2%}")
