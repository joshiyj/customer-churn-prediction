import streamlit as st
import pickle
import pandas as pd

# ---------------------------
# Load model and encoders
# ---------------------------
model_data = pickle.load(open('customer_churn_model.pkl', 'rb'))
model = model_data['model']
feature_names = model_data['features_names']

encoders = pickle.load(open('encoders.pkl', 'rb'))

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# ---------------------------
# Custom CSS for styling
# ---------------------------
st.markdown("""
    <style>
        /* Background */
        .main {
            background: linear-gradient(135deg, #e0f7fa, #e3f2fd);
            color: #0f172a;
            font-family: "Poppins", sans-serif;
        }

        /* Header */
        h1 {
            background: linear-gradient(90deg, #2563eb, #3b82f6);
            color: white;
            text-align: center;
            padding: 0.8em;
            border-radius: 12px;
            font-size: 2rem !important;
            margin-bottom: 0.5em;
        }

        h3 {
            text-align: center;
            color: #334155;
            font-weight: 400;
            margin-bottom: 2em;
        }

        /* Buttons */
        .stButton>button {
            background-color: #2563EB;
            color: white;
            border: none;
            padding: 0.7rem 2rem;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            transition: 0.2s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #1E40AF;
            transform: scale(1.02);
        }

        /* Success / Error boxes */
        .success-box {
            background-color: #DCFCE7;
            color: #14532D;
            border-left: 6px solid #16A34A;
            padding: 1rem;
            border-radius: 10px;
            font-weight: 500;
        }

        .error-box {
            background-color: #FEE2E2;
            color: #7F1D1D;
            border-left: 6px solid #DC2626;
            padding: 1rem;
            border-radius: 10px;
            font-weight: 500;
        }

        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.title("📊 Customer Churn Prediction")
st.markdown("### Enter customer details below to check whether they are **likely to churn** or **stay**.")

st.divider()

# ---------------------------
# Input Layout
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

st.divider()

# ---------------------------
# Prepare input
# ---------------------------
input_data = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# ---------------------------
# Prediction Section
# ---------------------------
if st.button("🔍 Predict Churn", use_container_width=True):
    try:
        df = pd.DataFrame([input_data])

        # Apply label encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Predict
        prediction = model.predict(df)[0]
        pred_prob = model.predict_proba(df)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div class="error-box">
                ⚠️ <b>The customer is likely to <u>Churn</u></b><br>
                <b>Churn Probability:</b> {pred_prob:.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                ✅ <b>The customer will Stay</b><br>
                <b>Churn Probability:</b> {pred_prob:.2f}
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
