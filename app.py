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
# Streamlit App Configuration
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Custom CSS for better look
st.markdown("""
    <style>
        /* Main page styling */
        .main {
            background-color: #F8FAFC;
            padding: 2rem;
        }
        h1 {
            color: #1F2937;
            text-align: center;
            font-size: 2.2rem !important;
            margin-bottom: 0.5rem;
        }
        h3 {
            color: #4B5563;
            text-align: center;
            font-weight: normal;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #2563EB;
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 10px;
            font-size: 16px;
            transition: 0.2s;
        }
        .stButton>button:hover {
            background-color: #1E40AF;
        }
        .success-box {
            background-color: #DCFCE7;
            color: #166534;
            border-left: 6px solid #16A34A;
            padding: 1rem;
            border-radius: 10px;
        }
        .error-box {
            background-color: #FEE2E2;
            color: #991B1B;
            border-left: 6px solid #DC2626;
            padding: 1rem;
            border-radius: 10px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown("### Predict whether a telecom customer is likely to **Churn** or **Stay** using a trained Random Forest model.")
st.divider()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/9442/9442834.png", width=120)
st.sidebar.markdown("### 💡 About")
st.sidebar.info("""
This tool helps telecom businesses identify customers at risk of churn.  
Enter customer details on the right to see predictions in real-time.
""")
st.sidebar.markdown("**Model:** Random Forest Classifier")  
st.sidebar.markdown("**Created by:** Tejas Hadge")  

# ---------------------------
# Input Layout (2 Columns)
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

st.divider()

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
                <h4>⚠️ The customer is <b>likely to Churn</b></h4>
                <p><b>Churn Probability:</b> {pred_prob:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ The customer will <b>Stay</b></h4>
                <p><b>Churn Probability:</b> {pred_prob:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
