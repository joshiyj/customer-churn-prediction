import streamlit as st
import pickle
import pandas as pd

# ---------------------------
# Load model and encoders
# ---------------------------
# This backend logic remains unchanged.
model_data = pickle.load(open('customer_churn_model.pkl', 'rb'))
model = model_data['model']
feature_names = model_data['features_names']

encoders = pickle.load(open('encoders.pkl', 'rb'))

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="centered" # Centered layout looks better for this kind of form
)

# ---------------------------
# Custom CSS for Glassmorphism UI
# ---------------------------
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

        /* Base Body Style */
        body {
            font-family: 'Poppins', sans-serif;
        }

        /* Background */
        .main {
    background: #111827; /* A nice dark gray */
    color: #ffffff;
}

        /* Main title */
        h1 {
            font-family: 'Poppins', sans-serif;
            text-align: center;
            font-weight: 700;
            color: #ffffff;
            padding-bottom: 20px;
        }

        /* Subheader */
        h3 {
            text-align: center;
            color: #c7d2fe; /* A lighter, softer color */
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        /* Main container for inputs - The 'Glass' */
        .block-container {
            padding: 2rem;
        }
        
        /* Custom container for the form inputs */
        .form-container {
            background: rgba(255, 255, 255, 0.1); /* Semi-transparent white */
            backdrop-filter: blur(10px); /* The frosted glass effect */
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 2rem 3rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }

        /* Styling for Streamlit Widgets */
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput div[data-baseweb="input"] > div {
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 8px;
        }
        
        .stSelectbox label, .stNumberInput label {
            color: #e0e7ff !important; /* Lighter label color */
        }

        /* Predict Button */
        .stButton>button {
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            transition: 0.3s ease-in-out;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #4338ca, #6d28d9);
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }

        /* Result Boxes */
        .result-box {
            backdrop-filter: blur(10px);
            border: 1px solid;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 500;
            margin-top: 2rem;
        }

        .success-box {
            background-color: rgba(34, 197, 94, 0.2);
            border-color: rgba(34, 197, 94, 0.5);
            color: #f0fdf4;
        }

        .error-box {
            background-color: rgba(239, 68, 68, 0.2);
            border-color: rgba(239, 68, 68, 0.5);
            color: #fef2f2;
        }

        footer, header {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.title("🔮 Customer Churn Predictor")
st.markdown("### Enter customer details to predict churn probability.")

# Create a container for the form
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    # ---------------------------
    # Input Layout
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], help="1 for Yes, 0 for No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

    with col2:
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

    # ---------------------------
    # Prediction Section
    # ---------------------------
    # This section is wrapped inside the container but the logic is unchanged.
    if st.button("🔍 Predict Churn", use_container_width=True):
        input_data = {
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        
        try:
            df = pd.DataFrame([input_data])
            for col, encoder in encoders.items():
                if col in df.columns:
                    # Handle unseen labels gracefully if necessary
                    df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                    df[col] = encoder.transform(df[col])
            
            # Reorder columns to match model's training order if necessary
            # df = df[feature_names]

            prediction = model.predict(df)[0]
            pred_prob = model.predict_proba(df)[0][1]

            if prediction == 1:
                st.markdown(f"""
                <div class="result-box error-box">
                    ⚠️ <b>Prediction: Customer is likely to Churn</b><br>
                    Confidence: {pred_prob:.0%}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box success-box">
                    ✅ <b>Prediction: Customer will likely Stay</b><br>
                    Churn Probability: {pred_prob:.0%}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")

    # Close the custom container div
    st.markdown('</div>', unsafe_allow_html=True)

