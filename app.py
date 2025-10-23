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
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a telecom customer is likely to **Churn** or **Stay** using a trained Random Forest model.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
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
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# Prepare input
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
if st.button("🔍 Predict"):
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
            st.error(f"⚠️ The customer is **likely to Churn**.\n\n**Probability:** {pred_prob:.2f}")
        else:
            st.success(f"✅ The customer will **Stay**.\n\n**Churn Probability:** {pred_prob:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
