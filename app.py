import streamlit as st
import pickle
import pandas as pd

# ---------------------------
# Load model and encoders (Mocking Data Load for runnable example)
# NOTE: This assumes 'customer_churn_model.pkl' and 'encoders.pkl' exist
# ---------------------------

# Mocking the data load since I cannot access local files
try:
    model_data = pickle.load(open('customer_churn_model.pkl', 'rb'))
    model = model_data['model']
    feature_names = model_data['features_names']
    encoders = pickle.load(open('encoders.pkl', 'rb'))
except FileNotFoundError:
    # Fallback/Mock data for demonstration
    class MockModel:
        def predict(self, df): return [0]
        def predict_proba(self, df): return [[0.8, 0.2]] # Low churn probability

    class MockEncoder:
        def __init__(self, classes): self.classes_ = classes
        def transform(self, x):
            mapping = {val: i for i, val in enumerate(self.classes_)}
            return x.map(mapping)

    model = MockModel()
    feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                     'MonthlyCharges', 'TotalCharges']

    encoders = {
        'gender': MockEncoder(['Male', 'Female']),
        'Partner': MockEncoder(['Yes', 'No']),
        'Dependents': MockEncoder(['Yes', 'No']),
        'PhoneService': MockEncoder(['Yes', 'No']),
        'MultipleLines': MockEncoder(['No phone service', 'No', 'Yes']),
        'InternetService': MockEncoder(['DSL', 'Fiber optic', 'No']),
        'OnlineSecurity': MockEncoder(['Yes', 'No', 'No internet service']),
        'OnlineBackup': MockEncoder(['Yes', 'No', 'No internet service']),
        'DeviceProtection': MockEncoder(['Yes', 'No', 'No internet service']),
        'TechSupport': MockEncoder(['Yes', 'No', 'No internet service']),
        'StreamingTV': MockEncoder(['Yes', 'No', 'No internet service']),
        'StreamingMovies': MockEncoder(['Yes', 'No', 'No internet service']),
        'Contract': MockEncoder(['Month-to-month', 'One year', 'Two year']),
        'PaperlessBilling': MockEncoder(['Yes', 'No']),
        'PaymentMethod': MockEncoder(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    }
    st.warning("Note: Using mock model and encoders. Please ensure your actual `.pkl` files are available to run the real prediction.")


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide"  # Wide layout for better use of space
)

# ---------------------------
# Custom CSS for Professional Dark/Glassmorphism UI
# ---------------------------
st.markdown("""
    <style>
        /* ------------------------------
           1. Base and Fonts
        ------------------------------ */
        @import url('https://fonts.fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            /* Deep, professional dark gradient */
            background: linear-gradient(135deg, #0F172A, #1C0A4D, #0F172A);
            color: #E5E7EB;
            transition: all 0.3s ease-in-out;
        }

        /* ------------------------------
           2. Headers and Containers
        ------------------------------ */
        h1 {
            text-align: center;
            color: #F9FAFB;
            font-weight: 800;
            letter-spacing: 1px;
            padding-bottom: 5px;
            text-shadow: 0 0 10px rgba(99, 102, 241, 0.5); /* Subtle glow */
        }

        h3 {
            text-align: center;
            color: #A5B4FC;
            font-weight: 400;
            margin-top: -10px;
            margin-bottom: 30px;
        }

        /* Main content container */
        .block-container {
            padding-top: 3rem !important;
            max-width: 1100px; /* Wider for 3 columns */
            margin: auto;
        }

        /* Form container (Glassmorphism effect) - TARGETING THE STREMLIT FORM ELEMENT */
        div[data-testid="stForm"] > div:nth-child(2) {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 25px;
            padding: 2.5rem 3rem;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.3);
            backdrop-filter: blur(15px);
            transition: all 0.3s ease-in-out;
            color: #E0E7FF;
        }

        div[data-testid="stForm"] > div:nth-child(2):hover {
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        }

        /* ------------------------------
           3. Inputs and Selectors
        ------------------------------ */
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput div[data-baseweb="input"] > div {
            background-color: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            border-radius: 10px;
            transition: 0.2s ease-in-out;
        }

        .stSelectbox div[data-baseweb="select"] > div:hover,
        .stNumberInput div[data-baseweb="input"] > div:hover {
            border-color: #A78BFA !important; /* Lighter hover focus */
            background-color: rgba(255, 255, 255, 0.15);
        }

        label {
            color: #F8FAFC !important;
            font-weight: 600 !important;
            margin-bottom: 0.3rem !important;
        }

        /* ------------------------------
           4. Predict Button
        ------------------------------ */
        .stButton>button {
            width: 100%;
            /* Gradient for the button */
            background: linear-gradient(90deg, #6366F1, #A78BFA); 
            color: white;
            border: none;
            padding: 0.9rem 2rem;
            border-radius: 15px;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(167, 139, 250, 0.6);
            filter: brightness(1.1);
        }

        /* ------------------------------
           5. Result Boxes
        ------------------------------ */
        .result-box {
            backdrop-filter: blur(12px);
            border: 1px solid;
            padding: 1.8rem;
            border-radius: 20px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 2.5rem;
            transition: all 0.3s ease-in-out;
        }

        .success-box {
            background: rgba(34, 197, 94, 0.15);
            border-color: rgba(34, 197, 94, 0.5);
            color: #D1FAE5;
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.25);
        }

        .error-box {
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.5);
            color: #FECACA;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
        }
        
        /* Hide default footer/header */
        footer, header, #MainMenu {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)


# ---------------------------
# Header
# ---------------------------
st.title("Customer Churn Risk Analyzer")
st.markdown("")


# ---------------------------
# Main Form
# ---------------------------
# The form is now solely defined by st.form
with st.form("churn_form", clear_on_submit=False):
    # REMOVED the redundant st.markdown('<div class="form-container">')
    st.markdown("## Input Customer Parameters")
    st.divider()

    col1, col2, col3 = st.columns(3)

    # --- Column 1: Demographics & Tenure ---
    with col1:
        st.markdown("#### Demographics & Account")
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="1 for Yes, 0 for No")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1, step=1)

    # --- Column 2: Service Subscriptions ---
    with col2:
        st.markdown("#### Service Status")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])


    # --- Column 3: Billing & Contract ---
    with col3:
        st.markdown("#### Billing & Usage")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=1.0, value=50.0, step=1.0, format="%.2f")
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=50.0, step=1.0, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True) # Add some space before the button
    submitted = st.form_submit_button("🔍 Run Churn Prediction", use_container_width=True)
    # REMOVED the redundant st.markdown('</div>')

# ---------------------------
# Prediction Section
# ---------------------------
if submitted:
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
        # Apply encoders
        for col, encoder in encoders.items():
            if col in df.columns and col in encoders:
                # Handle unseen labels by setting them to the first known class
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                df[col] = encoder.transform(df[col])
        
        # Reorder columns to match model's training order
        df = df[feature_names]

        prediction = model.predict(df)[0]
        # Predict probability of Churn (class 1)
        pred_prob_churn = model.predict_proba(df)[0][1]
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box error-box">
                ⚠️ <b>HIGH CHURN RISK: Customer is likely to Churn</b><br>
                Churn Probability: {pred_prob_churn:.1%}
            </div>
            """, unsafe_allow_html=True)
            st.progress(pred_prob_churn, text=f"Predicted Churn Risk ({pred_prob_churn:.1%})")
        else:
            st.markdown(f"""
            <div class="result-box success-box">
                ✅ <b>LOW CHURN RISK: Customer will likely Stay</b><br>
                Churn Probability: {pred_prob_churn:.1%}
            </div>
            """, unsafe_allow_html=True)
            st.progress(pred_prob_churn, text=f"Predicted Churn Risk ({pred_prob_churn:.1%})")

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction. Please check your inputs and ensure model files are correct. Error: {e}")

# Adding a subtle footer outside the main form container
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #4B5563;">
        <small>© 2024 Predictive Analytics Dashboard</small>
    </div>
""", unsafe_allow_html=True)
