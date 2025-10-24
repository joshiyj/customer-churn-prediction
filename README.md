# 📊 Customer Churn Prediction App

A Machine Learning–powered **Customer Churn Prediction** web application built with **Streamlit**.  
It predicts whether a telecom customer is likely to **churn** or **stay**, based on their account and service details.

🔗 **Live App:** [Customer Churn Prediction](https://customer-churn-prediction-ml-capstone.streamlit.app/)

---

## 🚀 Project Overview

Customer churn — when customers stop doing business with a company — is a critical problem for telecom companies.  
This app uses a trained **Random Forest Classifier** to predict churn probability from customer data.

Users can interactively input customer information and instantly see:
- The predicted churn status
- The churn probability score

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier  
- **Encoders:** Label Encoders (for categorical features)  
- **Imbalanced Data Handling:** SMOTE (Synthetic Minority Over-sampling Technique)  
- **Training Dataset:** Telco Customer Churn dataset (from Kaggle / IBM Sample Data)

---

## 🧩 Features Used

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

---

## 🛠️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| Frontend UI | Streamlit |
| Backend Logic | Python |
| ML Model | scikit-learn (RandomForestClassifier) |
| Data Handling | Pandas, NumPy |
| Deployment | Streamlit Cloud |
| Pickle Serialization | pickle |

---

## 📸UI/App :
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/6103e46d-bfe1-4783-85ee-c7d647b10299" />
<img width="1919" height="762" alt="image" src="https://github.com/user-attachments/assets/d589da24-832c-4f8f-bb77-6fdb78dece3f" />

---

## 💻 Run Locally

1️⃣ Clone the Repository
git clone https://github.com/joshiyj/customer-churn-prediction.git
cd customer-churn-prediction

2️⃣ Create and Activate a Virtual Environment

For Windows:

python -m venv venv
venv\Scripts\activate


For macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py


After running the command, Streamlit will launch the app and display a local URL (something like http://localhost:8501/).
Open that link in your browser to interact with the Customer Churn Prediction App.
