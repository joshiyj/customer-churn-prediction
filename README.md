# 📊 Customer Churn Prediction App

A Machine Learning–powered **Customer Churn Prediction** web application built with **Streamlit**.  
It predicts whether a telecom customer is likely to **churn** or **stay**, based on their account and service details.

🔗 **Live App:** [Customer Churn Prediction](https://customer-churn-prediction-n7ol7q2ums7twpuzbt85iz.streamlit.app/)

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

## 💻 Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/customer-churn-streamlit.git
cd customer-churn-streamlit
