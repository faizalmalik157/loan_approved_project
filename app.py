import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Page config ----------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0B1F3A 0%, #133A63 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.8rem;
    }
    .hero h1 { margin: 0; font-size: 1.9rem; }
    .hero p { margin-top: 0.5rem; color: #C9D6E3; font-size: 0.95rem; }

    div.stButton > button, div.stFormSubmitButton > button {
        background-color: #0F9D74;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #0C7C5C;
        color: white;
    }

    .result-card {
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .approved { background-color: #E6F7EF; color: #0C7C5C; border: 1px solid #0F9D74; }
    .rejected { background-color: #FDECEC; color: #B3261E; border: 1px solid #E63946; }

    .footer {
        color: #94A3B8;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero header ----------
st.markdown(
    """
    <div class="hero">
      <h1>🏦 Loan Approval Predictor</h1>
      <p>Fill in the applicant details below to get an instant, model-backed loan decision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Load model artifacts (cached so they load once) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_best_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

# ---------- Input form ----------
with st.form("loan_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, step=500)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=500)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, step=10)
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 480, 240, 120, 60])
        credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Check Loan Eligibility")

# ---------- Prediction ----------
if submitted:
    credit_val = 1 if credit_history.startswith("Good") else 0
    total_income = applicant_income + coapplicant_income
    income_loan_ratio = total_income / loan_amount if loan_amount > 0 else 0
    emi = loan_amount / loan_term if loan_term > 0 else 0

    raw_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_val,
        "Property_Area": property_area,
        "TotalIncome": total_income,
        "Income_Loan_Ratio": income_loan_ratio,
        "EMI": emi,
    }

    input_df = pd.DataFrame([raw_input])
    input_encoded = pd.get_dummies(input_df)

    # Align the one-hot encoded columns to whatever the scaler was fitted on,
    # so column order/missing dummy columns can't silently break predictions.
    expected_cols = getattr(scaler, "feature_names_in_", None)
    if expected_cols is not None:
        input_encoded = input_encoded.reindex(columns=expected_cols, fill_value=0)
    else:
        st.warning(
            "Couldn't detect the exact training column order from the saved scaler "
            "(it wasn't fit on a DataFrame with column names). Double check that "
            "the columns below match your training features exactly, in order."
        )

    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]

    proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        confidence = f" ({proba[1] * 100:.1f}% confidence)" if proba is not None else ""
        st.markdown(
            f'<div class="result-card approved">✅ Loan Approved{confidence}</div>',
            unsafe_allow_html=True,
        )
    else:
        confidence = f" ({proba[0] * 100:.1f}% confidence)" if proba is not None else ""
        st.markdown(
            f'<div class="result-card rejected">❌ Loan Rejected{confidence}</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">Built with Streamlit · Random Forest loan approval model</div>',
    unsafe_allow_html=True,
)