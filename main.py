import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="CredTrust Nigeria",
    description="""
    AI-Powered Credit Risk Assessment Platform

    Predicting loan default risk using machine learning,
    borrower financial data, and macroeconomic indicators.
    """,
    version="1.0.0"
)

# Safe absolute path loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_default_random_forest.pkl")

# Load your Pipeline model
model = joblib.load(MODEL_PATH)

# Added application_date so Python can auto-engineer the missing columns!
class LoanApplicant(BaseModel):
    application_date: str  # Format: "YYYY-MM-DD" e.g., "2024-02-01"
    principal_ngn: float
    interest_rate_annual: float
    tenor_months: int
    monthly_payment_ngn: float
    salary_ngn: float
    debt_to_income_ratio: float
    credit_score: int
    employment_type: str
    loan_purpose: str
    state: str
    salary_detected: bool
    dpd_current: int
    dpd_max_90d: int

@app.post("/predict")
def predict_default(applicant: LoanApplicant):
    # 1. Gather all basic fields into a dictionary
    input_dict = {
        "principal_ngn": [applicant.principal_ngn],
        "interest_rate_annual": [applicant.interest_rate_annual],
        "tenor_months": [applicant.tenor_months],
        "monthly_payment_ngn": [applicant.monthly_payment_ngn],
        "salary_ngn": [applicant.salary_ngn],
        "debt_to_income_ratio": [applicant.debt_to_income_ratio],
        "credit_score": [applicant.credit_score],
        "employment_type": [applicant.employment_type],
        "loan_purpose": [applicant.loan_purpose],
        "state": [applicant.state],
        "salary_detected": [applicant.salary_detected],
        "dpd_current": [applicant.dpd_current],
        "dpd_max_90d": [applicant.dpd_max_90d]
    }

    # 2. Convert to a temporary DataFrame
    input_df = pd.DataFrame(input_dict)
    
    # 3. AUTO-ENGINEER THE MISSING COLUMNS FROM THE INPUT DATE STRING
    try:
        date_obj = pd.to_datetime(applicant.application_date)
        input_df["application_year"] = date_obj.year
        input_df["application_month"] = date_obj.month
        input_df["application_dayofweek"] = date_obj.dayofweek
    except Exception:
        # Fallback placeholders if date format is invalid
        input_df["application_year"] = 2026
        input_df["application_month"] = 8
        input_df["application_dayofweek"] = 3

    # 4. Run the Pipeline prediction and extract metrics safely
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    
    # Extract probability of defaulting (Index 1)
    default_prob = float(probabilities[0][1]) 
    
    # 5. Define operational risk tiers for the fintech lender
    if default_prob < 0.20:
        result_text = "Low Risk of 90-Day Default"
    elif default_prob < 0.55:
        result_text = "Medium Risk - Proceed with Caution"
    else:
        result_text = "High Risk of 90-Day Default"

    # 6. Return clean JSON format
    return {
        "prediction": bool(prediction[0]), 
        "default_probability": round(default_prob, 4),
        "result": result_text
    }

from pydantic import BaseModel

class LoanPredictionInput(BaseModel):
    # Original Profile Fields
    income: float
    credit_score: int
    loan_amount: float
    employment_duration_months: int

    # New Nigerian Macroeconomic Indicators
    cbn_mpr: float         # Baseline: 26.50
    inflation_rate: float  # Baseline: 15.91
    usd_ngn_exchange_rate: float # Baseline: 1375.0



import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 🇳🇬 Automated macroeconomic utility helper
def get_nigerian_macro_indicators():
    """
    Automatically fetches live USD/NGN exchange rates via a free API
    and combines them with current CBN MPR and inflation baselines.
    """
    # 1. Set current official CBN baseline values (August 2026)
    macro_data = {
        "cbn_mpr": 26.50,        # Central Bank Monetary Policy Rate
        "inflation_rate": 15.91  # Headline inflation baseline
    }

    # 2. Automatically fetch live exchange rates from a free API
    try:
        url = "https://er-api.com"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            rates = response.json().get("rates", {})
            # Extract NGN exchange rate value dynamically
            macro_data["usd_ngn_exchange_rate"] = rates.get("NGN", 1375.0)
        else:
            macro_data["usd_ngn_exchange_rate"] = 1375.0  # Safe fallback

        Exception:
        macro_data["usd_ngn_exchange_rate"] = 1375.0      # Safe network fallback
        
    return macro_data


# The user only inputs individual profile info
class BorrowerProfile(BaseModel):
    income: float
    credit_score: int
    loan_amount: float
    employment_duration_months: int


    @app.post("/predict")
def predict_loan_default(profile: BorrowerProfile):
    # 1. Automatically fetch the live national macro numbers
    macro = get_nigerian_macro_indicators()

    # 2. Combine individual user profile with the macro metrics
    full_features = [
        profile.income,
        profile.credit_score,
        profile.loan_amount,
        # 2. Combine individual user profile with the macro metrics
    full_features = [
        profile.income,
        profile.credit_score,
        profile.loan_amount,

        # 3. Pass full_features directly into your loaded .pkl model
    # prediction = model.predict([full_features])
    
    return {
        "status": "Success",
        "fetched_macro_context": macro,
        "prediction_input_features": full_features
    }