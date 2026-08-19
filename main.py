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