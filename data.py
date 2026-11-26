"""Dataset loading, cleaning, and feature selection for ChurnIQ."""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"

# Columns used by the churn classifier in this Streamlit application.
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Load raw CSV, standardise values, impute TotalCharges and create target flag."""
    if uploaded_bytes is not None:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        df = pd.read_csv(DATA_PATH)

    # Remove accidental spaces around column names and text values.
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Empty strings in TotalCharges represent missing numeric values.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    # Convert SeniorCitizen from 0/1 into readable category labels.
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes", "0": "No", "1": "Yes"})
    df["ChurnFlag"] = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)
    return df


def classification_xy(df: pd.DataFrame):
    """Churn task: predict Churn using customer profile, service and billing data."""
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df["ChurnFlag"]
    return X, y


def regression_xy(df: pd.DataFrame):
    """Monthly-charge task: predict MonthlyCharges without using the target or TotalCharges."""
    cols = ["tenure"] + CATEGORICAL
    return df[cols].copy(), df["MonthlyCharges"]


def load_raw_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Load the untouched CSV for showing before/after preprocessing results."""
    if uploaded_bytes is not None:
        raw = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        raw = pd.read_csv(DATA_PATH)
    raw.columns = raw.columns.str.strip()
    return raw
