"""Phase 2 feature engineering, as a module that later phases can import.

This is the same code as section 2 of notebooks/02_phase2_code.ipynb (the notebook is the source of truth;
this file is a copy of it, checked to give identical output). Phase 3 and Phase 4 use it to turn raw customer
rows (the Phase 1 columns) into the Phase 2 columns that models/pipeline_v2.pkl expects.

Usage, from the phase2 folder:

    import pickle
    import pandas as pd
    from src.phase2_features import add_features, to_model_features

    pipeline = pickle.load(open("models/pipeline_v2.pkl", "rb"))     # needs the sklearn-features package
    new_customers = pd.read_csv("data/input/test.csv")               # any rows with the Phase 1 columns
    X = to_model_features(add_features(new_customers), pipeline)     # 49 columns, in the order the models use
"""

import numpy as np
import pandas as pd

target = "churn"

# Phase 1 column groups (unchanged)
numeric_cols = ["age", "annual_income", "dependents", "tenure", "monthlycharges", "totalcharges",
                "num_services", "customer_satisfaction", "num_complaints", "num_service_calls",
                "late_payments", "avg_monthly_gb", "days_since_last_interaction", "credit_score"]

service_cols = ["has_phone_service", "has_internet_service", "has_online_security", "has_online_backup",
                "has_device_protection", "has_tech_support", "has_streaming_tv", "has_streaming_movies"]

binary_cols = ["senior_citizen", "paperless_billing"] + service_cols

category_cols = ["gender", "education", "marital_status", "contract", "payment_method"]

# Phase 2 additions
addon_cols = service_cols[2:]
skewed_cols = ["annual_income", "totalcharges", "avg_monthly_gb", "days_since_last_interaction"]

new_numeric = ["active_services", "complaints_per_month", "calls_per_month", "late_payments_per_month",
               "charge_ratio", "m2m_x_complaints"]

# The column lists that pipeline_v2 uses
numeric_v2 = ["age", "log_annual_income", "dependents", "tenure", "monthlycharges", "log_totalcharges",
              "num_services", "customer_satisfaction", "num_complaints", "num_service_calls", "late_payments",
              "log_avg_monthly_gb", "log_days_since_last_interaction", "credit_score"] + new_numeric
binary_v2 = binary_cols + ["addons_without_internet"]
category_v2 = category_cols + ["tenure_group"]


def add_features(data):
    """Add the 12 Phase 2 features. Uses only the customer's own columns (never churn), so no leakage."""
    data = data.copy()
    months = data["tenure"].replace(0, 1)            # a new customer (tenure 0) counts as 1 month

    # 1. tenure groups (the last group is open, so a new customer with tenure > 72 still gets a group)
    data["tenure_group"] = pd.cut(data["tenure"], bins=[-1, 6, 12, 24, 48, np.inf],
                                  labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(str)

    # 2. number of active services, counted from the 8 has_* columns
    data["active_services"] = data[service_cols].sum(axis=1)

    # 3. internet add-ons without internet service (0/1)
    data["addons_without_internet"] = ((data["has_internet_service"] == 0) &
                                       (data[addon_cols].sum(axis=1) > 0)).astype(int)

    # 4-6. rates per month of tenure
    data["complaints_per_month"] = data["num_complaints"] / months
    data["calls_per_month"] = data["num_service_calls"] / months
    data["late_payments_per_month"] = data["late_payments"] / months

    # 7. billing ratio: current monthly charge / average monthly charge so far
    average_charge = (data["totalcharges"] / months).replace(0, np.nan)
    data["charge_ratio"] = data["monthlycharges"] / average_charge

    # 8. interaction: complaints of month-to-month customers
    data["m2m_x_complaints"] = (data["contract"] == "month_to_month").astype(int) * data["num_complaints"]

    # 9. log of the skewed columns (+1 because log(0) is undefined)
    for col in skewed_cols:
        data["log_" + col] = np.log(data[col] + 1)

    return data


def model_feature_names(pipeline):
    """The 49 output column names of a fitted pipeline_v2, in order."""
    ohe = pipeline.named_transformers["categorical"].named_steps["ohe"]
    return numeric_v2 + list(ohe.get_feature_names_out(category_v2)) + binary_v2


def to_model_features(data_v2, pipeline):
    """Run a fitted pipeline_v2 on rows that already have the Phase 2 columns; returns a named DataFrame.

    Extra columns (customer_id, churn) are ignored, because the pipeline selects its columns by name.
    """
    return pd.DataFrame(pipeline.transform(data_v2), columns=model_feature_names(pipeline), index=data_v2.index)
