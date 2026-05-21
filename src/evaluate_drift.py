import os
import urllib.parse
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

load_dotenv()

safe_password = urllib.parse.quote_plus(os.getenv("SUPABASE_PASSWORD", ""))

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('SUPABASE_USER')}:{safe_password}"
    f"@{os.getenv('SUPABASE_HOST')}:{os.getenv('SUPABASE_PORT')}/{os.getenv('SUPABASE_DBNAME')}"
)

features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "IsActiveMember",
]

# loading data from Supabase
df_all = pd.read_sql("SELECT * FROM fct_churn_predictions", engine)

df_clean = df_all[df_all["scenario_mode"] == "CLEAN"].copy()
df_recession = df_all[df_all["scenario_mode"] == "RECESSION"].copy()

if df_clean.empty:
    raise ValueError("No CLEAN rows found in fct_churn_predictions")

if df_recession.empty:
    raise ValueError("No RECESSION rows found in fct_churn_predictions")

# keeping only the features that actually exist in the table
available_features = [f for f in features if f in df_all.columns]

if not available_features:
    raise ValueError("None of the monitoring features exist in fct_churn_predictions")

# making sure all monitored columns are numeric
for col in available_features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_recession[col] = pd.to_numeric(df_recession[col], errors="coerce")

df_clean_features = df_clean[available_features].dropna()
df_recession_features = df_recession[available_features].dropna()

# CLEAN = healthy baseline state
clean_rows = [
    {
        "scenario_mode": "CLEAN",
        "feature_name": feature,
        "drift_score": 0.0,
        "p_value": 1.0,
        "drift_detected": 0,
        "global_dataset_drift": 0,
        "share_of_drifted_features": 0.0,
        "monitoring_status": "Healthy",
    }
    for feature in available_features
]

# RECESSION = compare CLEAN baseline against incoming RECESSION data
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(
    reference_data=df_clean_features,
    current_data=df_recession_features,
)

drift_results = drift_report.as_dict()["metrics"][1]["result"]

recession_rows = []

for feature, details in drift_results["drift_by_columns"].items():
    drift_score = float(details.get("drift_score", 0.0))

    # Evidently's own result
    evidently_drift_detected = bool(details.get("drift_detected", False))

    # if drift_score is high enough, it's shown as drift
    demo_drift_detected = drift_score >= 0.10

    drift_detected = evidently_drift_detected or demo_drift_detected

    recession_rows.append(
        {
            "scenario_mode": "RECESSION",
            "feature_name": feature,
            "drift_score": drift_score,
            "p_value": float(details.get("p_value")) if details.get("p_value") is not None else None,
            "drift_detected": 1 if drift_detected else 0,
            "global_dataset_drift": 0,  # filled below
            "share_of_drifted_features": 0.0,  # filled below
            "monitoring_status": "Drift detected" if drift_detected else "Healthy",
        }
    )

# calculating demo-level global drift
num_drifted = sum(row["drift_detected"] for row in recession_rows)
share_drifted = num_drifted / len(recession_rows) if recession_rows else 0.0
global_drift = 1 if num_drifted > 0 else 0

for row in recession_rows:
    row["global_dataset_drift"] = global_drift
    row["share_of_drifted_features"] = share_drifted
    row["monitoring_status"] = "Drift detected" if global_drift else "Healthy"

monitoring_df = pd.DataFrame(clean_rows + recession_rows)

print("\nGlobal drift by scenario:")
print(monitoring_df.groupby("scenario_mode")["global_dataset_drift"].mean())

print("\nNumber of drifted features by scenario:")
print(monitoring_df.groupby("scenario_mode")["drift_detected"].sum())

print("\nMonitoring preview:")
print(monitoring_df)

monitoring_df.to_sql(
    "model_monitoring_metrics",
    engine,
    if_exists="replace",
    index=False,
)

print("\nMonitoring table updated with CLEAN and RECESSION states.")