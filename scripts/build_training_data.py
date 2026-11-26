# Project: HospitalFlow AI
# Phase 4.3 - Machine Learning Dataset Preparation

from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. Define project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

AGENT_DIR = BASE_DIR / "data" / "processed" / "agents"
ML_DIR = BASE_DIR / "data" / "processed" / "ml"

ML_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Load prepared datasets
# ==========================================

patients = pd.read_csv(
    AGENT_DIR / "patient_features.csv",
    dtype={"patient_id": "string"}
)

discharge = pd.read_csv(
    AGENT_DIR / "discharge_events.csv",
    dtype={
        "patient_id": "string",
        "encounter_id": "string"
    },
    low_memory=False
)

print("Patient dataset:", patients.shape)
print("Discharge dataset:", discharge.shape)

# ==========================================
# 3. Select inpatient encounters
# ==========================================

inpatient = discharge.loc[
    discharge["event_type"] == "INPATIENT_STAY",
    [
        "patient_id",
        "encounter_id",
        "admission_time",
        "discharge_time",
        "length_of_stay_hours"
    ]
].copy()

print("\nInpatient records:", len(inpatient))

# ==========================================
# 4. Validate target variable
# ==========================================

TARGET = "length_of_stay_hours"

inpatient[TARGET] = pd.to_numeric(
    inpatient[TARGET],
    errors="coerce"
)

initial_rows = len(inpatient)

inpatient = inpatient.dropna(
    subset=[
        "patient_id",
        "encounter_id",
        TARGET
    ]
)

inpatient = inpatient[
    np.isfinite(inpatient[TARGET])
    & (inpatient[TARGET] > 0)
].copy()

print(
    "Invalid target or ID records removed:",
    initial_rows - len(inpatient)
)

# ==========================================
# 5. Validate unique encounters
# ==========================================

duplicate_count = inpatient.duplicated(
    subset=["patient_id", "encounter_id"]
).sum()

print("Duplicate encounters:", duplicate_count)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate encounters require investigation."
    )

if patients["patient_id"].duplicated().any():
    raise ValueError(
        "Duplicate patient identifiers detected."
    )

# ==========================================
# 6. Convert timestamps
# ==========================================

for column in ["admission_time", "discharge_time"]:

    inpatient[column] = pd.to_datetime(
        inpatient[column],
        format="ISO8601",
        errors="coerce"
    )

invalid_timestamps = inpatient[
    ["admission_time", "discharge_time"]
].isna().any(axis=1)

print(
    "Invalid timestamp records:",
    invalid_timestamps.sum()
)

inpatient = inpatient.loc[
    ~invalid_timestamps
].copy()

# Validate recorded length of stay

calculated_los = (
    inpatient["discharge_time"]
    - inpatient["admission_time"]
).dt.total_seconds() / 3600

# Allow small differences due to rounding

consistent_los = (
    (calculated_los > 0)
    & np.isclose(
        calculated_los,
        inpatient[TARGET],
        atol=0.1,
        rtol=0
    )
)

print(
    "Inconsistent LOS records:",
    (~consistent_los).sum()
)

inpatient = inpatient.loc[
    consistent_los
].copy()

# ==========================================
# 7. Merge patient and encounter data
# ==========================================

training = inpatient.merge(
    patients,
    on="patient_id",
    how="left",
    validate="many_to_one",
    indicator=True
)

unmatched = (
    training["_merge"] != "both"
).sum()

print("Unmatched patient records:", unmatched)

training = training.loc[
    training["_merge"] == "both"
].drop(columns="_merge")

# ==========================================
# 8. Feature engineering
# ==========================================

training["admission_hour"] = (
    training["admission_time"].dt.hour
)

training["admission_day_of_week"] = (
    training["admission_time"].dt.dayofweek
)

training["admission_month"] = (
    training["admission_time"].dt.month
)

training["is_weekend"] = (
    training["admission_day_of_week"] >= 5
).astype(int)

# ==========================================
# 9. Define numerical features
# ==========================================

numeric_features = [
    "age",
    "BMI",
    "chronic_condition_count",
    "comorbidity_burden",
    "baseline_medication_count",
    "prior_ED_visits",
    "prior_hospitalizations",
    "prior_outpatient_visits",
    "prior_surgeries",
    "previous_ICU_admissions",
    "admission_hour",
    "admission_day_of_week",
    "admission_month",
    "is_weekend"
]

for column in numeric_features:

    training[column] = pd.to_numeric(
        training[column],
        errors="coerce"
    )

    training[column] = training[column].replace(
        [np.inf, -np.inf],
        np.nan
    )

# ==========================================
# 10. Define categorical features
# ==========================================

categorical_features = [
    "clinical_complexity_category",
    "baseline_risk_category",
    "primary_department",
    "primary_condition_category",
    "expected_care_complexity"
]

for column in categorical_features:

    training[column] = (
        training[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

# ==========================================
# 11. Validate numerical ranges
# ==========================================

invalid_age = (
    (training["age"] < 0)
    | (training["age"] > 120)
)

training.loc[invalid_age, "age"] = np.nan

invalid_bmi = training["BMI"] <= 0

training.loc[invalid_bmi, "BMI"] = np.nan

# Check non-negative historical counts

non_negative_features = [
    "chronic_condition_count",
    "comorbidity_burden",
    "baseline_medication_count",
    "prior_ED_visits",
    "prior_hospitalizations",
    "prior_outpatient_visits",
    "prior_surgeries",
    "previous_ICU_admissions"
]

for column in non_negative_features:

    training.loc[
        training[column] < 0,
        column
    ] = np.nan

# ==========================================
# 12. Select final dataset columns
# ==========================================

metadata_columns = [
    "patient_id",
    "encounter_id",
    "admission_time",
    "discharge_time"
]

final_columns = (
    metadata_columns
    + numeric_features
    + categorical_features
    + [TARGET]
)

training = training[final_columns].copy()

training = training.sort_values(
    "admission_time"
).reset_index(drop=True)

if training.empty:
    raise ValueError(
        "No valid training records remain."
    )

# ==========================================
# 13. Save training dataset
# ==========================================

output_file = (
    ML_DIR / "discharge_training_data.csv"
)

training.to_csv(
    output_file,
    index=False
)

# ==========================================
# 14. Save feature configuration
# ==========================================

import json

feature_config = {
    "target": TARGET,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metadata_columns": metadata_columns
}

with open(
    ML_DIR / "feature_config.json",
    "w"
) as file:

    json.dump(
        feature_config,
        file,
        indent=4
    )

# ==========================================
# 15. Generate training data summary
# ==========================================

summary = pd.DataFrame({
    "metric": [
        "Total training records",
        "Unique patients",
        "Unique patient-encounter pairs",
        "Numerical features",
        "Categorical features",
        "Total input features",
        "Missing feature values",
        "Minimum LOS hours",
        "Maximum LOS hours",
        "Mean LOS hours",
        "Median LOS hours"
    ],
    "value": [
        len(training),
        training["patient_id"].nunique(),
        training[
            ["patient_id", "encounter_id"]
        ].drop_duplicates().shape[0],
        len(numeric_features),
        len(categorical_features),
        len(numeric_features + categorical_features),
        training[
            numeric_features + categorical_features
        ].isna().sum().sum(),
        training[TARGET].min(),
        training[TARGET].max(),
        training[TARGET].mean(),
        training[TARGET].median()
    ]
})

summary.to_csv(
    ML_DIR / "training_data_summary.csv",
    index=False
)

# ==========================================
# 16. Display final results
# ==========================================

print("\n" + "=" * 65)
print("MACHINE LEARNING DATASET SUMMARY")
print("=" * 65)

print(summary.to_string(index=False))

print("\nMissing values by feature:")

print(
    training[
        numeric_features + categorical_features
    ].isna().sum()
)

print("\nDataset shape:", training.shape)

print("\nDataset saved to:")
print(output_file)

print("\nTRAINING DATA PREPARATION COMPLETED")
