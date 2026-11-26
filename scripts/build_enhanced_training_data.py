
# HospitalFlow AI
# Enhanced Feature Engineering - Version 2

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

ML_DIR = BASE_DIR / "data" / "processed" / "ml"
AGENT_DIR = BASE_DIR / "data" / "processed" / "agents"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "ml_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "length_of_stay_hours"

KEYS = ["patient_id", "encounter_id"]

# ==========================================
# 2. Load original training dataset
# ==========================================

df = pd.read_csv(
    ML_DIR / "discharge_training_data.csv",
    dtype={
        "patient_id": "string",
        "encounter_id": "string"
    }
)

with open(
    ML_DIR / "feature_config.json"
) as file:
    config = json.load(file)

original_count = len(df)

for column in ["admission_time", "discharge_time"]:

    df[column] = pd.to_datetime(
        df[column],
        format="ISO8601",
        errors="raise"
    )

df["_row_id"] = np.arange(len(df))

print("Original dataset shape:", df.shape)

# ==========================================
# 3. Load ED admission information
# ==========================================

admissions = pd.read_csv(
    AGENT_DIR / "admission_events.csv",
    dtype={
        "patient_id": "string",
        "encounter_id": "string"
    },
    usecols=[
        "patient_id",
        "encounter_id",
        "event_type",
        "priority",
        "arrival_time",
        "triage_time",
        "disposition",
        "disposition_time"
    ]
)

# ==========================================
# 4. Extract recorded ED priority
# ==========================================

arrivals = admissions.loc[
    admissions["event_type"] == "ED_ARRIVAL",
    KEYS + [
        "priority",
        "arrival_time",
        "triage_time"
    ]
].copy()

for column in ["arrival_time", "triage_time"]:

    arrivals[column] = pd.to_datetime(
        arrivals[column],
        format="ISO8601",
        errors="coerce"
    )

arrivals = arrivals.sort_values(
    "arrival_time"
).drop_duplicates(
    KEYS,
    keep="first"
)

df = df.merge(
    arrivals,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

valid_priority = (
    df["arrival_time"].notna()
    & df["triage_time"].notna()
    & (df["arrival_time"] <= df["triage_time"])
    & (df["triage_time"] <= df["admission_time"])
    & df["priority"].isin([1, 2, 3, 4, 5])
)

df["ed_priority"] = df["priority"].where(
    valid_priority
)

print(
    "Valid ED priorities:",
    df["ed_priority"].notna().sum()
)

# ==========================================
# 5. Extract ED admission disposition
# ==========================================

dispositions = admissions.loc[
    admissions["event_type"] == "ED_DISPOSITION",
    KEYS + [
        "disposition",
        "disposition_time"
    ]
].copy()

dispositions["disposition_time"] = pd.to_datetime(
    dispositions["disposition_time"],
    format="ISO8601",
    errors="coerce"
)

dispositions = dispositions.sort_values(
    "disposition_time"
).drop_duplicates(
    KEYS,
    keep="first"
)

df = df.merge(
    dispositions,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

valid_disposition = (
    df["disposition_time"].notna()
    & (
        df["disposition_time"]
        <= df["admission_time"]
    )
)

df["admission_disposition"] = (
    df["disposition"]
    .where(valid_disposition)
    .fillna("Unknown")
)

print("\nAdmission disposition distribution:")

print(
    df["admission_disposition"]
    .value_counts()
    .to_string()
)

# ==========================================
# 6. Construct completed-stay history
# ==========================================

history = df[
    [
        "patient_id",
        "discharge_time",
        TARGET
    ]
].copy()

history = history.sort_values(
    ["patient_id", "discharge_time"]
)

history["historical_stay_count"] = (
    history.groupby("patient_id")
    .cumcount() + 1
)

history["historical_total_los"] = (
    history.groupby("patient_id")[TARGET]
    .cumsum()
)

history["historical_mean_los"] = (
    history["historical_total_los"]
    / history["historical_stay_count"]
)

history["historical_last_los"] = (
    history[TARGET]
)

history = history.rename(
    columns={
        "discharge_time":
        "previous_discharge_time"
    }
)

history = history[
    [
        "patient_id",
        "previous_discharge_time",
        "historical_stay_count",
        "historical_mean_los",
        "historical_last_los"
    ]
]

# ==========================================
# 7. Match only previously completed stays
# ==========================================

# merge_asof selects the latest historical
# discharge strictly before current admission.

df = pd.merge_asof(
    df.sort_values("admission_time"),
    history.sort_values("previous_discharge_time"),
    left_on="admission_time",
    right_on="previous_discharge_time",
    by="patient_id",
    direction="backward",
    allow_exact_matches=False
)

df["historical_stay_count"] = (
    df["historical_stay_count"]
    .fillna(0)
    .astype(int)
)

df["days_since_last_discharge"] = (
    df["admission_time"]
    - df["previous_discharge_time"]
).dt.total_seconds() / 86400

# ==========================================
# 8. Validate temporal leakage
# ==========================================

has_history = (
    df["previous_discharge_time"].notna()
)

invalid_history = (
    has_history
    & (
        df["previous_discharge_time"]
        >= df["admission_time"]
    )
)

if invalid_history.any():

    raise ValueError(
        "Future information detected in history."
    )

if len(df) != original_count:

    raise ValueError(
        "Unexpected change in encounter count."
    )

if df.duplicated(KEYS).any():

    raise ValueError(
        "Duplicate encounters detected."
    )

print(
    "\nHistorical leakage violations:",
    invalid_history.sum()
)

print(
    "Encounters with previous completed stays:",
    has_history.sum()
)

print(
    "Encounters without previous completed stays:",
    (~has_history).sum()
)

# ==========================================
# 9. Define enhanced feature configuration
# ==========================================

new_numeric_features = [
    "ed_priority",
    "historical_stay_count",
    "historical_mean_los",
    "historical_last_los",
    "days_since_last_discharge"
]

new_categorical_features = [
    "admission_disposition"
]

numeric_features = (
    config["numeric_features"]
    + new_numeric_features
)

categorical_features = (
    config["categorical_features"]
    + new_categorical_features
)

metadata_columns = config["metadata_columns"]

FEATURES = (
    numeric_features + categorical_features
)

# ==========================================
# 10. Prepare final dataset
# ==========================================

final_columns = (
    metadata_columns
    + FEATURES
    + [TARGET]
)

enhanced = df[final_columns].copy()

enhanced = enhanced.sort_values(
    "admission_time"
).reset_index(drop=True)

# Missing historical LOS values are intentional.
# They indicate no previous observed stay.
# Imputation will occur inside the ML pipeline.

enhanced.to_csv(
    OUTPUT_DIR / "discharge_training_data.csv",
    index=False
)

# ==========================================
# 11. Save feature configuration
# ==========================================

enhanced_config = {
    "target": TARGET,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metadata_columns": metadata_columns,
    "version": 2
}

with open(
    OUTPUT_DIR / "feature_config.json",
    "w"
) as file:

    json.dump(
        enhanced_config,
        file,
        indent=4
    )

# ==========================================
# 12. Save feature summary
# ==========================================

summary = pd.DataFrame({
    "feature": FEATURES,
    "missing_values": [
        enhanced[column].isna().sum()
        for column in FEATURES
    ],
    "unique_values": [
        enhanced[column].nunique()
        for column in FEATURES
    ]
})

summary.to_csv(
    OUTPUT_DIR / "enhanced_feature_summary.csv",
    index=False
)

# ==========================================
# 13. Display results
# ==========================================

print("\n" + "=" * 60)
print("ENHANCED DATASET SUMMARY")
print("=" * 60)

print("Total records:", len(enhanced))

print(
    "Numerical features:",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)

print("Total input features:", len(FEATURES))

print("Final dataset shape:", enhanced.shape)

print("\nNew feature statistics:")

print(
    enhanced[new_numeric_features]
    .describe()
    .round(2)
    .to_string()
)

print("\nFeature missingness:")

print(summary.to_string(index=False))

print("\nDataset saved to:", OUTPUT_DIR)

print("\nENHANCED FEATURE ENGINEERING COMPLETED")
