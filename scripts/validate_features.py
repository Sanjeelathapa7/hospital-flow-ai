
# CN7050 - Intelligent Systems
# HospitalFlow AI
# Phase 4.2 - Feature Validation

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed" / "agents"
RAW_DIR = BASE_DIR / "data" / "raw"

# Features requiring validation
validation_config = {

    "patient_features.csv": [
        "age",
        "BMI",
        "comorbidity_burden",
        "clinical_complexity_category",
        "baseline_risk_category",
        "expected_care_complexity"
    ],

    "admission_events.csv": [
        "event_type",
        "priority",
        "disposition",
        "event_status",
        "arrival_time",
        "wait_time_minutes"
    ],

    "bed_events.csv": [
        "event_type",
        "bed_type",
        "occupancy_status",
        "availability_status",
        "care_level",
        "transfer_status",
        "assignment_time",
        "release_time"
    ],

    "discharge_events.csv": [
        "event_type",
        "care_level",
        "admission_time",
        "discharge_time",
        "actual_discharge_time",
        "length_of_stay_hours",
        "length_of_stay_days"
    ],

    "coordination_events.csv": [
        "event_type",
        "staff_role",
        "shift_type",
        "workload_category",
        "scheduled_staff_count",
        "active_staff_count",
        "escalation_flag"
    ]
}

# Inspect each dataset
for filename, columns in validation_config.items():

    print("\n" + "=" * 70)
    print(filename.upper())
    print("=" * 70)

    df = pd.read_csv(
        DATA_DIR / filename,
        usecols=columns,
        low_memory=False
    )

    print("Total records:", len(df))

    for column in columns:

        print(f"\nColumn: {column}")
        print("Data type:", df[column].dtype)
        print("Missing values:", df[column].isna().sum())
        print("Unique values:", df[column].nunique())

        if df[column].nunique() <= 30:

            print("Value distribution:")
            print(
                df[column].value_counts(
                    dropna=False
                ).to_string()
            )

        elif pd.api.types.is_numeric_dtype(df[column]):

            print("Numerical statistics:")
            print(df[column].describe())

        else:

            print("Example values:")
            print(
                df[column]
                .dropna()
                .head(5)
                .tolist()
            )

# Inspect length-of-stay target by event type
print("\n" + "=" * 70)
print("LENGTH OF STAY TARGET VALIDATION")
print("=" * 70)

discharge = pd.read_csv(
    DATA_DIR / "discharge_events.csv",
    low_memory=False
)

discharge["length_of_stay_hours"] = pd.to_numeric(
    discharge["length_of_stay_hours"],
    errors="coerce"
)

print(
    discharge.groupby("event_type")[
        "length_of_stay_hours"
    ].agg(
        ["size", "count", "min", "max", "mean", "median"]
    )
)

# Check negative and zero target values
print(
    "\nNegative LOS values:",
    (discharge["length_of_stay_hours"] < 0).sum()
)

print(
    "Zero LOS values:",
    (discharge["length_of_stay_hours"] == 0).sum()
)

# Check repeated encounters
valid_los = discharge.dropna(
    subset=[
        "patient_id",
        "encounter_id",
        "length_of_stay_hours"
    ]
)

print("\nRecords with valid LOS:", len(valid_los))

print(
    "Unique patient-encounter pairs:",
    valid_los[
        ["patient_id", "encounter_id"]
    ].drop_duplicates().shape[0]
)

print(
    "Repeated patient-encounter pairs:",
    valid_los.duplicated(
        subset=["patient_id", "encounter_id"]
    ).sum()
)

# Inspect relevant dictionary definitions
dictionary = pd.read_csv(
    RAW_DIR / "DATA_DICTIONARY (1).csv",
    dtype="string"
)

terms = [
    "length_of_stay_hours",
    "bed_id",
    "bed_event_id",
    "priority",
    "availability_status",
    "occupancy_status",
    "expected_care_complexity",
    "admission_time",
    "actual_discharge_time"
]

mask = dictionary.apply(
    lambda col: col.isin(terms)
).any(axis=1)

print("\n" + "=" * 70)
print("RELEVANT DATA DICTIONARY DEFINITIONS")
print("=" * 70)

print(dictionary.loc[mask].to_string(index=False))

print("\nFEATURE VALIDATION COMPLETED")
