
# HospitalFlow AI
# Phase 5.4 - Predictive Feature Investigation

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]

AGENT_DIR = BASE_DIR / "data" / "processed" / "agents"
ML_DIR = BASE_DIR / "data" / "processed" / "ml"
EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "length_of_stay_hours"

# ==========================================
# 1. Load existing training dataset
# ==========================================

df = pd.read_csv(
    ML_DIR / "discharge_training_data.csv"
)

df["admission_time"] = pd.to_datetime(
    df["admission_time"],
    format="ISO8601"
)

df["discharge_time"] = pd.to_datetime(
    df["discharge_time"],
    format="ISO8601"
)

df = df.sort_values(
    "admission_time"
).reset_index(drop=True)

# Investigate only the original training period.
# Do not use test outcomes for feature selection.

validation_start = df.iloc[
    int(len(df) * 0.70)
]["admission_time"]

df = df.loc[
    (df["admission_time"] < validation_start)
    & (df["discharge_time"] < validation_start)
].copy()

print("Investigation records:", len(df))

# ==========================================
# 2. Investigate numerical correlations
# ==========================================

numeric = df.select_dtypes(
    include=np.number
)

correlations = (
    numeric.corr()[TARGET]
    .drop(TARGET)
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("\nNUMERICAL FEATURE CORRELATIONS")
print(correlations.to_string())

correlations.to_csv(
    EVAL_DIR / "los_correlations.csv",
    header=["correlation"]
)

# Correlation heatmap

plt.figure(figsize=(14, 11))

sns.heatmap(
    numeric.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Training Data Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "14_feature_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 3. Load encounter-specific information
# ==========================================

events = pd.read_csv(
    AGENT_DIR / "discharge_events.csv",
    usecols=[
        "patient_id",
        "encounter_id",
        "event_type",
        "department",
        "care_level"
    ]
)

inpatient = events.loc[
    events["event_type"] == "INPATIENT_STAY",
    [
        "patient_id",
        "encounter_id",
        "department",
        "care_level"
    ]
].copy()

inpatient = inpatient.rename(
    columns={
        "department": "encounter_department"
    }
)

df = df.merge(
    inpatient,
    on=["patient_id", "encounter_id"],
    how="left",
    validate="one_to_one"
)

# ==========================================
# 4. Investigate LOS by care level
# ==========================================

care_summary = df.groupby(
    "care_level",
    dropna=False
)[TARGET].agg([
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max"
])

print("\nLENGTH OF STAY BY CARE LEVEL")
print(care_summary.round(2).to_string())

care_summary.to_csv(
    EVAL_DIR / "los_by_care_level.csv"
)

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="care_level",
    y=TARGET,
    showfliers=False
)

plt.title("Length of Stay by Care Level")
plt.xlabel("Care Level")
plt.ylabel("Length of Stay (Hours)")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "15_los_by_care_level.png",
    dpi=300
)

plt.close()

# ==========================================
# 5. Investigate LOS by department
# ==========================================

department_summary = df.groupby(
    "encounter_department",
    dropna=False
)[TARGET].agg([
    "count",
    "mean",
    "median",
    "std"
]).sort_values(
    "median",
    ascending=False
)

print("\nLENGTH OF STAY BY DEPARTMENT")
print(department_summary.round(2).to_string())

department_summary.to_csv(
    EVAL_DIR / "los_by_department.csv"
)

plt.figure(figsize=(12, 8))

department_summary["median"].sort_values().plot(
    kind="barh"
)

plt.title("Median Length of Stay by Department")
plt.xlabel("Median LOS (Hours)")
plt.ylabel("Department")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "16_los_by_department.png",
    dpi=300
)

plt.close()

# ==========================================
# 6. Investigate ED priority
# ==========================================

admissions = pd.read_csv(
    AGENT_DIR / "admission_events.csv",
    usecols=[
        "patient_id",
        "encounter_id",
        "event_type",
        "priority",
        "arrival_time",
        "triage_time"
    ]
)

arrivals = admissions.loc[
    admissions["event_type"] == "ED_ARRIVAL"
].copy()

arrivals["arrival_time"] = pd.to_datetime(
    arrivals["arrival_time"],
    format="ISO8601",
    errors="coerce"
)

arrivals["triage_time"] = pd.to_datetime(
    arrivals["triage_time"],
    format="ISO8601",
    errors="coerce"
)

# Select the earliest ED arrival per encounter.

arrivals = arrivals.sort_values(
    "arrival_time"
).drop_duplicates(
    ["patient_id", "encounter_id"],
    keep="first"
)

df = df.merge(
    arrivals[
        [
            "patient_id",
            "encounter_id",
            "priority",
            "arrival_time",
            "triage_time"
        ]
    ],
    on=["patient_id", "encounter_id"],
    how="left",
    validate="one_to_one"
)

# Only use priority recorded before admission.

valid_priority = (
    df["arrival_time"].notna()
    & df["triage_time"].notna()
    & (df["arrival_time"] <= df["triage_time"])
    & (df["triage_time"] <= df["admission_time"])
    & df["priority"].isin([1, 2, 3, 4, 5])
)

df.loc[~valid_priority, "priority"] = np.nan

print("\nED PRIORITY COVERAGE")

print(
    "Matched valid priority records:",
    df["priority"].notna().sum()
)

print(
    "Missing priority records:",
    df["priority"].isna().sum()
)

priority_summary = df.groupby(
    "priority",
    dropna=False
)[TARGET].agg([
    "count",
    "mean",
    "median",
    "std"
])

print("\nLENGTH OF STAY BY ED PRIORITY")
print(priority_summary.round(2).to_string())

priority_summary.to_csv(
    EVAL_DIR / "los_by_priority.csv"
)

# ==========================================
# 7. Investigate patient history coverage
# ==========================================

patient_counts = df.groupby(
    "patient_id"
).size()

print("\nPATIENT HISTORY INVESTIGATION")

print("Unique patients:", len(patient_counts))

print(
    "Patients with multiple inpatient encounters:",
    (patient_counts > 1).sum()
)

print(
    "Average encounters per patient:",
    round(patient_counts.mean(), 2)
)

# ==========================================
# 8. Inspect relevant dictionary definitions
# ==========================================

dictionary = pd.read_csv(
    BASE_DIR / "data" / "raw" /
    "DATA_DICTIONARY (1).csv"
)

terms = [
    "care_level",
    "department",
    "priority",
    "triage_time",
    "length_of_stay_hours"
]

definitions = dictionary.loc[
    (dictionary["file_name"] == "longitudinal.csv")
    & dictionary["variable_name"].isin(terms),
    [
        "variable_name",
        "description",
        "synthetic_generation_logic"
    ]
]

print("\nRELEVANT DICTIONARY DEFINITIONS")
print(definitions.to_string(index=False))

print("\nFEATURE INVESTIGATION COMPLETED")
