
# HospitalFlow AI
# Phase 3.2 - Data Quality Analysis

from pathlib import Path
from collections import Counter
import pandas as pd

# ==========================================
# 1. Define file paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_FILE = RAW_DIR / "cross_sectional.csv"
EVENT_FILE = RAW_DIR / "longitudinal.csv"

# ==========================================
# 2. Patient dataset quality analysis
# ==========================================

print("\n" + "=" * 60)
print("PATIENT DATA QUALITY ANALYSIS")
print("=" * 60)

patients = pd.read_csv(
    PATIENT_FILE,
    dtype={"patient_id": "string"}
)

print("\nTotal patient records:", len(patients))
print("Total columns:", len(patients.columns))
print("Duplicate rows:", patients.duplicated().sum())

# Check patient IDs

valid_patient_ids = set()

if "patient_id" in patients.columns:

    valid_patient_ids = set(
        patients["patient_id"].dropna()
    )

    print("Unique patient IDs:", len(valid_patient_ids))

    print(
        "Missing patient IDs:",
        patients["patient_id"].isna().sum()
    )

    print(
        "Duplicate non-null patient IDs:",
        patients["patient_id"].dropna().duplicated().sum()
    )

# Missing value analysis

patient_missing = pd.DataFrame({
    "column": patients.columns,
    "missing_count": patients.isna().sum().values,
    "missing_percentage": (
        patients.isna().mean().values * 100
    ).round(2)
})

patient_missing = patient_missing.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nPatient missing values:")
print(patient_missing.to_string(index=False))

patient_missing.to_csv(
    OUTPUT_DIR / "patient_missing_values.csv",
    index=False
)

# ==========================================
# 3. Initialise event analysis variables
# ==========================================

print("\n" + "=" * 60)
print("HOSPITAL EVENT DATA QUALITY ANALYSIS")
print("=" * 60)

total_events = 0
missing_counts = Counter()
event_counts = Counter()

seen_event_ids = set()
duplicate_event_ids = 0
missing_event_ids = 0

missing_event_patient_ids = 0
invalid_patient_references = 0

chunk_size = 50000

# ==========================================
# 4. Process hospital events in chunks
# ==========================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        EVENT_FILE,
        chunksize=chunk_size,
        dtype={
            "patient_id": "string",
            "event_id": "string"
        },
        low_memory=False
    ),
    start=1
):

    total_events += len(chunk)

    # Count missing values
    missing_counts.update(
        chunk.isna().sum().to_dict()
    )

    # Count hospital event categories
    if "event_type" in chunk.columns:

        event_counts.update(
            chunk["event_type"]
            .value_counts(dropna=False)
            .to_dict()
        )

    # Check duplicate event IDs
    if "event_id" in chunk.columns:

        missing_event_ids += (
            chunk["event_id"].isna().sum()
        )

        for event_id in chunk["event_id"].dropna():

            if event_id in seen_event_ids:
                duplicate_event_ids += 1
            else:
                seen_event_ids.add(event_id)

    # Validate patient references
    if "patient_id" in chunk.columns:

        missing_event_patient_ids += (
            chunk["patient_id"].isna().sum()
        )

        event_patient_ids = (
            chunk["patient_id"].dropna()
        )

        if valid_patient_ids:

            invalid_patient_references += (
                ~event_patient_ids.isin(valid_patient_ids)
            ).sum()

    print(
        f"Chunk {chunk_number} completed | "
        f"Total records processed: {total_events:,}"
    )

# ==========================================
# 5. Generate event quality summary
# ==========================================

print("\n" + "=" * 60)
print("FINAL EVENT QUALITY SUMMARY")
print("=" * 60)

print("Total hospital events:", total_events)
print("Unique event IDs:", len(seen_event_ids))
print("Duplicate event IDs:", duplicate_event_ids)
print("Missing event IDs:", missing_event_ids)

print(
    "Events without patient IDs:",
    missing_event_patient_ids
)

print(
    "Invalid patient references:",
    invalid_patient_references
)

# ==========================================
# 6. Missing value report
# ==========================================

event_missing = pd.DataFrame({
    "column": list(missing_counts.keys()),
    "missing_count": list(missing_counts.values())
})

event_missing["missing_percentage"] = (
    event_missing["missing_count"] / total_events * 100
).round(2)

event_missing = event_missing.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nEvent missing values:")
print(event_missing.to_string(index=False))

event_missing.to_csv(
    OUTPUT_DIR / "event_missing_values.csv",
    index=False
)

# ==========================================
# 7. Event distribution report
# ==========================================

event_distribution = pd.DataFrame(
    event_counts.most_common(),
    columns=["event_type", "count"]
)

event_distribution["percentage"] = (
    event_distribution["count"] / total_events * 100
).round(2)

print("\nHospital event distribution:")
print(event_distribution.to_string(index=False))

event_distribution.to_csv(
    OUTPUT_DIR / "event_distribution.csv",
    index=False
)

# ==========================================
# 8. Save overall quality summary
# ==========================================

quality_summary = pd.DataFrame({
    "metric": [
        "Total patient records",
        "Unique patient IDs",
        "Duplicate patient rows",
        "Total hospital events",
        "Unique event IDs",
        "Duplicate event IDs",
        "Missing event IDs",
        "Events without patient IDs",
        "Invalid patient references"
    ],
    "value": [
        len(patients),
        len(valid_patient_ids),
        patients.duplicated().sum(),
        total_events,
        len(seen_event_ids),
        duplicate_event_ids,
        missing_event_ids,
        missing_event_patient_ids,
        invalid_patient_references
    ]
})

quality_summary.to_csv(
    OUTPUT_DIR / "data_quality_summary.csv",
    index=False
)

print("\nReports saved in:", OUTPUT_DIR)
print("\nDATA QUALITY ANALYSIS COMPLETED")
