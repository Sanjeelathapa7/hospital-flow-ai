
# Project: HospitalFlow AI
# Phase 4.1 - Agent-Specific Dataset Preparation

from pathlib import Path
from collections import Counter
import pandas as pd

# ==========================================
# 1. Define project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "agents"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_FILE = RAW_DIR / "cross_sectional.csv"
EVENT_FILE = RAW_DIR / "longitudinal.csv"

# ==========================================
# 2. Select patient-level features
# ==========================================

patient_features = [
    "patient_id",
    "age",
    "BMI",
    "chronic_condition_count",
    "comorbidity_burden",
    "baseline_medication_count",
    "clinical_complexity_category",
    "baseline_risk_category",
    "prior_ED_visits",
    "prior_hospitalizations",
    "prior_outpatient_visits",
    "prior_surgeries",
    "previous_ICU_admissions",
    "primary_department",
    "primary_condition_category",
    "expected_care_complexity"
]

patients = pd.read_csv(
    PATIENT_FILE,
    usecols=patient_features,
    dtype={"patient_id": "string"}
)

patients.to_csv(
    OUTPUT_DIR / "patient_features.csv",
    index=False
)

print("Patient features saved:", patients.shape)

# ==========================================
# 3. Define agent-specific configurations
# ==========================================

common_columns = [
    "event_id",
    "patient_id",
    "encounter_id",
    "event_date",
    "event_time",
    "event_type",
    "department"
]

agent_configs = {

    "admission_events": {

        "event_types": [
            "ED_ARRIVAL",
            "ED_DISPOSITION"
        ],

        "columns": common_columns + [
            "priority",
            "arrival_time",
            "registration_time",
            "triage_time",
            "wait_time_minutes",
            "queue_time_minutes",
            "disposition",
            "disposition_time",
            "event_status"
        ]
    },

    "bed_events": {

        "event_types": [
            "BED_MANAGEMENT",
            "TRANSFER",
            "INPATIENT_STAY",
            "ICU_STAY"
        ],

        "columns": common_columns + [
            "bed_event_id",
            "bed_id",
            "bed_type",
            "occupancy_status",
            "availability_status",
            "assignment_time",
            "release_time",
            "cleaning_start_time",
            "cleaning_end_time",
            "care_level",
            "ICU_bed_id",
            "ICU_bed_status",
            "from_department",
            "to_department",
            "transfer_status",
            "transfer_request_time",
            "transfer_completion_time",
            "admission_time",
            "discharge_time",
            "event_status"
        ]
    },

    "discharge_events": {

        "event_types": [
            "INPATIENT_STAY",
            "DISCHARGE_PROCESS",
            "ICU_STAY"
        ],

        "columns": common_columns + [
            "admission_time",
            "discharge_time",
            "discharge_decision_time",
            "discharge_ready_time",
            "actual_discharge_time",
            "discharge_delay_minutes",
            "discharge_delay_category",
            "length_of_stay_hours",
            "length_of_stay_days",
            "care_level"
        ]
    },

    "coordination_events": {

        "event_types": [
            "STAFFING_SHIFT",
            "BED_MANAGEMENT",
            "TRANSFER",
            "ED_ARRIVAL",
            "DISCHARGE_PROCESS"
        ],

        "columns": common_columns + [
            "staff_event_id",
            "staff_id",
            "staff_role",
            "shift_type",
            "shift_start",
            "shift_end",
            "scheduled_staff_count",
            "active_staff_count",
            "workload_category",
            "delay_minutes",
            "wait_time_minutes",
            "queue_time_minutes",
            "transfer_status",
            "availability_status",
            "occupancy_status",
            "discharge_delay_minutes",
            "escalation_flag"
        ]
    }
}

# ==========================================
# 4. Validate required columns
# ==========================================

available_columns = pd.read_csv(
    EVENT_FILE,
    nrows=0
).columns.tolist()

required_columns = sorted({
    column
    for config in agent_configs.values()
    for column in config["columns"]
})

missing_columns = set(required_columns) - set(
    available_columns
)

if missing_columns:
    raise ValueError(
        f"Required columns not found: {missing_columns}"
    )

print("\nAll required event columns validated.")

# ==========================================
# 5. Initialise output files
# ==========================================

for dataset_name, config in agent_configs.items():

    output_file = OUTPUT_DIR / f"{dataset_name}.csv"

    # Create a fresh CSV with column headers.
    # This prevents duplicate data when rerunning.
    pd.DataFrame(
        columns=config["columns"]
    ).to_csv(
        output_file,
        index=False
    )

# ==========================================
# 6. Process longitudinal data in chunks
# ==========================================

record_counts = Counter()

total_processed = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        EVENT_FILE,
        usecols=required_columns,
        dtype="string",
        chunksize=50000
    ),
    start=1
):

    total_processed += len(chunk)

    # Create datasets for individual agents
    for dataset_name, config in agent_configs.items():

        selected = chunk.loc[
            chunk["event_type"].isin(
                config["event_types"]
            ),
            config["columns"]
        ].copy()

        if not selected.empty:

            selected.to_csv(
                OUTPUT_DIR / f"{dataset_name}.csv",
                mode="a",
                header=False,
                index=False
            )

            record_counts[dataset_name] += len(selected)

    print(
        f"Chunk {chunk_number} completed | "
        f"Records processed: {total_processed:,}"
    )

# ==========================================
# 7. Generate preparation summary
# ==========================================

summary_records = [
    {
        "dataset": "patient_features",
        "rows": len(patients),
        "columns": len(patient_features)
    }
]

for dataset_name, config in agent_configs.items():

    summary_records.append({
        "dataset": dataset_name,
        "rows": record_counts[dataset_name],
        "columns": len(config["columns"])
    })

summary = pd.DataFrame(summary_records)

summary.to_csv(
    OUTPUT_DIR / "preparation_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("AGENT DATASET PREPARATION SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

print("\nOriginal hospital events processed:", total_processed)

print("\nOutput directory:", OUTPUT_DIR)

print("\nDATASET PREPARATION COMPLETED")
