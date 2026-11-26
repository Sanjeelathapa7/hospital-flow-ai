
# HospitalFlow AI
# Hospital Environment Investigation

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from core.config import (
    EXPECTED_TOTAL_BEDS,
    DEFAULT_SIMULATION_TIME
)

from core.data_loader import (
    get_dataset_summary,
    load_bed_events,
    load_admissions,
    load_coordination_events
)

# ==========================================
# 1. Dataset summary
# ==========================================

print("=" * 60)
print("HOSPITAL ENVIRONMENT INSPECTION")
print("=" * 60)

print(
    get_dataset_summary().to_string(index=False)
)

# ==========================================
# 2. Load bed management records
# ==========================================

events = load_bed_events()

beds = events.loc[
    events["event_type"] == "BED_MANAGEMENT"
].copy()

print("\nBED MANAGEMENT RECORDS")

print("Total records:", len(beds))

print(
    "Missing physical bed IDs:",
    beds["bed_id"].isna().sum()
)

print(
    "Unique physical bed IDs:",
    beds["bed_id"].nunique()
)

print(
    "Expected hospital capacity:",
    EXPECTED_TOTAL_BEDS
)

# ==========================================
# 3. Investigate physical bed types
# ==========================================

inventory = beds[
    ["bed_id", "bed_type"]
].dropna().drop_duplicates()

print("\nPHYSICAL BED INVENTORY")

print(
    inventory.groupby("bed_type")["bed_id"]
    .nunique()
    .sort_values(ascending=False)
    .to_string()
)

type_conflicts = (
    inventory.groupby("bed_id")["bed_type"]
    .nunique()
)

print(
    "\nBeds with conflicting bed types:",
    (type_conflicts > 1).sum()
)

# ==========================================
# 4. Inspect department consistency
# ==========================================

department_counts = (
    beds.dropna(subset=["bed_id", "department"])
    .groupby("bed_id")["department"]
    .nunique()
)

print(
    "Beds associated with multiple departments:",
    (department_counts > 1).sum()
)

# ==========================================
# 5. Inspect occupancy states
# ==========================================

print("\nOCCUPANCY STATUS")

print(
    beds["occupancy_status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nAVAILABILITY STATUS")

print(
    beds["availability_status"]
    .value_counts(dropna=False)
    .to_string()
)

# ==========================================
# 6. Inspect timestamp coverage
# ==========================================

time_columns = [
    "assignment_time",
    "release_time",
    "cleaning_start_time",
    "cleaning_end_time"
]

print("\nTIMESTAMP MISSINGNESS BY STATUS")

print(
    beds.groupby(
        "occupancy_status",
        dropna=False
    )[time_columns]
    .agg(lambda x: x.isna().sum())
    .to_string()
)

occupied = beds.loc[
    beds["occupancy_status"] == "Occupied"
].copy()

valid_intervals = occupied.dropna(
    subset=[
        "bed_id",
        "assignment_time",
        "release_time"
    ]
).copy()

invalid_intervals = (
    valid_intervals["release_time"]
    <= valid_intervals["assignment_time"]
)

print(
    "\nInvalid occupied intervals:",
    invalid_intervals.sum()
)

valid_intervals = valid_intervals.loc[
    ~invalid_intervals
].copy()

# ==========================================
# 7. Check overlapping occupancy intervals
# ==========================================

valid_intervals = valid_intervals.sort_values(
    ["bed_id", "assignment_time", "release_time"]
)

running_max = valid_intervals.groupby(
    "bed_id"
)["release_time"].cummax()

previous_max = running_max.groupby(
    valid_intervals["bed_id"]
).shift(1)

overlaps = (
    valid_intervals["assignment_time"]
    < previous_max
)

print(
    "Overlapping occupied intervals:",
    overlaps.sum()
)

# ==========================================
# 8. Inspect simulation-time occupancy
# ==========================================

simulation_time = pd.Timestamp(
    DEFAULT_SIMULATION_TIME
)

active_occupancy = valid_intervals.loc[
    (
        valid_intervals["assignment_time"]
        <= simulation_time
    )
    & (
        valid_intervals["release_time"]
        > simulation_time
    )
]

active_cleaning = beds.loc[
    (
        beds["cleaning_start_time"]
        <= simulation_time
    )
    & (
        beds["cleaning_end_time"]
        > simulation_time
    )
    & beds["bed_id"].notna()
]

occupied_ids = set(
    active_occupancy["bed_id"].dropna()
)

cleaning_ids = set(
    active_cleaning["bed_id"].dropna()
)

all_bed_ids = set(
    inventory["bed_id"]
)

unobserved_active_ids = (
    all_bed_ids
    - occupied_ids
    - cleaning_ids
)

print("\nSIMULATION SNAPSHOT")

print("Simulation time:", simulation_time)

print(
    "Observed physical beds:",
    len(all_bed_ids)
)

print(
    "Occupied beds:",
    len(occupied_ids)
)

print(
    "Beds in cleaning intervals:",
    len(cleaning_ids)
)

print(
    "Occupancy-cleaning conflicts:",
    len(occupied_ids & cleaning_ids)
)

print(
    "Beds outside observed active intervals:",
    len(unobserved_active_ids)
)

# ==========================================
# 9. Inspect ICU bed identifiers
# ==========================================

icu_events = events.loc[
    events["event_type"] == "ICU_STAY"
]

icu_ids = set(
    icu_events["ICU_bed_id"]
    .dropna()
    .astype(str)
)

print("\nICU BED INVESTIGATION")

print("Unique ICU bed IDs:", len(icu_ids))

print(
    "ICU IDs matching physical inventory:",
    len(icu_ids & all_bed_ids)
)

# ==========================================
# 10. Inspect admission event coverage
# ==========================================

admissions = load_admissions()

print("\nADMISSION EVENT DISTRIBUTION")

print(
    admissions["event_type"]
    .value_counts()
    .to_string()
)

print("\nADMISSION TIMESTAMP MISSINGNESS")

print(
    admissions.groupby("event_type")[
        [
            "arrival_time",
            "triage_time",
            "disposition_time"
        ]
    ].agg(
        lambda x: x.isna().sum()
    ).to_string()
)

print("\nADMISSION DISPOSITIONS")

print(
    admissions["disposition"]
    .value_counts(dropna=False)
    .to_string()
)

# ==========================================
# 11. Inspect staffing information
# ==========================================

coordination = load_coordination_events()

staffing = coordination.loc[
    coordination["event_type"] == "STAFFING_SHIFT"
].copy()

active_staffing = staffing.loc[
    (staffing["shift_start"] <= simulation_time)
    & (staffing["shift_end"] > simulation_time)
]

print("\nSTAFFING INVESTIGATION")

print("Staffing records:", len(staffing))

print(
    "Active staffing records:",
    len(active_staffing)
)

print("\nACTIVE STAFFING BY DEPARTMENT")

print(
    active_staffing.groupby("department")
    .size()
    .to_string()
)

print("\nACTIVE STAFFING WORKLOAD")

print(
    active_staffing["workload_category"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nENVIRONMENT INSPECTION COMPLETED")
