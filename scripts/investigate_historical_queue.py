import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_loader import (
    load_admissions,
    load_bed_events,
)

OUTPUT_DIR = ROOT / "outputs" / "system_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["patient_id", "encounter_id"]

print("=" * 65)
print("HOSPITALFLOW AI - HISTORICAL QUEUE INVESTIGATION")
print("=" * 65)

admissions = load_admissions().copy()
beds = load_bed_events().copy()

# 1. Identify admission decisions

dispositions = admissions[
    admissions["event_type"].eq("ED_DISPOSITION")
].copy()

dispositions["disposition"] = (
    dispositions["disposition"]
    .astype("string")
    .str.strip()
    .str.lower()
)

requests = dispositions[
    dispositions["disposition"].isin(
        ["inpatient admission", "icu admission"]
    )
].copy()

requests = requests.dropna(
    subset=KEYS + ["disposition_time"]
)

requests = (
    requests.sort_values("disposition_time")
    .drop_duplicates(KEYS, keep="last")
)

print("\n1. ADMISSION REQUESTS")
print("Total requests:", len(requests))
print(requests["disposition"].value_counts())

# 2. Identify first recorded bed assignment

assignments = beds[
    beds["event_type"].eq("BED_MANAGEMENT")
    & beds["occupancy_status"].eq("Occupied")
].copy()

assignments = assignments.dropna(
    subset=KEYS + ["assignment_time", "bed_id"]
)

first_assignment = (
    assignments.groupby(KEYS, as_index=False)
    ["assignment_time"]
    .min()
)

# 3. Identify first recorded inpatient admission

stays = beds[
    beds["event_type"].isin(
        ["INPATIENT_STAY", "ICU_STAY"]
    )
].copy()

stays = stays.dropna(
    subset=KEYS + ["admission_time"]
)

first_admission = (
    stays.groupby(KEYS, as_index=False)
    ["admission_time"]
    .min()
)

# 4. Combine admission and bed records

analysis = requests[
    KEYS + ["disposition", "disposition_time"]
].merge(
    first_assignment,
    on=KEYS,
    how="left",
    validate="one_to_one",
)

analysis = analysis.merge(
    first_admission,
    on=KEYS,
    how="left",
    validate="one_to_one",
)

analysis["queue_exit_time"] = analysis[
    ["assignment_time", "admission_time"]
].min(axis=1)

analysis["queue_duration_minutes"] = (
    analysis["queue_exit_time"]
    - analysis["disposition_time"]
).dt.total_seconds() / 60

# 5. Classify timestamp relationships

duration = analysis["queue_duration_minutes"]

summary = {
    "total_admission_requests": len(analysis),
    "positive_waiting_intervals": int(
        (duration > 0).sum()
    ),
    "zero_waiting_intervals": int(
        (duration == 0).sum()
    ),
    "negative_waiting_intervals": int(
        (duration < 0).sum()
    ),
    "missing_queue_exit_time": int(
        duration.isna().sum()
    ),
}

print("\n2. TIMESTAMP RELATIONSHIPS")

for key, value in summary.items():
    print(f"{key}: {value}")

# 6. Analyse positive waiting intervals

positive = analysis[duration > 0].copy()

print("\n3. POSITIVE WAITING INTERVALS")

if positive.empty:
    print("No positive completed waiting intervals found.")
else:
    print(
        positive["queue_duration_minutes"]
        .describe()
    )

# 7. Compare selected historical snapshots

snapshot_times = [
    "2031-10-03 12:00:00",
    "2031-11-06 08:00:00",
    "2031-11-06 12:00:00",
    "2031-11-07 12:00:00",
    "2031-11-08 12:00:00",
    "2031-11-09 12:00:00",
]

snapshot_results = []

print("\n4. HISTORICAL SNAPSHOT ANALYSIS")

for value in snapshot_times:

    now = pd.Timestamp(value)

    waiting = analysis[
        (analysis["disposition_time"] <= now)
        & (
            (analysis["queue_exit_time"] > now)
            | analysis["queue_exit_time"].isna()
        )
    ]

    result = {
        "simulation_time": value,
        "candidate_waiting_patients": len(waiting),
        "icu_requests": int(
            waiting["disposition"]
            .eq("icu admission")
            .sum()
        ),
        "inpatient_requests": int(
            waiting["disposition"]
            .eq("inpatient admission")
            .sum()
        ),
    }

    snapshot_results.append(result)

snapshot_df = pd.DataFrame(snapshot_results)

print(snapshot_df.to_string(index=False))

# 8. Save evaluation evidence

analysis.to_csv(
    OUTPUT_DIR / "historical_queue_analysis.csv",
    index=False,
)

pd.DataFrame([summary]).to_csv(
    OUTPUT_DIR / "historical_queue_summary.csv",
    index=False,
)

snapshot_df.to_csv(
    OUTPUT_DIR / "historical_snapshot_comparison.csv",
    index=False,
)

print("\nResults saved to:", OUTPUT_DIR)
print("\nINVESTIGATION COMPLETED")