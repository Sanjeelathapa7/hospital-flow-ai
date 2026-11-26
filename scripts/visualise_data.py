
# CN7050 - Intelligent Systems
# Project: HospitalFlow AI
# Phase 3.3 - Exploratory Data Visualisation

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Define file paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Configure visualisation style
# ==========================================

sns.set_theme(style="whitegrid")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

# ==========================================
# 3. Load data quality reports
# ==========================================

event_distribution = pd.read_csv(
    PROCESSED_DIR / "event_distribution.csv"
)

patient_missing = pd.read_csv(
    PROCESSED_DIR / "patient_missing_values.csv"
)

event_missing = pd.read_csv(
    PROCESSED_DIR / "event_missing_values.csv"
)

quality_summary = pd.read_csv(
    PROCESSED_DIR / "data_quality_summary.csv"
)

metrics = quality_summary.set_index("metric")["value"]

# ==========================================
# 4. Function to save figures
# ==========================================

def save_figure(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ==========================================
# 5. Hospital event distribution
# ==========================================

plt.figure(figsize=(12, 7))

sns.barplot(
    data=event_distribution,
    x="count",
    y="event_type",
    color="steelblue"
)

plt.title(
    "Distribution of Hospital Operational Events",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Number of Events")
plt.ylabel("Event Type")

save_figure("01_hospital_event_distribution.png")


# ==========================================
# 6. Patient dataset missing values
# ==========================================

patient_missing = patient_missing[
    patient_missing["missing_count"] > 0
]

plt.figure(figsize=(12, 7))

if patient_missing.empty:

    plt.text(
        0.5, 0.5,
        "No missing values detected",
        ha="center",
        va="center",
        fontsize=15
    )

    plt.axis("off")

else:

    sns.barplot(
        data=patient_missing,
        x="missing_percentage",
        y="column",
        color="darkorange"
    )

    plt.xlabel("Missing Values (%)")
    plt.ylabel("Column")
    plt.xlim(0, 100)

plt.title(
    "Missing Values in Patient Dataset",
    fontsize=15,
    fontweight="bold"
)

save_figure("02_patient_missing_values.png")


# ==========================================
# 7. Hospital event missing values
# ==========================================

top_missing = event_missing.sort_values(
    "missing_percentage",
    ascending=False
).head(20)

plt.figure(figsize=(12, 9))

sns.barplot(
    data=top_missing,
    x="missing_percentage",
    y="column",
    color="indianred"
)

plt.title(
    "Top 20 Hospital Event Columns by Missingness",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Missing Values (%)")
plt.ylabel("Column")
plt.xlim(0, 100)

save_figure("03_event_missing_values.png")


# ==========================================
# 8. Patient ID availability
# ==========================================

total_events = int(
    metrics["Total hospital events"]
)

missing_ids = int(
    metrics["Events without patient IDs"]
)

available_ids = total_events - missing_ids

plt.figure(figsize=(8, 8))

plt.pie(
    [available_ids, missing_ids],
    labels=[
        "Patient-linked events",
        "Events without patient IDs"
    ],
    autopct="%1.2f%%",
    startangle=90,
    colors=["seagreen", "orange"],
    wedgeprops={"width": 0.45}
)

plt.title(
    "Patient ID Availability",
    fontsize=15,
    fontweight="bold"
)

save_figure("04_patient_id_availability.png")


# ==========================================
# 9. Investigate missing patient IDs
# ==========================================

missing_by_event = pd.Series(dtype="int64")

for chunk in pd.read_csv(
    RAW_DIR / "longitudinal.csv",
    usecols=["patient_id", "event_type"],
    dtype={"patient_id": "string"},
    chunksize=50000
):

    missing_records = chunk[
        chunk["patient_id"].isna()
    ]

    counts = missing_records[
        "event_type"
    ].value_counts(dropna=False)

    missing_by_event = missing_by_event.add(
        counts,
        fill_value=0
    )

missing_by_event = (
    missing_by_event
    .astype(int)
    .sort_values(ascending=False)
)

print("\nMissing patient IDs by event type:")
print(missing_by_event)

missing_by_event.to_csv(
    PROCESSED_DIR / "missing_patient_ids_by_event.csv",
    header=["count"]
)

plt.figure(figsize=(12, 6))

if missing_by_event.empty:

    plt.text(
        0.5, 0.5,
        "No missing patient IDs",
        ha="center",
        va="center"
    )

    plt.axis("off")

else:

    missing_by_event.plot(
        kind="barh",
        color="darkorange"
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Events Without Patient IDs")
    plt.ylabel("Event Type")

plt.title(
    "Missing Patient IDs by Hospital Event Type",
    fontsize=15,
    fontweight="bold"
)

save_figure("05_missing_patient_ids_by_event.png")


# ==========================================
# 10. Completion message
# ==========================================

print("\n" + "=" * 60)
print("DATA VISUALISATION COMPLETED")
print("=" * 60)

print("\nFigures saved in:")
print(FIGURE_DIR)
