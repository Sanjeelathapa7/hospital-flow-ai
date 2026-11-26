
# HospitalFlow AI
# Hospital Data Loading Module

from functools import lru_cache
import pandas as pd

from core.config import (
    PATIENT_FILE,
    ADMISSION_FILE,
    BED_FILE,
    DISCHARGE_FILE,
    COORDINATION_FILE
)


def validate_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )


def parse_timestamps(df, columns):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                format="ISO8601",
                errors="coerce"
            )

    return df


@lru_cache(maxsize=1)
def load_patients():

    validate_file(PATIENT_FILE)

    return pd.read_csv(
        PATIENT_FILE,
        dtype={"patient_id": "string"}
    )


@lru_cache(maxsize=1)
def load_admissions():

    validate_file(ADMISSION_FILE)

    df = pd.read_csv(
        ADMISSION_FILE,
        dtype={
            "patient_id": "string",
            "encounter_id": "string"
        }
    )

    df = parse_timestamps(
        df,
        [
            "arrival_time",
            "registration_time",
            "triage_time",
            "disposition_time"
        ]
    )

    df["priority"] = pd.to_numeric(
        df["priority"],
        errors="coerce"
    )

    return df


@lru_cache(maxsize=1)
def load_bed_events():

    validate_file(BED_FILE)

    df = pd.read_csv(
        BED_FILE,
        dtype={
            "patient_id": "string",
            "encounter_id": "string",
            "bed_id": "string",
            "bed_event_id": "string"
        }
    )

    df = parse_timestamps(
        df,
        [
            "assignment_time",
            "release_time",
            "cleaning_start_time",
            "cleaning_end_time",
            "admission_time",
            "discharge_time",
            "transfer_request_time",
            "transfer_completion_time"
        ]
    )

    return df


@lru_cache(maxsize=1)
def load_discharge_events():

    validate_file(DISCHARGE_FILE)

    df = pd.read_csv(
        DISCHARGE_FILE,
        dtype={
            "patient_id": "string",
            "encounter_id": "string"
        }
    )

    df = parse_timestamps(
        df,
        [
            "admission_time",
            "discharge_time",
            "discharge_decision_time",
            "discharge_ready_time",
            "actual_discharge_time"
        ]
    )

    df["length_of_stay_hours"] = pd.to_numeric(
        df["length_of_stay_hours"],
        errors="coerce"
    )

    return df


@lru_cache(maxsize=1)
def load_coordination_events():

    validate_file(COORDINATION_FILE)

    df = pd.read_csv(
        COORDINATION_FILE,
        low_memory=False
    )

    df = parse_timestamps(
        df,
        ["shift_start", "shift_end"]
    )

    return df


def get_dataset_summary():

    datasets = {
        "Patients": load_patients(),
        "Admission events": load_admissions(),
        "Bed events": load_bed_events(),
        "Discharge events": load_discharge_events(),
        "Coordination events": load_coordination_events()
    }

    return pd.DataFrame([
        {
            "Dataset": name,
            "Records": len(df),
            "Columns": len(df.columns)
        }
        for name, df in datasets.items()
    ])
