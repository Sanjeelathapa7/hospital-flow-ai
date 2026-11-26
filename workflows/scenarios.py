
# HospitalFlow AI
# Controlled Hospital Scenarios

from uuid import uuid4

import pandas as pd

from core.state import create_initial_state
from core.data_loader import load_patients


SCENARIOS = {
    "normal": "Normal Hospital Operations",
    "bed_shortage": "Hospital Bed Shortage",
    "icu_crisis": "ICU Capacity Crisis"
}


def make_bed(bed_id, bed_type, status):

    return {
        "bed_id": bed_id,
        "bed_type": bed_type,
        "status": status,
        "allocatable": status == "Available"
    }


def create_scenario(
    scenario_name="bed_shortage",
    simulation_time="2031-11-07 12:00:00"
):

    if scenario_name not in SCENARIOS:

        raise ValueError(
            f"Unknown scenario: {scenario_name}"
        )

    state = create_initial_state(
        simulation_time=simulation_time,
        simulation_id=f"SIM-{uuid4().hex[:10]}"
    )

    state["scenario_mode"] = scenario_name

    now = pd.Timestamp(simulation_time)

    if scenario_name == "normal":

        priorities = [1, 2, 3]

        waits = [30, 60, 90]

        requirements = [
            "ICU",
            "medical ward",
            "surgical ward"
        ]

        beds = [

            make_bed(
                "SIM-ICU-001",
                "ICU",
                "Available"
            ),

            make_bed(
                "SIM-MEDW-001",
                "medical ward",
                "Available"
            ),

            make_bed(
                "SIM-SURG-001",
                "surgical ward",
                "Available"
            )
        ]

        occupied_count = 7

        workload = "Normal"

    elif scenario_name == "bed_shortage":

        priorities = [1, 2, 3, 4, 5]

        waits = [90, 150, 260, 180, 60]

        requirements = [
            "ICU",
            "medical ward",
            "medical ward",
            "surgical ward",
            "medical ward"
        ]

        beds = [

            make_bed(
                "SIM-ICU-001",
                "ICU",
                "Available"
            ),

            make_bed(
                "SIM-MEDW-001",
                "medical ward",
                "Available"
            )
        ]

        occupied_count = 18

        workload = "High"

    else:

        priorities = [1, 1, 2, 3, 4]

        waits = [260, 180, 150, 90, 60]

        requirements = [
            "ICU",
            "ICU",
            "ICU",
            "medical ward",
            "medical ward"
        ]

        beds = [

            make_bed(
                "SIM-ICU-001",
                "ICU",
                "Occupied"
            ),

            make_bed(
                "SIM-ICU-002",
                "ICU",
                "Occupied"
            ),

            make_bed(
                "SIM-MEDW-001",
                "medical ward",
                "Available"
            )
        ]

        occupied_count = 17

        workload = "Critical"

    for index in range(occupied_count):

        beds.append(

            make_bed(
                f"SIM-OCC-{index + 1:03d}",
                "medical ward",
                "Occupied"
            )
        )

    patient_ids = (
        load_patients()
        .head(len(priorities))["patient_id"]
        .astype(str)
        .tolist()
    )

    patients = []

    for index, patient_id in enumerate(patient_ids):

        bed_type = requirements[index]

        wait = waits[index]

        disposition_time = (
            now - pd.Timedelta(minutes=wait)
        )

        arrival_time = (
            disposition_time
            - pd.Timedelta(minutes=30)
        )

        patients.append({

            "patient_id": patient_id,

            "encounter_id":
                f"SIM-ENC-{index + 1:03d}",

            "priority": priorities[index],

            "waiting_time_minutes": wait,

            "ed_elapsed_minutes": wait + 30,

            "arrival_time":
                arrival_time.isoformat(),

            "triage_time":
                (
                    arrival_time
                    + pd.Timedelta(minutes=10)
                ).isoformat(),

            "disposition_time":
                disposition_time.isoformat(),

            "disposition":
                "icu admission"
                if bed_type == "ICU"
                else "inpatient admission",

            "requested_bed_type": bed_type,

            "requirement_source":
                "Controlled synthetic scenario"
        })

    state["scenario_patients"] = patients

    state["beds"] = beds

    state["staffing"] = [

        {
            "department": "Emergency Department",
            "workload_category": workload
        },

        {
            "department": "ICU",
            "workload_category": workload
        }
    ]

    return state
