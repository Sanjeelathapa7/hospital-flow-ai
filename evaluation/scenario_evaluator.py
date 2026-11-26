from copy import deepcopy
from time import perf_counter
from statistics import median

import numpy as np
import pandas as pd

from agents.bed_allocation_agent import BedAllocationAgent
from evaluation.fcfs_baseline import FCFSBaseline
from evaluation.performance_metrics import calculate_performance


SIMULATION_TIME = pd.Timestamp("2031-11-07 12:00:00")
RANDOM_STATE = 42


def make_patient(
    index,
    priority,
    waiting_minutes,
    bed_type="medical ward",
):
    arrival = SIMULATION_TIME - pd.Timedelta(
        minutes=waiting_minutes
    )

    return {
        "patient_id": f"EVAL-P{index:05d}",
        "encounter_id": f"EVAL-ENC{index:05d}",
        "priority": int(priority),
        "arrival_time": arrival.isoformat(),
        "waiting_time_minutes": float(waiting_minutes),
        "requested_care": (
            "ICU" if bed_type == "ICU"
            else "inpatient"
        ),
        "requested_bed_type": bed_type,
        "requirement_source": "Controlled evaluation scenario",
    }


def make_bed(index, bed_type, status):
    return {
        "bed_id": f"EVAL-B{index:05d}",
        "bed_type": bed_type,
        "status": status,
        "allocatable": status == "Available",
    }


def build_inventory(
    total_beds,
    available_medical,
    available_icu=0,
    occupied_icu=0,
):
    beds = []

    for _ in range(available_medical):
        beds.append(
            make_bed(
                len(beds) + 1,
                "medical ward",
                "Available",
            )
        )

    for _ in range(available_icu):
        beds.append(
            make_bed(
                len(beds) + 1,
                "ICU",
                "Available",
            )
        )

    for _ in range(occupied_icu):
        beds.append(
            make_bed(
                len(beds) + 1,
                "ICU",
                "Occupied",
            )
        )

    if len(beds) > total_beds:
        raise ValueError("Invalid bed capacity")

    while len(beds) < total_beds:
        beds.append(
            make_bed(
                len(beds) + 1,
                "medical ward",
                "Occupied",
            )
        )

    return beds


def generate_patients(
    number,
    seed=RANDOM_STATE,
    icu_probability=0.15,
):
    rng = np.random.default_rng(seed)

    patients = []

    for index in range(1, number + 1):

        priority = rng.choice(
            [1, 2, 3, 4, 5],
            p=[0.10, 0.20, 0.35, 0.25, 0.10],
        )

        waiting_minutes = rng.integers(15, 361)

        bed_type = (
            "ICU"
            if rng.random() < icu_probability
            else "medical ward"
        )

        patients.append(
            make_patient(
                index,
                priority,
                waiting_minutes,
                bed_type,
            )
        )

    return patients


def build_scenarios():

    return {
        "Normal Operations": {
            "patients": [
                make_patient(1, 3, 120),
                make_patient(2, 1, 60, "ICU"),
                make_patient(3, 2, 90),
            ],
            "beds": build_inventory(
                total_beds=10,
                available_medical=2,
                available_icu=1,
            ),
        },

        "Hospital Bed Shortage": {
            "patients": [
                make_patient(1, 4, 240),
                make_patient(2, 3, 180),
                make_patient(3, 1, 120),
                make_patient(4, 2, 90),
                make_patient(5, 5, 60),
            ],
            "beds": build_inventory(
                total_beds=20,
                available_medical=2,
            ),
        },

        "ICU Capacity Crisis": {
            "patients": [
                make_patient(1, 1, 240, "ICU"),
                make_patient(2, 1, 180, "ICU"),
                make_patient(3, 2, 120, "ICU"),
                make_patient(4, 3, 90),
                make_patient(5, 4, 60),
            ],
            "beds": build_inventory(
                total_beds=20,
                available_medical=1,
                available_icu=0,
                occupied_icu=5,
            ),
        },

        "Severe Overcrowding": {
            "patients": generate_patients(
                20,
                seed=42,
            ),
            "beds": build_inventory(
                total_beds=50,
                available_medical=4,
                available_icu=1,
                occupied_icu=5,
            ),
        },

        "Full Capacity": {
            "patients": generate_patients(
                10,
                seed=43,
            ),
            "beds": build_inventory(
                total_beds=20,
                available_medical=0,
                available_icu=0,
                occupied_icu=5,
            ),
        },

        "High Patient Demand": {
            "patients": generate_patients(
                100,
                seed=44,
            ),
            "beds": build_inventory(
                total_beds=200,
                available_medical=25,
                available_icu=5,
                occupied_icu=15,
            ),
        },
    }


def run_fcfs(patients, beds):

    return FCFSBaseline().allocate(
        patients,
        beds,
    )


def run_intelligent(patients, beds):

    agent = BedAllocationAgent()

    available_beds = [
        bed for bed in beds
        if bed["status"] == "Available"
        and bed.get("allocatable") is True
    ]

    return agent.allocate(
        patients,
        available_beds,
    )


def evaluate_method(
    method_name,
    method_function,
    patients,
    beds,
    repetitions=10,
):

    if repetitions < 1:
        raise ValueError(
            "Repetitions must be at least 1"
        )

    # Warm-up run
    method_function(
        deepcopy(patients),
        deepcopy(beds),
    )

    timings = []

    for _ in range(repetitions):

        patient_copy = deepcopy(patients)
        bed_copy = deepcopy(beds)

        start = perf_counter()

        recommendations, unallocated = method_function(
            patient_copy,
            bed_copy,
        )

        timings.append(
            perf_counter() - start
        )

    metrics = calculate_performance(
        method=method_name,
        patients=patients,
        beds=beds,
        recommendations=recommendations,
        unallocated_patients=unallocated,
        execution_time=median(timings),
    )

    metrics["timing_repetitions"] = repetitions

    return metrics, recommendations, unallocated


def evaluate_scenario(
    scenario_name,
    patients,
    beds,
    repetitions=10,
):

    methods = {
        "FCFS": run_fcfs,
        "HospitalFlow AI": run_intelligent,
    }

    results = []
    allocation_details = []
    waiting_details = []

    for method_name, method_function in methods.items():

        metrics, allocated, waiting = evaluate_method(
            method_name,
            method_function,
            patients,
            beds,
            repetitions,
        )

        metrics["scenario"] = scenario_name

        results.append(metrics)

        for allocation in allocated:
            allocation_details.append({
                "scenario": scenario_name,
                "method": method_name,
                **allocation,
            })

        for patient in waiting:
            waiting_details.append({
                "scenario": scenario_name,
                "method": method_name,
                **patient,
            })

    return (
        results,
        allocation_details,
        waiting_details,
    )