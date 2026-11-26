import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.scenario_evaluator import (
    generate_patients,
    build_inventory,
    evaluate_scenario,
)


OUTPUT_DIR = ROOT / "outputs" / "system_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_LEVELS = [50, 100, 250, 500, 1000]

AVAILABILITY_LEVELS = [0.05, 0.15, 0.30]

RANDOM_SEEDS = list(range(42, 52))

TIMING_REPETITIONS = 5

SAFETY_COLUMNS = [
    "compatibility_violations",
    "unavailable_bed_violations",
    "invalid_reference_violations",
    "duplicate_bed_allocations",
    "duplicate_encounter_allocations",
]


def build_stress_scenario(
    patient_count,
    availability,
    seed,
):

    patients = generate_patients(
        number=patient_count,
        seed=seed,
        icu_probability=0.15,
    )

    # Keep the bed-to-patient ratio constant
    # across demand levels.

    total_beds = patient_count * 2

    total_icu_beds = int(total_beds * 0.15)

    total_available = int(
        total_beds * availability
    )

    available_icu = min(
        total_icu_beds,
        int(total_available * 0.15),
    )

    available_medical = (
        total_available - available_icu
    )

    occupied_icu = (
        total_icu_beds - available_icu
    )

    beds = build_inventory(
        total_beds=total_beds,
        available_medical=available_medical,
        available_icu=available_icu,
        occupied_icu=occupied_icu,
    )

    return patients, beds


def run_stress_tests():

    results = []

    total_scenarios = (
        len(PATIENT_LEVELS)
        * len(AVAILABILITY_LEVELS)
        * len(RANDOM_SEEDS)
    )

    completed = 0

    print("=" * 75)
    print("HOSPITALFLOW AI - RANDOMISED STRESS TESTING")
    print("=" * 75)

    print("Total scenarios:", total_scenarios)

    for patient_count in PATIENT_LEVELS:

        for availability in AVAILABILITY_LEVELS:

            for seed in RANDOM_SEEDS:

                patients, beds = build_stress_scenario(
                    patient_count,
                    availability,
                    seed,
                )

                scenario_name = (
                    f"Demand-{patient_count}"
                    f"-Availability-{int(availability * 100)}"
                    f"-Seed-{seed}"
                )

                metrics, _, _ = evaluate_scenario(
                    scenario_name,
                    patients,
                    beds,
                    repetitions=TIMING_REPETITIONS,
                )

                for record in metrics:

                    record["patient_demand"] = (
                        patient_count
                    )

                    record["bed_availability_percent"] = (
                        availability * 100
                    )

                    record["random_seed"] = seed

                    results.append(record)

                completed += 1

                print(
                    f"[{completed}/{total_scenarios}] "
                    f"{scenario_name} completed"
                )

    return pd.DataFrame(results)


def summarise_results(results):

    summary = (
        results.groupby(
            [
                "method",
                "patient_demand",
                "bed_availability_percent",
            ],
            as_index=False,
        )
        .agg(
            trials=("random_seed", "count"),

            mean_allocation_rate=(
                "allocation_rate_percent", "mean"
            ),

            mean_high_priority_rate=(
                "high_priority_allocation_rate", "mean"
            ),

            std_high_priority_rate=(
                "high_priority_allocation_rate", "std"
            ),

            mean_remaining_patients=(
                "remaining_patients", "mean"
            ),

            mean_projected_occupancy=(
                "projected_occupancy_percent", "mean"
            ),

            mean_execution_time_ms=(
                "execution_time_ms", "mean"
            ),

            std_execution_time_ms=(
                "execution_time_ms", "std"
            ),

            max_execution_time_ms=(
                "execution_time_ms", "max"
            ),

            compatibility_violations=(
                "compatibility_violations", "sum"
            ),

            duplicate_bed_allocations=(
                "duplicate_bed_allocations", "sum"
            ),
        )
    )

    return summary.round(4)


def compare_paired_results(results):

    keys = [
        "scenario",
        "patient_demand",
        "bed_availability_percent",
        "random_seed",
    ]

    metrics = [
        "allocation_rate_percent",
        "high_priority_allocation_rate",
        "high_priority_allocated",
        "execution_time_ms",
    ]

    fcfs = results[
        results["method"].eq("FCFS")
    ][keys + metrics]

    intelligent = results[
        results["method"].eq("HospitalFlow AI")
    ][keys + metrics]

    paired = fcfs.merge(
        intelligent,
        on=keys,
        suffixes=("_fcfs", "_ai"),
        validate="one_to_one",
    )

    paired["urgent_allocation_improvement_pp"] = (
        paired["high_priority_allocation_rate_ai"]
        - paired["high_priority_allocation_rate_fcfs"]
    )

    paired["additional_urgent_patients"] = (
        paired["high_priority_allocated_ai"]
        - paired["high_priority_allocated_fcfs"]
    )

    paired["overall_allocation_difference_pp"] = (
        paired["allocation_rate_percent_ai"]
        - paired["allocation_rate_percent_fcfs"]
    )

    paired["execution_time_difference_ms"] = (
        paired["execution_time_ms_fcfs"]
        - paired["execution_time_ms_ai"]
    )

    return paired.round(4)


def verify_safety(results):

    safety = results[SAFETY_COLUMNS].sum()

    print("\nSAFETY VERIFICATION")
    print(safety.to_string())

    assert safety.sum() == 0, (
        "Safety violations detected."
    )

    assert (
        results["recommended_allocations"]
        <= results["initial_available_beds"]
    ).all(), "Bed capacity exceeded."

    assert (
        results["remaining_patients"]
        == results["reported_unallocated_patients"]
    ).all(), "Inconsistent waiting patient counts."

    assert (
        results["recommended_allocations"]
        + results["remaining_patients"]
        == results["total_patients"]
    ).all(), "Patient conservation failed."

    print("\nAll stress testing safety checks passed.")

    return safety


def main():

    results = run_stress_tests()

    summary = summarise_results(results)

    paired = compare_paired_results(results)

    safety = verify_safety(results)

    improvement = paired[
        "urgent_allocation_improvement_pp"
    ]

    tolerance = 1e-8

    better = int(
        (improvement > tolerance).sum()
    )

    equal = int(
        (improvement.abs() <= tolerance).sum()
    )

    worse = int(
        (improvement < -tolerance).sum()
    )

    overall = (
        results.groupby("method", as_index=False)
        .agg(
            total_patients=("total_patients", "sum"),

            total_recommendations=(
                "recommended_allocations", "sum"
            ),

            total_urgent_patients=(
                "high_priority_patients", "sum"
            ),

            urgent_patients_selected=(
                "high_priority_allocated", "sum"
            ),

            mean_execution_time_ms=(
                "execution_time_ms", "mean"
            ),
        )
    )

    overall["overall_allocation_rate"] = (
        overall["total_recommendations"]
        / overall["total_patients"] * 100
    )

    overall["overall_high_priority_rate"] = (
        overall["urgent_patients_selected"]
        / overall["total_urgent_patients"] * 100
    )

    overall = overall.round(4)

    outcome_summary = pd.DataFrame([{
        "total_scenarios": len(paired),
        "ai_better_scenarios": better,
        "equal_scenarios": equal,
        "ai_worse_scenarios": worse,
        "mean_urgent_improvement_pp": improvement.mean(),
        "median_urgent_improvement_pp": improvement.median(),
        "minimum_improvement_pp": improvement.min(),
        "maximum_improvement_pp": improvement.max(),
        "additional_urgent_patients": paired[
            "additional_urgent_patients"
        ].sum(),
    }]).round(4)

    print("\n" + "=" * 75)
    print("OVERALL STRESS TEST RESULTS")
    print("=" * 75)

    print(overall.to_string(index=False))

    print("\nPAIRED COMPARISON")
    print(outcome_summary.to_string(index=False))

    print("\nSCALABILITY SUMMARY")

    scalability = (
        results.groupby(
            ["method", "patient_demand"],
            as_index=False,
        )
        .agg(
            mean_execution_time_ms=(
                "execution_time_ms", "mean"
            ),
            max_execution_time_ms=(
                "execution_time_ms", "max"
            ),
        )
        .round(4)
    )

    print(scalability.to_string(index=False))

    # Verify expected experiment size.

    assert len(results) == 300
    assert len(paired) == 150
    assert len(summary) == 30

    # Save all evaluation evidence.

    results.to_csv(
        OUTPUT_DIR / "stress_test_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "stress_test_summary.csv",
        index=False,
    )

    paired.to_csv(
        OUTPUT_DIR / "stress_test_paired_comparison.csv",
        index=False,
    )

    overall.to_csv(
        OUTPUT_DIR / "stress_test_overall.csv",
        index=False,
    )

    outcome_summary.to_csv(
        OUTPUT_DIR / "stress_test_outcomes.csv",
        index=False,
    )

    scalability.to_csv(
        OUTPUT_DIR / "scalability_summary.csv",
        index=False,
    )

    safety.rename("violations").to_csv(
        OUTPUT_DIR / "stress_test_safety.csv"
    )

    print("\nResults saved to:", OUTPUT_DIR)
    print("STRESS TESTING COMPLETED")


if __name__ == "__main__":
    main()