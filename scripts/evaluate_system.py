import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.scenario_evaluator import (
    build_scenarios,
    evaluate_scenario,
)


OUTPUT_DIR = ROOT / "outputs" / "system_evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    print("=" * 75)
    print("HOSPITALFLOW AI - COMPARATIVE SYSTEM EVALUATION")
    print("=" * 75)

    scenarios = build_scenarios()

    all_results = []
    all_allocations = []
    all_waiting = []
    scenario_summary = []

    for scenario_name, scenario in scenarios.items():

        patients = scenario["patients"]
        beds = scenario["beds"]

        print("\n" + "-" * 75)
        print("SCENARIO:", scenario_name)
        print("-" * 75)

        available = sum(
            bed["status"] == "Available"
            and bed["allocatable"]
            for bed in beds
        )

        urgent = sum(
            patient["priority"] <= 2
            for patient in patients
        )

        print("Total patients:", len(patients))
        print("Total beds:", len(beds))
        print("Available beds:", available)
        print("High-priority patients:", urgent)

        scenario_summary.append({
            "scenario": scenario_name,
            "total_patients": len(patients),
            "total_beds": len(beds),
            "available_beds": available,
            "high_priority_patients": urgent,
        })

        results, allocations, waiting = evaluate_scenario(
            scenario_name,
            patients,
            beds,
            repetitions=10,
        )

        all_results.extend(results)
        all_allocations.extend(allocations)
        all_waiting.extend(waiting)

        comparison = pd.DataFrame(results)

        columns = [
            "method",
            "recommended_allocations",
            "remaining_patients",
            "allocation_rate_percent",
            "high_priority_allocation_rate",
            "compatibility_violations",
            "duplicate_bed_allocations",
            "execution_time_ms",
        ]

        print("\nCOMPARATIVE RESULTS")

        print(
            comparison[columns].to_string(
                index=False
            )
        )

    results_df = pd.DataFrame(all_results)
    allocations_df = pd.DataFrame(all_allocations)
    waiting_df = pd.DataFrame(all_waiting)

    scenario_df = pd.DataFrame(scenario_summary)

    results_df.to_csv(
        OUTPUT_DIR / "scenario_performance.csv",
        index=False,
    )

    allocations_df.to_csv(
        OUTPUT_DIR / "allocation_details.csv",
        index=False,
    )

    waiting_df.to_csv(
        OUTPUT_DIR / "unallocated_patient_details.csv",
        index=False,
    )

    scenario_df.to_csv(
        OUTPUT_DIR / "scenario_configuration.csv",
        index=False,
    )

    # Aggregate results across controlled scenarios.

    overall = (
        results_df.groupby("method", as_index=False)
        .agg(
            total_patients=("total_patients", "sum"),
            recommended_allocations=(
                "recommended_allocations", "sum"
            ),
            remaining_patients=(
                "remaining_patients", "sum"
            ),
            high_priority_patients=(
                "high_priority_patients", "sum"
            ),
            high_priority_allocated=(
                "high_priority_allocated", "sum"
            ),
            compatibility_violations=(
                "compatibility_violations", "sum"
            ),
            unavailable_bed_violations=(
                "unavailable_bed_violations", "sum"
            ),
            invalid_reference_violations=(
                "invalid_reference_violations", "sum"
            ),
            duplicate_bed_allocations=(
                "duplicate_bed_allocations", "sum"
            ),
            duplicate_encounter_allocations=(
                "duplicate_encounter_allocations", "sum"
            ),
            mean_execution_time_ms=(
                "execution_time_ms", "mean"
            ),
        )
    )

    overall["overall_allocation_rate"] = (
        overall["recommended_allocations"]
        / overall["total_patients"] * 100
    )

    overall["overall_high_priority_rate"] = (
        overall["high_priority_allocated"]
        / overall["high_priority_patients"] * 100
    )

    overall = overall.round(4)

    overall.to_csv(
        OUTPUT_DIR / "overall_system_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("OVERALL SYSTEM COMPARISON")
    print("=" * 75)

    print(overall.to_string(index=False))

    # Safety verification

    safety_columns = [
        "compatibility_violations",
        "unavailable_bed_violations",
        "invalid_reference_violations",
        "duplicate_bed_allocations",
        "duplicate_encounter_allocations",
    ]

    safety_totals = results_df[safety_columns].sum()

    print("\nSAFETY VERIFICATION")
    print(safety_totals.to_string())

    assert safety_totals.sum() == 0, (
        "Allocation safety violations detected."
    )

    assert (
        results_df["remaining_patients"]
        == results_df["reported_unallocated_patients"]
    ).all(), "Inconsistent patient counts"

    assert (
        results_df["recommended_allocations"]
        <= results_df["initial_available_beds"]
    ).all(), "Recommendations exceed available beds"

    print("\nAll allocation safety checks passed.")

    print("\nResults saved to:", OUTPUT_DIR)
    print("SYSTEM EVALUATION COMPLETED")


if __name__ == "__main__":
    main()