from collections import Counter

from agents.bed_allocation_agent import BedAllocationAgent


def percentage(numerator, denominator):

    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100,
        2,
    )


def calculate_performance(
    method,
    patients,
    beds,
    recommendations,
    unallocated_patients,
    execution_time,
):

    compatibility_agent = BedAllocationAgent()

    patient_lookup = {
        (p["patient_id"], p["encounter_id"]): p
        for p in patients
    }

    bed_lookup = {
        bed["bed_id"]: bed
        for bed in beds
    }

    total_patients = len(patient_lookup)
    total_beds = len(bed_lookup)

    occupied_beds = sum(
        bed["status"] == "Occupied"
        for bed in bed_lookup.values()
    )

    available_beds = sum(
        bed["status"] == "Available"
        and bed.get("allocatable") is True
        for bed in bed_lookup.values()
    )

    urgent_patients = {
        key
        for key, patient in patient_lookup.items()
        if int(patient["priority"]) <= 2
    }

    allocated_keys = {
        (r["patient_id"], r["encounter_id"])
        for r in recommendations
    }

    allocated_urgent = (
        allocated_keys & urgent_patients
    )

    bed_counts = Counter(
        r["bed_id"] for r in recommendations
    )

    encounter_counts = Counter(
        (r["patient_id"], r["encounter_id"])
        for r in recommendations
    )

    duplicate_beds = sum(
        count - 1
        for count in bed_counts.values()
        if count > 1
    )

    duplicate_encounters = sum(
        count - 1
        for count in encounter_counts.values()
        if count > 1
    )

    compatibility_violations = 0
    unavailable_bed_violations = 0
    invalid_reference_violations = 0

    valid_recommended_beds = set()

    for recommendation in recommendations:

        key = (
            recommendation["patient_id"],
            recommendation["encounter_id"],
        )

        patient = patient_lookup.get(key)

        bed = bed_lookup.get(
            recommendation["bed_id"]
        )

        if patient is None or bed is None:
            invalid_reference_violations += 1
            continue

        compatible_types = {
            str(value).strip().lower()
            for value in compatibility_agent
            .get_compatible_types(patient)
        }

        compatible = (
            str(bed["bed_type"]).strip().lower()
            in compatible_types
        )

        available = (
            bed["status"] == "Available"
            and bed.get("allocatable") is True
        )

        if not compatible:
            compatibility_violations += 1

        if not available:
            unavailable_bed_violations += 1

        if compatible and available:
            valid_recommended_beds.add(
                bed["bed_id"]
            )

    selected_waits = [
        float(patient_lookup[key].get(
            "waiting_time_minutes", 0
        ))
        for key in allocated_keys
        if key in patient_lookup
    ]

    remaining_waits = [
        float(patient.get(
            "waiting_time_minutes", 0
        ))
        for key, patient in patient_lookup.items()
        if key not in allocated_keys
    ]

    valid_allocated_keys = (
        allocated_keys & patient_lookup.keys()
    )

    return {
        "method": method,

        "total_patients": total_patients,
        "total_beds": total_beds,
        "initial_available_beds": available_beds,

        "recommended_allocations": len(
            recommendations
        ),

        "remaining_patients": (
            total_patients
            - len(valid_allocated_keys)
        ),

        "reported_unallocated_patients": len(
            unallocated_patients
        ),

        "allocation_rate_percent": percentage(
            len(valid_allocated_keys),
            total_patients,
        ),

        "high_priority_patients": len(
            urgent_patients
        ),

        "high_priority_allocated": len(
            allocated_urgent
        ),

        "high_priority_allocation_rate": percentage(
            len(allocated_urgent),
            len(urgent_patients),
        ),

        "initial_occupancy_percent": percentage(
            occupied_beds,
            total_beds,
        ),

        "projected_occupancy_percent": percentage(
            occupied_beds
            + len(valid_recommended_beds),
            total_beds,
        ),

        "compatibility_violations": (
            compatibility_violations
        ),

        "unavailable_bed_violations": (
            unavailable_bed_violations
        ),

        "invalid_reference_violations": (
            invalid_reference_violations
        ),

        "duplicate_bed_allocations": duplicate_beds,

        "duplicate_encounter_allocations": (
            duplicate_encounters
        ),

        "average_selected_wait_minutes": (
            round(
                sum(selected_waits)
                / len(selected_waits),
                2,
            )
            if selected_waits else 0
        ),

        "average_remaining_wait_minutes": (
            round(
                sum(remaining_waits)
                / len(remaining_waits),
                2,
            )
            if remaining_waits else 0
        ),

        "execution_time_ms": round(
            execution_time * 1000,
            4,
        ),
    }