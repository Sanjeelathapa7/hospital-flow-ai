from copy import deepcopy

import pandas as pd

from agents.bed_allocation_agent import BedAllocationAgent


class FCFSBaseline:
    """
    Traditional First-Come, First-Served allocation.

    Patients are processed according to arrival time.
    Bed compatibility and availability constraints
    remain identical to HospitalFlow AI.
    """

    def __init__(self):
        self.name = "First-Come, First-Served"
        self.compatibility_agent = BedAllocationAgent()

    def allocate(self, patients, beds):

        patients = deepcopy(patients)
        beds = deepcopy(beds)

        available_beds = [
            bed for bed in beds
            if bed.get("status") == "Available"
            and bed.get("allocatable") is True
        ]

        # FCFS ordering: earliest arrival first.
        ordered_patients = sorted(
            patients,
            key=lambda patient: (
                pd.Timestamp(patient["arrival_time"]),
                str(patient["patient_id"]),
                str(patient["encounter_id"]),
            ),
        )

        reserved_beds = set()
        processed_encounters = set()

        recommendations = []
        unallocated_patients = []

        for patient in ordered_patients:

            encounter_key = (
                patient["patient_id"],
                patient["encounter_id"],
            )

            if encounter_key in processed_encounters:
                continue

            processed_encounters.add(encounter_key)

            compatible_types = (
                self.compatibility_agent
                .get_compatible_types(patient)
            )

            compatible_types = {
                str(value).strip().lower()
                for value in compatible_types
            }

            candidates = [
                bed for bed in available_beds
                if str(bed["bed_type"]).strip().lower()
                in compatible_types
                and bed["bed_id"] not in reserved_beds
            ]

            candidates.sort(
                key=lambda bed: str(bed["bed_id"])
            )

            if not candidates:

                unallocated = patient.copy()

                unallocated["allocation_status"] = (
                    "Waiting"
                )

                unallocated["reason"] = (
                    "No compatible available bed"
                    if compatible_types
                    else "Bed requirements need human review"
                )

                unallocated_patients.append(unallocated)
                continue

            selected_bed = candidates[0]

            reserved_beds.add(selected_bed["bed_id"])

            recommendations.append({
                "patient_id": patient["patient_id"],
                "encounter_id": patient["encounter_id"],
                "priority": patient["priority"],
                "waiting_time_minutes": patient.get(
                    "waiting_time_minutes", 0
                ),
                "requested_care": patient.get(
                    "requested_care"
                ),
                "bed_id": selected_bed["bed_id"],
                "bed_type": selected_bed["bed_type"],
                "allocation_status": "Pending Approval",
                "human_approval_required": True,
                "allocation_method": "FCFS",
                "reason": (
                    "Earliest arrival with compatible "
                    "available bed"
                ),
            })

        return recommendations, unallocated_patients