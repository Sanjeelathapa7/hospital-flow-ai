
# HospitalFlow AI
# Agent 2: Intelligent Bed Allocation

from core.state import HospitalState


class BedAllocationAgent:

    def __init__(self):

        self.name = "Bed Allocation Agent"

        self.agent_id = "AGENT-02"

        self.goal = (
            "Recommend compatible available beds "
            "without duplicate allocations."
        )

    # ==========================================
    # 1. Perceive available hospital resources
    # ==========================================

    def perceive(self, state):

        return [
            bed.copy()
            for bed in state.get("beds", [])
            if bed.get("status") == "Available"
            and bed.get("allocatable") is True
        ]

    # ==========================================
    # 2. Determine requested bed compatibility
    # ==========================================

    def get_compatible_types(self, patient):

        requested_type = patient.get(
            "requested_bed_type"
        )

        requested_care = patient.get(
            "requested_care"
        )

        valid_types = {
            "medical ward",
            "surgical ward",
            "shared room",
            "private room",
            "isolation room",
            "icu",
            "step-down unit",
            "ed observation",
            "pediatric ward"
        }

        if requested_care == "ICU":

            if requested_type and (
                str(requested_type).strip().lower()
                != "icu"
            ):
                return []

            return ["icu"]

        # Inpatient care alone is insufficient
        # to determine a safe physical bed type.

        if not requested_type:
            return []

        requested_type = (
            str(requested_type).strip().lower()
        )

        if requested_type not in valid_types:
            return []

        return [requested_type]

    # ==========================================
    # 3. Identify compatible candidate beds
    # ==========================================

    def find_candidates(
        self,
        patient,
        available_beds,
        reserved_bed_ids
    ):

        compatible_types = (
            self.get_compatible_types(patient)
        )

        candidates = [

            bed for bed in available_beds

            if bed["bed_id"]
            not in reserved_bed_ids

            and str(
                bed["bed_type"]
            ).strip().lower()
            in compatible_types
        ]

        # Deterministic selection.
        # No random bed assignment.

        return sorted(
            candidates,
            key=lambda bed: str(bed["bed_id"])
        )

    # ==========================================
    # 4. Reason and recommend allocations
    # ==========================================

    def allocate(self, queue, available_beds):

        recommendations = []

        unallocated = []

        reserved_bed_ids = set()

        processed_encounters = set()

        # Preserve urgency-based ordering.

        queue = sorted(
            queue,
            key=lambda patient: (
                int(patient["priority"]),
                -float(
                    patient["waiting_time_minutes"]
                ),
                str(patient.get("arrival_time", "")),
                str(patient["patient_id"]),
                str(patient["encounter_id"])
            )
        )

        for patient in queue:

            encounter_key = (
                patient["patient_id"],
                patient["encounter_id"]
            )

            if encounter_key in processed_encounters:
                continue

            processed_encounters.add(encounter_key)

            compatible_types = (
                self.get_compatible_types(patient)
            )

            if not compatible_types:

                unallocated.append({
                    **patient,
                    "allocation_status": "Waiting",
                    "reason":
                        "Bed requirements need human review"
                })

                continue

            candidates = self.find_candidates(
                patient,
                available_beds,
                reserved_bed_ids
            )

            if not candidates:

                unallocated.append({
                    **patient,
                    "allocation_status": "Waiting",
                    "reason":
                        "No compatible available bed"
                })

                continue

            selected_bed = candidates[0]

            reserved_bed_ids.add(
                selected_bed["bed_id"]
            )

            recommendations.append({

                "patient_id":
                    patient["patient_id"],

                "encounter_id":
                    patient["encounter_id"],

                "priority":
                    patient["priority"],

                "waiting_time_minutes":
                    patient["waiting_time_minutes"],

                "requested_care":
                    patient.get("requested_care"),

                "bed_id":
                    selected_bed["bed_id"],

                "bed_type":
                    selected_bed["bed_type"],

                "requirement_source":
                    patient.get(
                        "requirement_source",
                        "Recorded admission disposition"
                    ),

                "allocation_status":
                    "Pending Approval",

                "human_approval_required":
                    True,

                "reason":
                    "Available bed matches "
                    "requested bed category"
            })

        return recommendations, unallocated

    # ==========================================
    # 5. Generate allocation alerts
    # ==========================================

    def generate_alerts(self, unallocated):

        alerts = []

        for patient in unallocated:

            severity = (
                "Critical"
                if int(patient["priority"]) <= 2
                else "Warning"
            )

            alerts.append({

                "agent_id": self.agent_id,

                "alert_type":
                    "BED_ALLOCATION_PENDING",

                "severity": severity,

                "patient_id":
                    patient["patient_id"],

                "encounter_id":
                    patient["encounter_id"],

                "message":
                    patient["reason"]
            })

        return alerts

    # ==========================================
    # 6. Calculate operational metrics
    # ==========================================

    def calculate_metrics(
        self,
        recommendations,
        unallocated,
        available_beds
    ):

        total = (
            len(recommendations)
            + len(unallocated)
        )

        allocated_ids = [
            item["bed_id"]
            for item in recommendations
        ]

        return {

            "total_admission_requests": total,

            "available_beds":
                len(available_beds),

            "recommended_allocations":
                len(recommendations),

            "unallocated_patients":
                len(unallocated),

            "recommendation_rate": round(
                len(recommendations) / total * 100,
                2
            ) if total else 0.0,

            "duplicate_bed_recommendations":
                len(allocated_ids)
                - len(set(allocated_ids)),

            "icu_recommendations": sum(
                str(item["bed_type"]).lower()
                == "icu"
                for item in recommendations
            ),

            "remaining_unreserved_beds":
                len(available_beds)
                - len(recommendations)
        }

    # ==========================================
    # 7. Execute Agent 2
    # ==========================================

    def run(
        self,
        state: HospitalState
    ) -> HospitalState:

        queue = state.get(
            "prioritised_queue", []
        )

        available_beds = self.perceive(state)

        recommendations, unallocated = (
            self.allocate(
                queue,
                available_beds
            )
        )

        alerts = self.generate_alerts(
            unallocated
        )

        metrics = self.calculate_metrics(
            recommendations,
            unallocated,
            available_beds
        )

        state["available_beds"] = available_beds

        state["allocation_recommendations"] = (
            recommendations
        )

        state["unallocated_patients"] = (
            unallocated
        )

        state["alerts"] = [

            alert
            for alert in state.get("alerts", [])

            if alert.get("agent_id")
            != self.agent_id

        ] + alerts

        state.setdefault(
            "metrics", {}
        )["bed_allocation"] = metrics

        state["approval_required"] = bool(
            recommendations
        )

        state["approval_status"] = (
            "Pending"
            if recommendations
            else "Not Required"
        )

        state.setdefault(
            "agent_messages", []
        ).append({

            "sender": self.agent_id,

            "receiver": "AGENT-04",

            "message_type":
                "BED_ALLOCATION_RECOMMENDATIONS",

            "recommendation_count":
                len(recommendations),

            "unallocated_count":
                len(unallocated),

            "simulation_time":
                state["simulation_time"]
        })

        completed = state.setdefault(
            "completed_agents", []
        )

        if self.agent_id not in completed:

            completed.append(self.agent_id)

        return state


# LangGraph-compatible node.

def bed_allocation_agent_node(
    state: HospitalState
) -> HospitalState:

    agent = BedAllocationAgent()

    return agent.run(state)
