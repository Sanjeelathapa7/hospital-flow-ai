
# HospitalFlow AI
# Agent 4: Coordination and Human Approval

from collections import Counter
from datetime import datetime, timezone

from core.state import HospitalState

from core.config import (
    BED_OCCUPANCY_WARNING,
    BED_OCCUPANCY_CRITICAL
)


class CoordinationAgent:

    def __init__(self):

        self.name = "Coordination and Human Approval Agent"

        self.agent_id = "AGENT-04"

        self.goal = (
            "Monitor hospital operations, detect "
            "bottlenecks and coordinate safe "
            "human-reviewed bed recommendations."
        )

    # ==========================================
    # 1. Create standardised alerts
    # ==========================================

    def make_alert(
        self,
        alert_type,
        severity,
        message,
        **details
    ):

        return {
            "agent_id": self.agent_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            **details
        }

    # ==========================================
    # 2. Monitor hospital bed capacity
    # ==========================================

    def monitor_capacity(self, state):

        beds = state.get("beds", [])

        if not beds:
            return []

        occupied = sum(
            bed.get("status") == "Occupied"
            for bed in beds
        )

        occupancy = occupied / len(beds)

        if occupancy >= BED_OCCUPANCY_CRITICAL:

            return [
                self.make_alert(
                    "CRITICAL_BED_OCCUPANCY",
                    "Critical",
                    "Hospital bed occupancy has "
                    "reached the critical threshold.",
                    occupancy_rate=round(
                        occupancy * 100, 2
                    )
                )
            ]

        if occupancy >= BED_OCCUPANCY_WARNING:

            return [
                self.make_alert(
                    "HIGH_BED_OCCUPANCY",
                    "Warning",
                    "Hospital capacity requires review.",
                    occupancy_rate=round(
                        occupancy * 100, 2
                    )
                )
            ]

        return []

    # ==========================================
    # 3. Monitor capacity by bed type
    # ==========================================

    def monitor_bed_categories(self, state):

        alerts = []

        beds = state.get("beds", [])

        bed_types = sorted({
            bed["bed_type"]
            for bed in beds
            if bed.get("bed_type")
        })

        for bed_type in bed_types:

            category = [
                bed for bed in beds
                if bed.get("bed_type") == bed_type
            ]

            occupied = sum(
                bed.get("status") == "Occupied"
                for bed in category
            )

            occupancy = occupied / len(category)

            if occupancy >= BED_OCCUPANCY_CRITICAL:

                severity = "Critical"

            elif occupancy >= BED_OCCUPANCY_WARNING:

                severity = "Warning"

            else:

                continue

            alerts.append(
                self.make_alert(
                    "BED_CATEGORY_PRESSURE",
                    severity,
                    f"{bed_type} capacity requires review.",
                    bed_type=bed_type,
                    occupancy_rate=round(
                        occupancy * 100, 2
                    )
                )
            )

        return alerts

    # ==========================================
    # 4. Monitor staffing workload
    # ==========================================

    def monitor_staffing(self, state):

        alerts = []

        departments = {}

        for record in state.get("staffing", []):

            department = record.get(
                "department", "Unknown"
            )

            workload = str(
                record.get("workload_category", "")
            ).strip().lower()

            if workload == "critical":

                departments[department] = "Critical"

            elif workload == "high":

                if departments.get(
                    department
                ) != "Critical":

                    departments[department] = "Warning"

        for department, severity in departments.items():

            alerts.append(
                self.make_alert(
                    "STAFFING_WORKLOAD_PRESSURE",
                    severity,
                    f"Staffing workload requires "
                    f"review in {department}.",
                    department=department
                )
            )

        return alerts

    # ==========================================
    # 5. Validate allocation recommendations
    # ==========================================

    def validate_allocations(self, state):

        recommendations = state.get(
            "allocation_recommendations", []
        )

        beds = {
            bed["bed_id"]: bed
            for bed in state.get("beds", [])
        }

        bed_counts = Counter(
            item["bed_id"]
            for item in recommendations
        )

        encounter_counts = Counter(
            (
                item["patient_id"],
                item["encounter_id"]
            )
            for item in recommendations
        )

        queue = {
            (
                patient["patient_id"],
                patient["encounter_id"]
            ): patient
            for patient in state.get(
                "prioritised_queue", []
            )
        }

        valid = []

        alerts = []

        for item in recommendations:

            bed_id = item["bed_id"]

            key = (
                item["patient_id"],
                item["encounter_id"]
            )

            bed = beds.get(bed_id)

            patient = queue.get(key)

            reason = None

            if bed_counts[bed_id] > 1:

                reason = "Duplicate bed recommendation."

            elif encounter_counts[key] > 1:

                reason = "Duplicate encounter recommendation."

            elif patient is None:

                reason = "Patient is not in admission queue."

            elif bed is None:

                reason = "Bed does not exist in inventory."

            elif (
                bed.get("status") != "Available"
                or bed.get("allocatable") is not True
            ):

                reason = "Bed is no longer available."

            elif str(
                item.get("bed_type", "")
            ).lower() != str(
                bed.get("bed_type", "")
            ).lower():

                reason = "Bed type does not match inventory."

            else:

                from agents.bed_allocation_agent import (
                    BedAllocationAgent
                )

                compatible_types = (
                    BedAllocationAgent()
                    .get_compatible_types(patient)
                )

                if str(
                    bed["bed_type"]
                ).strip().lower() not in compatible_types:

                    reason = "Bed requirements are incompatible."

            if reason:

                alerts.append(
                    self.make_alert(
                        "INVALID_ALLOCATION",
                        "Critical",
                        reason,
                        bed_id=bed_id,
                        patient_id=item["patient_id"],
                        encounter_id=item["encounter_id"]
                    )
                )

            else:

                valid.append(item.copy())

        return valid, alerts

    # ==========================================
    # 6. Generate operational recommendations
    # ==========================================

    def generate_recommendations(self, alerts):

        recommendations = []

        action_map = {

            "CRITICAL_BED_OCCUPANCY":
                "Escalate hospital capacity pressure.",

            "HIGH_BED_OCCUPANCY":
                "Review available hospital resources.",

            "BED_CATEGORY_PRESSURE":
                "Review capacity for the affected bed type.",

            "STAFFING_WORKLOAD_PRESSURE":
                "Request staffing supervisor review.",

            "ADMISSION_WAITING_TIME":
                "Review prolonged admission waiting.",

            "BED_ALLOCATION_PENDING":
                "Review unmet bed requirements.",

            "INVALID_ALLOCATION":
                "Block the conflicting allocation.",

            "MODEL_RELIABILITY_LIMITATION":
                "Use LOS estimates for planning only.",

            "PREDICTION_UNAVAILABLE":
                "Continue manual resource planning."
        }

        for alert in alerts:

            recommendations.append({

                "source_agent":
                    alert.get("agent_id"),

                "alert_type":
                    alert.get("alert_type"),

                "severity":
                    alert.get("severity"),

                "action":
                    action_map.get(
                        alert.get("alert_type"),
                        "Request operational review."
                    ),

                "patient_id":
                    alert.get("patient_id"),

                "bed_id":
                    alert.get("bed_id"),

                "department":
                    alert.get("department"),

                "bed_type":
                    alert.get("bed_type"),

                "human_review_required": True
            })

        return recommendations

    # ==========================================
    # 7. Create human approval requests
    # ==========================================

    def create_approval_requests(
        self,
        state,
        allocations
    ):

        predictions = {

            (
                item["patient_id"],
                item["encounter_id"]
            ): item

            for item in state.get(
                "discharge_predictions", []
            )
        }

        requests = []

        for index, allocation in enumerate(
            allocations, start=1
        ):

            key = (
                allocation["patient_id"],
                allocation["encounter_id"]
            )

            prediction = predictions.get(key, {})

            requests.append({

                **allocation,

                "approval_id": (
                    f"{state.get('simulation_id', 'SIM')}"
                    f"-APR-{index:04d}"
                ),

                "approval_status": "Pending",

                "predicted_los_hours":
                    prediction.get(
                        "predicted_los_hours"
                    ),

                "prediction_reliability":
                    prediction.get(
                        "reliability", "Unavailable"
                    ),

                "reviewer": None,

                "decision_reason": None,

                "decision_time": None
            })

        return requests

    # ==========================================
    # 8. Apply an explicit human decision
    # ==========================================

    def apply_decision(
        self,
        state,
        approval_id,
        decision,
        reviewer,
        reason
    ):

        decision = decision.strip().lower()

        if decision not in ["approve", "reject"]:

            raise ValueError(
                "Decision must be approve or reject."
            )

        if not reviewer.strip() or not reason.strip():

            raise ValueError(
                "Reviewer and decision reason are required."
            )

        requests = state.get(
            "approval_requests", []
        )

        request = next(
            (
                item for item in requests
                if item["approval_id"] == approval_id
            ),
            None
        )

        if request is None:

            raise ValueError(
                "Approval request not found."
            )

        if request["approval_status"] != "Pending":

            raise ValueError(
                "This request has already been reviewed."
            )

        if decision == "approve":

            bed = next(
                (
                    item
                    for item in state.get("beds", [])
                    if item["bed_id"] == request["bed_id"]
                ),
                None
            )

            if (
                bed is None
                or bed.get("status") != "Available"
                or bed.get("allocatable") is not True
            ):

                raise ValueError(
                    "Approval blocked: bed unavailable."
                )

            already_approved = state.get(
                "approved_allocations", []
            )

            if any(
                item["bed_id"] == request["bed_id"]
                or (
                    item["patient_id"] == request["patient_id"]
                    and item["encounter_id"] == request["encounter_id"]
                )
                for item in already_approved
            ):

                raise ValueError(
                    "Approval blocked: duplicate allocation."
                )

            # Revalidate compatibility before approval.

            from agents.bed_allocation_agent import (
                BedAllocationAgent
            )

            patient = next(
                (
                    item for item in state.get(
                        "prioritised_queue", []
                    )
                    if item["patient_id"] == request["patient_id"]
                    and item["encounter_id"] == request["encounter_id"]
                ),
                None
            )

            if patient is None:

                raise ValueError(
                    "Approval blocked: patient not in queue."
                )

            compatible_types = (
                BedAllocationAgent()
                .get_compatible_types(patient)
            )

            if str(
                bed["bed_type"]
            ).strip().lower() not in compatible_types:

                raise ValueError(
                    "Approval blocked: incompatible bed."
                )

            # Update the simulated bed state.
            # No real hospital system is modified.

            bed["status"] = "Occupied"

            bed["allocatable"] = False

            bed["assigned_patient_id"] = (
                request["patient_id"]
            )

            bed["assigned_encounter_id"] = (
                request["encounter_id"]
            )

            request["approval_status"] = "Approved"

            request["allocation_status"] = "Approved"

            destination = "approved_allocations"

            # Remove approved patient from active queues.

            for queue_name in [
                "waiting_patients",
                "prioritised_queue",
                "unallocated_patients"
            ]:

                state[queue_name] = [

                    item
                    for item in state.get(queue_name, [])

                    if not (
                        item["patient_id"] == request["patient_id"]
                        and item["encounter_id"] == request["encounter_id"]
                    )
                ]

            state["available_beds"] = [

                item
                for item in state.get("available_beds", [])

                if item["bed_id"] != request["bed_id"]
            ]

        else:

            request["approval_status"] = "Rejected"

            request["allocation_status"] = "Rejected"

            destination = "rejected_allocations"

            # Rejected patients remain in the queue.

            patient = next(
                (
                    item for item in state.get(
                        "prioritised_queue", []
                    )
                    if item["patient_id"] == request["patient_id"]
                    and item["encounter_id"] == request["encounter_id"]
                ),
                None
            )

            if patient is not None:

                state.setdefault(
                    "unallocated_patients", []
                ).append({

                    **patient,

                    "allocation_status": "Waiting",

                    "reason": "Allocation rejected by reviewer."
                })

        request["reviewer"] = reviewer

        request["decision_reason"] = reason

        request["decision_time"] = (
            datetime.now(timezone.utc).isoformat()
        )

        state.setdefault(
            destination, []
        ).append(request.copy())

        # Synchronise the original recommendation.

        for item in state.get(
            "allocation_recommendations", []
        ):

            if (
                item["patient_id"] == request["patient_id"]
                and item["encounter_id"] == request["encounter_id"]
                and item["bed_id"] == request["bed_id"]
            ):

                item["allocation_status"] = (
                    request["allocation_status"]
                )

        # Record the human decision.

        state.setdefault(
            "audit_log", []
        ).append({

            "event_type": "HUMAN_ALLOCATION_DECISION",

            "approval_id": approval_id,

            "patient_id": request["patient_id"],

            "encounter_id": request["encounter_id"],

            "bed_id": request["bed_id"],

            "decision": request["approval_status"],

            "reviewer": reviewer,

            "reason": reason,

            "decision_time": request["decision_time"],

            "simulation_time": state["simulation_time"]
        })

        # Refresh approval status.

        pending = sum(
            item["approval_status"] == "Pending"
            for item in requests
        )

        state["approval_required"] = pending > 0

        state["approval_status"] = (
            "Pending" if pending else "Completed"
        )

        state.setdefault(
            "metrics", {}
        )["coordination"] = self.calculate_metrics(state)

        return state

    # ==========================================
    # 9. Calculate coordination metrics
    # ==========================================

    def calculate_metrics(self, state):

        alerts = state.get("alerts", [])

        requests = state.get(
            "approval_requests", []
        )

        return {

            "total_alerts": len(alerts),

            "critical_alerts": sum(
                item.get("severity") == "Critical"
                for item in alerts
            ),

            "warning_alerts": sum(
                item.get("severity") == "Warning"
                for item in alerts
            ),

            "operational_recommendations":
                len(state.get("recommendations", [])),

            "approval_requests": len(requests),

            "pending_approvals": sum(
                item["approval_status"] == "Pending"
                for item in requests
            ),

            "approved_allocations":
                len(state.get("approved_allocations", [])),

            "rejected_allocations":
                len(state.get("rejected_allocations", [])),

            "audit_events":
                len(state.get("audit_log", []))
        }

    # ==========================================
    # 10. Execute the coordination agent
    # ==========================================

    def run(
        self,
        state: HospitalState
    ) -> HospitalState:

        # Prevent accidental recreation of existing
        # approval requests within the same run.

        if state.get("approval_requests"):

            raise ValueError(
                "Approval requests already exist. "
                "Review them before starting a new run."
            )

        capacity_alerts = self.monitor_capacity(state)

        category_alerts = self.monitor_bed_categories(state)

        staffing_alerts = self.monitor_staffing(state)

        valid_allocations, safety_alerts = (
            self.validate_allocations(state)
        )

        state["alerts"] = [

            alert
            for alert in state.get("alerts", [])

            if alert.get("agent_id") != self.agent_id

        ] + (
            capacity_alerts
            + category_alerts
            + staffing_alerts
            + safety_alerts
        )

        state["recommendations"] = (
            self.generate_recommendations(
                state["alerts"]
            )
        )

        state["approval_requests"] = (
            self.create_approval_requests(
                state,
                valid_allocations
            )
        )

        state["approval_required"] = bool(
            state["approval_requests"]
        )

        state["approval_status"] = (
            "Pending"
            if state["approval_required"]
            else "Not Required"
        )

        state.setdefault(
            "metrics", {}
        )["coordination"] = self.calculate_metrics(state)

        state.setdefault(
            "agent_messages", []
        ).append({

            "sender": self.agent_id,

            "receiver": "HUMAN_ADMINISTRATOR",

            "message_type": "OPERATIONAL_REVIEW",

            "alert_count": len(state["alerts"]),

            "approval_count":
                len(state["approval_requests"]),

            "simulation_time": state["simulation_time"]
        })

        completed = state.setdefault(
            "completed_agents", []
        )

        if self.agent_id not in completed:

            completed.append(self.agent_id)

        return state


# LangGraph-compatible node.

def coordination_agent_node(
    state: HospitalState
) -> HospitalState:

    agent = CoordinationAgent()

    return agent.run(state)
