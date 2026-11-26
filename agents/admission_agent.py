
# HospitalFlow AI
# Agent 1: Admission Prioritisation Agent

import pandas as pd

from core.config import (
    PRIORITY_LABELS,
    WAITING_TIME_WARNING_MINUTES,
    WAITING_TIME_CRITICAL_MINUTES
)

from core.state import HospitalState

from core.data_loader import (
    load_admissions,
    load_patients,
    load_bed_events
)


class AdmissionPrioritisationAgent:

    def __init__(self):

        self.name = "Admission Prioritisation Agent"

        self.agent_id = "AGENT-01"

        self.goal = (
            "Prioritise eligible hospital admission "
            "requests using recorded clinical urgency "
            "and operational waiting time."
        )

    # ==========================================
    # 1. Perceive the hospital environment
    # ==========================================

    def perceive(self, simulation_time):

        now = pd.Timestamp(simulation_time)

        admissions = load_admissions()

        patients = load_patients()

        bed_events = load_bed_events()

        keys = ["patient_id", "encounter_id"]

        # Only patients who have arrived
        # and completed triage by simulation time.

        arrivals = admissions.loc[
            (admissions["event_type"] == "ED_ARRIVAL")
            & admissions["patient_id"].notna()
            & admissions["encounter_id"].notna()
            & (admissions["arrival_time"] <= now)
            & (admissions["triage_time"] <= now)
            & (
                admissions["arrival_time"]
                <= admissions["triage_time"]
            )
            & admissions["priority"].isin(
                [1, 2, 3, 4, 5]
            ),
            keys + [
                "department",
                "arrival_time",
                "triage_time",
                "priority"
            ]
        ].copy()

        arrivals = arrivals.sort_values(
            "arrival_time"
        ).drop_duplicates(keys)

        # Only admission decisions already recorded
        # at the current simulation time.

        dispositions = admissions.loc[
            (
                admissions["event_type"]
                == "ED_DISPOSITION"
            )
            & (
                admissions["disposition_time"]
                <= now
            ),
            keys + [
                "disposition",
                "disposition_time"
            ]
        ].copy()

        dispositions = dispositions.sort_values(
            "disposition_time"
        ).drop_duplicates(
            keys,
            keep="last"
        )

        dispositions["disposition"] = (
            dispositions["disposition"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        eligible_dispositions = [
            "inpatient admission",
            "icu admission"
        ]

        dispositions = dispositions.loc[
            dispositions["disposition"].isin(
                eligible_dispositions
            )
        ]

        # Match arrival and admission decision.

        queue = arrivals.merge(
            dispositions,
            on=keys,
            how="inner",
            validate="one_to_one"
        )

        queue = queue.loc[
            queue["triage_time"]
            <= queue["disposition_time"]
        ].copy()

        # Exclude encounters already assigned
        # a physical bed by simulation time.

        assigned = bed_events.loc[
            (
                bed_events["event_type"]
                == "BED_MANAGEMENT"
            )
            & (
                bed_events["occupancy_status"]
                == "Occupied"
            )
            & bed_events["bed_id"].notna()
            & (
                bed_events["assignment_time"]
                <= now
            ),
            keys
        ].dropna().drop_duplicates()

        # Also exclude inpatient stays that have
        # already started, including those without
        # a matched bed-management record.

        started_stays = bed_events.loc[
            bed_events["event_type"].isin(
                ["INPATIENT_STAY", "ICU_STAY"]
            )
            & (
                bed_events["admission_time"]
                <= now
            ),
            keys
        ].dropna().drop_duplicates()

        completed_requests = pd.concat(
            [assigned, started_stays],
            ignore_index=True
        ).drop_duplicates()

        queue = queue.merge(
            completed_requests.assign(
                already_admitted=True
            ),
            on=keys,
            how="left"
        )

        queue = queue.loc[
            queue["already_admitted"].isna()
        ].drop(
            columns=["already_admitted"]
        )

        # Attach baseline patient information.

        patient_columns = [
            "patient_id",
            "age",
            "clinical_complexity_category",
            "baseline_risk_category",
            "primary_department"
        ]

        queue = queue.merge(
            patients[patient_columns],
            on="patient_id",
            how="left",
            validate="many_to_one"
        )

        # Time since arrival in the ED.

        queue["ed_elapsed_minutes"] = (
            now - queue["arrival_time"]
        ).dt.total_seconds() / 60

        # Time since the recorded admission request.

        queue["waiting_time_minutes"] = (
            now - queue["disposition_time"]
        ).dt.total_seconds() / 60

        queue["priority"] = (
            queue["priority"].astype(int)
        )

        return queue

    # ==========================================
    # 2. Reason about patient urgency
    # ==========================================

    def reason(self, queue):

        queue = queue.copy()

        queue["priority_label"] = (
            queue["priority"].map(
                PRIORITY_LABELS
            )
        )

        # Preliminary requested care category.
        # This is not a final physical bed type.

        queue["requested_care"] = (
            queue["disposition"].map({
                "icu admission": "ICU",
                "inpatient admission": "inpatient"
            })
        )

        queue["waiting_status"] = "Normal"

        queue.loc[
            queue["waiting_time_minutes"]
            >= WAITING_TIME_WARNING_MINUTES,
            "waiting_status"
        ] = "Warning"

        queue.loc[
            queue["waiting_time_minutes"]
            >= WAITING_TIME_CRITICAL_MINUTES,
            "waiting_status"
        ] = "Critical"

        # Priority first, longest wait second.
        # IDs provide deterministic tie-breaking.

        queue = queue.sort_values(
            by=[
                "priority",
                "waiting_time_minutes",
                "arrival_time",
                "patient_id",
                "encounter_id"
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True
            ]
        ).reset_index(drop=True)

        queue["queue_position"] = (
            queue.index + 1
        )

        queue["agent_recommendation"] = (
            "Await compatible bed assessment"
        )

        return queue

    # ==========================================
    # 3. Generate waiting-time alerts
    # ==========================================

    def generate_alerts(self, queue):

        alerts = []

        delayed = queue.loc[
            queue["waiting_status"].isin(
                ["Warning", "Critical"]
            )
        ]

        for _, patient in delayed.iterrows():

            alerts.append({
                "agent_id": self.agent_id,

                "alert_type":
                    "ADMISSION_WAITING_TIME",

                "severity":
                    patient["waiting_status"],

                "patient_id":
                    str(patient["patient_id"]),

                "encounter_id":
                    str(patient["encounter_id"]),

                "waiting_time_minutes":
                    round(
                        float(
                            patient[
                                "waiting_time_minutes"
                            ]
                        ),
                        2
                    ),

                "message":
                    "Admission request requires "
                    "operational review."
            })

        return alerts

    # ==========================================
    # 4. Calculate agent performance metrics
    # ==========================================

    def calculate_metrics(self, queue):

        return {
            "waiting_patients": len(queue),

            "highest_urgency_patients": int(
                (queue["priority"] == 1).sum()
            ),

            "high_urgency_patients": int(
                (queue["priority"] == 2).sum()
            ),

            "average_waiting_minutes": (
                round(
                    float(
                        queue[
                            "waiting_time_minutes"
                        ].mean()
                    ),
                    2
                )
                if not queue.empty else 0.0
            ),

            "maximum_waiting_minutes": (
                round(
                    float(
                        queue[
                            "waiting_time_minutes"
                        ].max()
                    ),
                    2
                )
                if not queue.empty else 0.0
            ),

            "warning_waiting_patients": int(
                (
                    queue["waiting_status"]
                    == "Warning"
                ).sum()
            ),

            "critical_waiting_patients": int(
                (
                    queue["waiting_status"]
                    == "Critical"
                ).sum()
            ),

            "icu_requests": int(
                (
                    queue["requested_care"]
                    == "ICU"
                ).sum()
            )
        }

    # ==========================================
    # 5. Convert data to shared-state records
    # ==========================================

    def to_records(self, df):

        # Convert timestamps to ISO strings
        # and missing values to JSON null.

        import json

        return json.loads(
            df.to_json(
                orient="records",
                date_format="iso"
            )
        )

    # ==========================================
    # 6. Execute the intelligent agent
    # ==========================================

    def run(
        self,
        state: HospitalState
    ) -> HospitalState:

        simulation_time = state[
            "simulation_time"
        ]

        waiting = self.perceive(
            simulation_time
        )

        prioritised = self.reason(waiting)

        alerts = self.generate_alerts(
            prioritised
        )

        metrics = self.calculate_metrics(
            prioritised
        )

        state["waiting_patients"] = (
            self.to_records(waiting)
        )

        state["prioritised_queue"] = (
            self.to_records(prioritised)
        )

        state["alerts"] = (
            [
                alert
                for alert in state.get("alerts", [])
                if alert.get("agent_id")
                != self.agent_id
            ] + alerts
        )

        state.setdefault(
            "metrics", {}
        )["admission"] = metrics

        state.setdefault(
            "agent_messages", []
        ).append({
            "sender": self.agent_id,

            "receiver": "AGENT-02",

            "message_type":
                "PRIORITISED_PATIENT_QUEUE",

            "patient_count":
                len(prioritised),

            "simulation_time":
                simulation_time
        })

        state["agent_messages"].append({
            "sender": self.agent_id,

            "receiver": "AGENT-04",

            "message_type":
                "ADMISSION_ALERTS",

            "alert_count": len(alerts),

            "simulation_time":
                simulation_time
        })

        completed = state.setdefault(
            "completed_agents", []
        )

        if self.agent_id not in completed:

            completed.append(self.agent_id)

        return state


# LangGraph-compatible node function.

def admission_agent_node(
    state: HospitalState
) -> HospitalState:

    agent = AdmissionPrioritisationAgent()

    return agent.run(state)
