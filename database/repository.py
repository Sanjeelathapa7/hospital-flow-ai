
# HospitalFlow AI
# Database Repository Layer

import json

from datetime import datetime, timezone

from core.config import DATABASE_PATH

from database.connection import (
    get_connection,
    database_transaction,
    initialise_database
)


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def encode_json(data):

    return json.dumps(
        data,
        default=str,
        allow_nan=False
    )


class HospitalRepository:

    def __init__(
        self,
        db_path=DATABASE_PATH
    ):

        self.db_path = db_path

        initialise_database(db_path)

    # ==========================================
    # 1. Save complete simulation state
    # ==========================================

    def save_state(self, state):

        simulation_id = state["simulation_id"]

        with database_transaction(
            self.db_path
        ) as connection:

            self._save_simulation(
                connection, state
            )

            self._save_patients(
                connection, state
            )

            self._save_beds(
                connection, state
            )

            self._save_predictions(
                connection, state
            )

            self._save_allocations(
                connection, state
            )

            self._save_alerts(
                connection, state
            )

            self._save_audit_logs(
                connection, state
            )

            self._save_messages(
                connection, state
            )

        return simulation_id

    # ==========================================
    # 2. Save simulation information
    # ==========================================

    def _save_simulation(
        self, connection, state
    ):

        now = utc_now()

        connection.execute(
            """
            INSERT INTO simulation_runs (
                simulation_id,
                simulation_time,
                scenario_mode,
                workflow_status,
                approval_status,
                created_at,
                updated_at,
                final_summary_json,
                state_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(simulation_id)
            DO UPDATE SET
                workflow_status =
                    excluded.workflow_status,
                approval_status =
                    excluded.approval_status,
                updated_at =
                    excluded.updated_at,
                final_summary_json =
                    excluded.final_summary_json,
                state_json =
                    excluded.state_json
            """,
            (
                state["simulation_id"],
                state["simulation_time"],
                state.get(
                    "scenario_mode", "historical"
                ),
                state.get(
                    "workflow_status", "Initialised"
                ),
                state.get("approval_status"),
                now,
                now,
                encode_json(
                    state.get("final_summary", {})
                ),
                encode_json(state)
            )
        )

    # ==========================================
    # 3. Save patient queue
    # ==========================================

    def _save_patients(
        self, connection, state
    ):

        simulation_id = state["simulation_id"]

        patients = {}

        sources = [
            state.get("scenario_patients", []),
            state.get("waiting_patients", []),
            state.get("prioritised_queue", []),
            state.get("unallocated_patients", [])
        ]

        for source in sources:

            for patient in source:

                key = (
                    patient["patient_id"],
                    patient["encounter_id"]
                )

                patients.setdefault(
                    key, {}
                ).update(patient)

        approved = {

            (
                item["patient_id"],
                item["encounter_id"]
            )

            for item in state.get(
                "approved_allocations", []
            )
        }

        pending = {

            (
                item["patient_id"],
                item["encounter_id"]
            )

            for item in state.get(
                "approval_requests", []
            )

            if item["approval_status"] == "Pending"
        }

        for key in approved:

            patients.setdefault(
                key,
                {
                    "patient_id": key[0],
                    "encounter_id": key[1]
                }
            )

        for key, patient in patients.items():

            if key in approved:

                status = "Allocated"

            elif key in pending:

                status = "Pending Approval"

            else:

                status = "Waiting"

            connection.execute(
                """
                INSERT INTO patient_queue (
                    simulation_id,
                    patient_id,
                    encounter_id,
                    priority,
                    queue_position,
                    waiting_time_minutes,
                    requested_care,
                    requested_bed_type,
                    queue_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(
                    simulation_id,
                    patient_id,
                    encounter_id
                )
                DO UPDATE SET
                    priority = COALESCE(
                        excluded.priority,
                        patient_queue.priority
                    ),
                    queue_position = COALESCE(
                        excluded.queue_position,
                        patient_queue.queue_position
                    ),
                    waiting_time_minutes = COALESCE(
                        excluded.waiting_time_minutes,
                        patient_queue.waiting_time_minutes
                    ),
                    requested_care = COALESCE(
                        excluded.requested_care,
                        patient_queue.requested_care
                    ),
                    requested_bed_type = COALESCE(
                        excluded.requested_bed_type,
                        patient_queue.requested_bed_type
                    ),
                    queue_status =
                        excluded.queue_status
                """,
                (
                    simulation_id,
                    key[0],
                    key[1],
                    patient.get("priority"),
                    patient.get("queue_position"),
                    patient.get(
                        "waiting_time_minutes"
                    ),
                    patient.get("requested_care"),
                    patient.get(
                        "requested_bed_type"
                    ),
                    status
                )
            )

    # ==========================================
    # 4. Save bed inventory
    # ==========================================

    def _save_beds(
        self, connection, state
    ):

        for bed in state.get("beds", []):

            connection.execute(
                """
                INSERT INTO bed_inventory (
                    simulation_id,
                    bed_id,
                    bed_type,
                    status,
                    allocatable,
                    assigned_patient_id,
                    assigned_encounter_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(
                    simulation_id,
                    bed_id
                )
                DO UPDATE SET
                    status = excluded.status,
                    allocatable =
                        excluded.allocatable,
                    assigned_patient_id =
                        excluded.assigned_patient_id,
                    assigned_encounter_id =
                        excluded.assigned_encounter_id
                """,
                (
                    state["simulation_id"],
                    bed["bed_id"],
                    bed["bed_type"],
                    bed["status"],
                    int(
                        bed.get(
                            "allocatable", False
                        )
                    ),
                    bed.get(
                        "assigned_patient_id"
                    ),
                    bed.get(
                        "assigned_encounter_id"
                    )
                )
            )

    # ==========================================
    # 5. Save LOS predictions
    # ==========================================

    def _save_predictions(
        self, connection, state
    ):

        for item in state.get(
            "discharge_predictions", []
        ):

            connection.execute(
                """
                INSERT INTO discharge_predictions (
                    simulation_id,
                    patient_id,
                    encounter_id,
                    model_name,
                    predicted_los_hours,
                    predicted_los_days,
                    assumed_admission_time,
                    projected_stay_end,
                    reliability,
                    human_review_required,
                    autonomous_discharge_allowed
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(
                    simulation_id,
                    patient_id,
                    encounter_id
                )
                DO UPDATE SET
                    model_name =
                        excluded.model_name,
                    predicted_los_hours =
                        excluded.predicted_los_hours,
                    predicted_los_days =
                        excluded.predicted_los_days,
                    assumed_admission_time =
                        excluded.assumed_admission_time,
                    projected_stay_end =
                        excluded.projected_stay_end,
                    reliability =
                        excluded.reliability
                """,
                (
                    state["simulation_id"],
                    item["patient_id"],
                    item["encounter_id"],
                    item.get("model_name"),
                    item.get("predicted_los_hours"),
                    item.get("predicted_los_days"),
                    item.get("assumed_admission_time"),
                    item.get("projected_stay_end"),
                    item.get("reliability"),
                    int(
                        item.get(
                            "human_review_required", True
                        )
                    ),
                    int(
                        item.get(
                            "autonomous_discharge_allowed",
                            False
                        )
                    )
                )
            )

    # ==========================================
    # 6. Save allocation approval requests
    # ==========================================

    def _save_allocations(
        self, connection, state
    ):

        for item in state.get(
            "approval_requests", []
        ):

            existing = connection.execute(
                """
                SELECT *
                FROM bed_allocations
                WHERE simulation_id = ?
                AND approval_id = ?
                """,
                (
                    state["simulation_id"],
                    item["approval_id"]
                )
            ).fetchone()

            if existing is not None:

                for field in [
                    "patient_id",
                    "encounter_id",
                    "bed_id"
                ]:

                    if existing[field] != item[field]:

                        raise ValueError(
                            "Allocation identity cannot change."
                        )

                if existing["approval_status"] != "Pending":

                    for field in [
                        "approval_status",
                        "reviewer",
                        "decision_reason",
                        "decision_time"
                    ]:

                        if existing[field] != item.get(field):

                            raise ValueError(
                                "A completed approval "
                                "decision cannot be changed."
                            )

            connection.execute(
                """
                INSERT INTO bed_allocations (
                    simulation_id,
                    approval_id,
                    patient_id,
                    encounter_id,
                    bed_id,
                    bed_type,
                    approval_status,
                    reviewer,
                    decision_reason,
                    decision_time,
                    predicted_los_hours
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(
                    simulation_id,
                    approval_id
                )
                DO UPDATE SET
                    approval_status =
                        excluded.approval_status,
                    reviewer =
                        excluded.reviewer,
                    decision_reason =
                        excluded.decision_reason,
                    decision_time =
                        excluded.decision_time
                """,
                (
                    state["simulation_id"],
                    item["approval_id"],
                    item["patient_id"],
                    item["encounter_id"],
                    item["bed_id"],
                    item.get("bed_type"),
                    item["approval_status"],
                    item.get("reviewer"),
                    item.get("decision_reason"),
                    item.get("decision_time"),
                    item.get("predicted_los_hours")
                )
            )

    # ==========================================
    # 7. Save operational alerts
    # ==========================================

    def _save_alerts(
        self, connection, state
    ):

        simulation_id = state["simulation_id"]

        connection.execute(
            """
            DELETE FROM operational_alerts
            WHERE simulation_id = ?
            """,
            (simulation_id,)
        )

        for item in state.get("alerts", []):

            connection.execute(
                """
                INSERT INTO operational_alerts (
                    simulation_id,
                    agent_id,
                    alert_type,
                    severity,
                    message,
                    patient_id,
                    bed_id,
                    department,
                    details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    simulation_id,
                    item.get("agent_id"),
                    item.get("alert_type"),
                    item.get("severity"),
                    item.get("message"),
                    item.get("patient_id"),
                    item.get("bed_id"),
                    item.get("department"),
                    encode_json(item)
                )
            )

    # ==========================================
    # 8. Save append-only human audit records
    # ==========================================

    def _save_audit_logs(
        self, connection, state
    ):

        for item in state.get("audit_log", []):

            connection.execute(
                """
                INSERT INTO audit_logs (
                    simulation_id,
                    event_type,
                    approval_id,
                    patient_id,
                    encounter_id,
                    bed_id,
                    decision,
                    reviewer,
                    reason,
                    decision_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(
                    simulation_id,
                    approval_id
                )
                DO NOTHING
                """,
                (
                    state["simulation_id"],
                    item["event_type"],
                    item["approval_id"],
                    item.get("patient_id"),
                    item.get("encounter_id"),
                    item.get("bed_id"),
                    item["decision"],
                    item["reviewer"],
                    item["reason"],
                    item["decision_time"]
                )
            )

    # ==========================================
    # 9. Save inter-agent communication
    # ==========================================

    def _save_messages(
        self, connection, state
    ):

        simulation_id = state["simulation_id"]

        connection.execute(
            """
            DELETE FROM agent_messages
            WHERE simulation_id = ?
            """,
            (simulation_id,)
        )

        for item in state.get(
            "agent_messages", []
        ):

            connection.execute(
                """
                INSERT INTO agent_messages (
                    simulation_id,
                    sender,
                    receiver,
                    message_type,
                    simulation_time,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    simulation_id,
                    item.get("sender"),
                    item.get("receiver"),
                    item.get("message_type"),
                    item.get("simulation_time"),
                    encode_json(item)
                )
            )

    # ==========================================
    # 10. Retrieve a saved simulation
    # ==========================================

    def load_state(self, simulation_id):

        connection = get_connection(
            self.db_path
        )

        try:

            row = connection.execute(
                """
                SELECT state_json
                FROM simulation_runs
                WHERE simulation_id = ?
                """,
                (simulation_id,)
            ).fetchone()

            if row is None:

                return None

            return json.loads(
                row["state_json"]
            )

        finally:

            connection.close()

    # ==========================================
    # 11. Retrieve simulation history
    # ==========================================

    def list_simulations(self):

        connection = get_connection(
            self.db_path
        )

        try:

            rows = connection.execute(
                """
                SELECT
                    simulation_id,
                    simulation_time,
                    scenario_mode,
                    workflow_status,
                    approval_status,
                    created_at,
                    updated_at
                FROM simulation_runs
                ORDER BY updated_at DESC
                """
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            connection.close()

    # ==========================================
    # 12. Retrieve individual table records
    # ==========================================

    def get_records(
        self,
        table_name,
        simulation_id
    ):

        allowed_tables = {

            "patient_queue",
            "bed_inventory",
            "discharge_predictions",
            "bed_allocations",
            "operational_alerts",
            "audit_logs",
            "agent_messages"
        }

        if table_name not in allowed_tables:

            raise ValueError(
                "Invalid database table."
            )

        connection = get_connection(
            self.db_path
        )

        try:

            rows = connection.execute(
                f"""
                SELECT *
                FROM {table_name}
                WHERE simulation_id = ?
                """,
                (simulation_id,)
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            connection.close()

    # ==========================================
    # 13. Retrieve database summary
    # ==========================================

    def get_database_summary(self):

        tables = [

            "simulation_runs",
            "patient_queue",
            "bed_inventory",
            "discharge_predictions",
            "bed_allocations",
            "operational_alerts",
            "audit_logs",
            "agent_messages"
        ]

        connection = get_connection(
            self.db_path
        )

        try:

            return {

                table: connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]

                for table in tables
            }

        finally:

            connection.close()
