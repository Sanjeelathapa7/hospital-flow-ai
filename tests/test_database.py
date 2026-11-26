
# HospitalFlow AI
# Automated Database Tests

import sqlite3

import pytest

from core.state import create_initial_state

from database.connection import (
    initialise_database,
    get_table_names,
    get_connection
)

from database.repository import (
    HospitalRepository
)


@pytest.fixture
def repository(tmp_path):

    return HospitalRepository(
        tmp_path / "test_hospital.db"
    )


def make_state():

    state = create_initial_state(
        simulation_time="2031-11-07 12:00:00",
        simulation_id="TEST-DB-001"
    )

    state["scenario_mode"] = "normal"

    state["workflow_status"] = "Running"

    state["beds"] = [

        {
            "bed_id": "B001",
            "bed_type": "medical ward",
            "status": "Available",
            "allocatable": True
        }
    ]

    state["prioritised_queue"] = [

        {
            "patient_id": "P001",
            "encounter_id": "E001",
            "priority": 1,
            "queue_position": 1,
            "waiting_time_minutes": 60,
            "requested_bed_type": "medical ward"
        }
    ]

    state["approval_requests"] = [

        {
            "approval_id": "APR-001",
            "patient_id": "P001",
            "encounter_id": "E001",
            "bed_id": "B001",
            "bed_type": "medical ward",
            "approval_status": "Pending",
            "reviewer": None,
            "decision_reason": None,
            "decision_time": None
        }
    ]

    return state


def approve_state(state):

    request = state["approval_requests"][0]

    request.update({

        "approval_status": "Approved",

        "reviewer": "Test Reviewer",

        "decision_reason": "Compatible bed.",

        "decision_time":
            "2026-11-26T12:00:00+00:00"
    })

    state["approved_allocations"] = [
        request.copy()
    ]

    state["beds"][0].update({

        "status": "Occupied",

        "allocatable": False,

        "assigned_patient_id": "P001",

        "assigned_encounter_id": "E001"
    })

    state["prioritised_queue"] = []

    state["workflow_status"] = "Completed"

    state["approval_status"] = "Completed"

    state["audit_log"] = [

        {
            "event_type":
                "HUMAN_ALLOCATION_DECISION",

            "approval_id": "APR-001",

            "patient_id": "P001",

            "encounter_id": "E001",

            "bed_id": "B001",

            "decision": "Approved",

            "reviewer": "Test Reviewer",

            "reason": "Compatible bed.",

            "decision_time":
                "2026-11-26T12:00:00+00:00"
        }
    ]

    return state


def test_database_creation(tmp_path):

    path = tmp_path / "hospital.db"

    initialise_database(path)

    assert path.exists()


def test_all_tables_created(repository):

    tables = get_table_names(
        repository.db_path
    )

    assert len(tables) == 8


def test_save_and_load_state(repository):

    state = make_state()

    repository.save_state(state)

    saved = repository.load_state(
        state["simulation_id"]
    )

    assert saved["simulation_id"] == (
        state["simulation_id"]
    )


def test_patient_records(repository):

    state = make_state()

    repository.save_state(state)

    records = repository.get_records(
        "patient_queue",
        state["simulation_id"]
    )

    assert len(records) == 1

    assert records[0]["priority"] == 1


def test_bed_records(repository):

    state = make_state()

    repository.save_state(state)

    records = repository.get_records(
        "bed_inventory",
        state["simulation_id"]
    )

    assert len(records) == 1

    assert records[0]["status"] == "Available"


def test_approval_persistence(repository):

    state = make_state()

    repository.save_state(state)

    state = approve_state(state)

    repository.save_state(state)

    records = repository.get_records(
        "bed_allocations",
        state["simulation_id"]
    )

    assert records[0]["approval_status"] == (
        "Approved"
    )


def test_audit_persistence(repository):

    state = approve_state(
        make_state()
    )

    repository.save_state(state)

    records = repository.get_records(
        "audit_logs",
        state["simulation_id"]
    )

    assert len(records) == 1

    assert records[0]["reviewer"] == (
        "Test Reviewer"
    )


def test_repeated_save_no_duplicates(repository):

    state = approve_state(
        make_state()
    )

    repository.save_state(state)

    repository.save_state(state)

    summary = repository.get_database_summary()

    assert summary["simulation_runs"] == 1

    assert summary["bed_allocations"] == 1

    assert summary["audit_logs"] == 1


def test_duplicate_approved_bed_blocked(repository):

    state = approve_state(
        make_state()
    )

    duplicate = state[
        "approval_requests"
    ][0].copy()

    duplicate["approval_id"] = "APR-002"

    duplicate["patient_id"] = "P002"

    duplicate["encounter_id"] = "E002"

    state["approval_requests"].append(
        duplicate
    )

    with pytest.raises(sqlite3.IntegrityError):

        repository.save_state(state)

    # The entire failed transaction rolls back.

    assert repository.load_state(
        state["simulation_id"]
    ) is None


def test_completed_decision_immutable(repository):

    state = approve_state(
        make_state()
    )

    repository.save_state(state)

    state["approval_requests"][0][
        "approval_status"
    ] = "Rejected"

    with pytest.raises(ValueError):

        repository.save_state(state)

    records = repository.get_records(
        "bed_allocations",
        state["simulation_id"]
    )

    assert records[0]["approval_status"] == (
        "Approved"
    )


def test_missing_simulation(repository):

    assert repository.load_state(
        "DOES-NOT-EXIST"
    ) is None


def test_invalid_table_blocked(repository):

    with pytest.raises(ValueError):

        repository.get_records(
            "invalid_table",
            "TEST-DB-001"
        )


def test_simulation_history(repository):

    repository.save_state(
        make_state()
    )

    history = repository.list_simulations()

    assert len(history) == 1


def test_foreign_key_protection(repository):

    connection = get_connection(
        repository.db_path
    )

    try:

        with pytest.raises(sqlite3.IntegrityError):

            connection.execute(
                """
                INSERT INTO bed_inventory (
                    simulation_id,
                    bed_id,
                    bed_type,
                    status,
                    allocatable
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "UNKNOWN-SIM",
                    "B001",
                    "medical ward",
                    "Available",
                    1
                )
            )

    finally:

        connection.close()
