
# Automated Tests for Agent 4

import pytest

from core.state import create_initial_state

from agents.coordination_agent import (
    CoordinationAgent
)


def make_state():

    state = create_initial_state(
        "2031-11-07 12:00:00"
    )

    state["beds"] = [

        {
            "bed_id": "B001",
            "bed_type": "medical ward",
            "status": "Available",
            "allocatable": True
        }
    ]

    patient = {

        "patient_id": "P001",

        "encounter_id": "E001",

        "priority": 1,

        "requested_care": "inpatient",

        "requested_bed_type": "medical ward",

        "waiting_time_minutes": 150
    }

    state["prioritised_queue"] = [patient]

    state["waiting_patients"] = [patient.copy()]

    state["available_beds"] = [
        state["beds"][0].copy()
    ]

    state["allocation_recommendations"] = [

        {
            "patient_id": "P001",
            "encounter_id": "E001",
            "bed_id": "B001",
            "bed_type": "medical ward",
            "allocation_status": "Pending Approval",
            "human_approval_required": True
        }
    ]

    return state


def make_review_state():

    agent = CoordinationAgent()

    state = agent.run(make_state())

    approval_id = state[
        "approval_requests"
    ][0]["approval_id"]

    return agent, state, approval_id


def test_normal_occupancy():

    agent = CoordinationAgent()

    assert agent.monitor_capacity(
        make_state()
    ) == []


def test_high_occupancy():

    agent = CoordinationAgent()

    state = make_state()

    state["beds"] = [

        {"status": "Occupied"}

        for _ in range(9)

    ] + [{"status": "Available"}]

    alerts = agent.monitor_capacity(state)

    assert alerts[0]["severity"] == "Warning"


def test_critical_occupancy():

    agent = CoordinationAgent()

    state = make_state()

    state["beds"] = [

        {"status": "Occupied"}

        for _ in range(10)
    ]

    alerts = agent.monitor_capacity(state)

    assert alerts[0]["severity"] == "Critical"


def test_staffing_alert():

    agent = CoordinationAgent()

    state = make_state()

    state["staffing"] = [

        {
            "department": "ICU",
            "workload_category": "Critical"
        }
    ]

    alerts = agent.monitor_staffing(state)

    assert alerts[0]["severity"] == "Critical"


def test_valid_allocation():

    agent = CoordinationAgent()

    valid, alerts = agent.validate_allocations(
        make_state()
    )

    assert len(valid) == 1

    assert alerts == []


def test_duplicate_bed_blocked():

    agent = CoordinationAgent()

    state = make_state()

    duplicate = state[
        "allocation_recommendations"
    ][0].copy()

    duplicate["patient_id"] = "P002"

    duplicate["encounter_id"] = "E002"

    state["allocation_recommendations"].append(
        duplicate
    )

    valid, alerts = agent.validate_allocations(state)

    assert valid == []

    assert len(alerts) == 2


def test_unavailable_bed_blocked():

    agent = CoordinationAgent()

    state = make_state()

    state["beds"][0]["status"] = "Occupied"

    state["beds"][0]["allocatable"] = False

    valid, alerts = agent.validate_allocations(state)

    assert valid == []

    assert alerts[0]["severity"] == "Critical"


def test_incompatible_bed_blocked():

    agent = CoordinationAgent()

    state = make_state()

    state["prioritised_queue"][0][
        "requested_bed_type"
    ] = "isolation room"

    valid, alerts = agent.validate_allocations(state)

    assert valid == []

    assert len(alerts) == 1


def test_approval_request_created():

    agent, state, approval_id = make_review_state()

    assert len(state["approval_requests"]) == 1

    assert state["approval_required"] is True


def test_approval_updates_bed():

    agent, state, approval_id = make_review_state()

    state = agent.apply_decision(
        state,
        approval_id,
        "approve",
        "Test Reviewer",
        "Compatible bed confirmed."
    )

    assert state["beds"][0]["status"] == "Occupied"

    assert state["beds"][0]["allocatable"] is False

    assert len(state["approved_allocations"]) == 1

    assert state["prioritised_queue"] == []

    assert state["available_beds"] == []


def test_rejection_retains_patient():

    agent, state, approval_id = make_review_state()

    state = agent.apply_decision(
        state,
        approval_id,
        "reject",
        "Test Reviewer",
        "Further review required."
    )

    assert len(state["rejected_allocations"]) == 1

    assert len(state["prioritised_queue"]) == 1

    assert state["beds"][0]["status"] == "Available"


def test_audit_log_created():

    agent, state, approval_id = make_review_state()

    state = agent.apply_decision(
        state,
        approval_id,
        "approve",
        "Test Reviewer",
        "Approved for simulation."
    )

    assert len(state["audit_log"]) == 1

    assert state["audit_log"][0][
        "reviewer"
    ] == "Test Reviewer"


def test_repeat_decision_blocked():

    agent, state, approval_id = make_review_state()

    state = agent.apply_decision(
        state,
        approval_id,
        "approve",
        "Test Reviewer",
        "Approved."
    )

    with pytest.raises(ValueError):

        agent.apply_decision(
            state,
            approval_id,
            "approve",
            "Test Reviewer",
            "Repeated decision."
        )


def test_missing_reviewer_blocked():

    agent, state, approval_id = make_review_state()

    with pytest.raises(ValueError):

        agent.apply_decision(
            state,
            approval_id,
            "approve",
            "",
            "Approved."
        )


def test_stale_bed_approval_blocked():

    agent, state, approval_id = make_review_state()

    state["beds"][0]["status"] = "Occupied"

    state["beds"][0]["allocatable"] = False

    with pytest.raises(ValueError):

        agent.apply_decision(
            state,
            approval_id,
            "approve",
            "Test Reviewer",
            "Approved."
        )


def test_no_approval_required():

    agent = CoordinationAgent()

    state = create_initial_state(
        "2031-11-07 12:00:00"
    )

    state = agent.run(state)

    assert state["approval_required"] is False

    assert state["approval_status"] == "Not Required"
