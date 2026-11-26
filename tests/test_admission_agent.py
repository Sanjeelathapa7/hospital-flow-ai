
# Automated Tests for Agent 1

import pandas as pd

from agents.admission_agent import (
    AdmissionPrioritisationAgent
)


def create_test_patients():

    return pd.DataFrame({

        "patient_id": [
            "TEST-001",
            "TEST-002",
            "TEST-003",
            "TEST-004",
            "TEST-005"
        ],

        "encounter_id": [
            "ENC-001",
            "ENC-002",
            "ENC-003",
            "ENC-004",
            "ENC-005"
        ],

        "priority": [3, 1, 2, 1, 5],

        "waiting_time_minutes": [
            180, 30, 250, 90, 10
        ],

        "arrival_time": pd.to_datetime([
            "2031-10-03 08:00:00",
            "2031-10-03 11:00:00",
            "2031-10-03 07:00:00",
            "2031-10-03 10:00:00",
            "2031-10-03 11:30:00"
        ]),

        "disposition": [
            "inpatient admission",
            "icu admission",
            "inpatient admission",
            "icu admission",
            "inpatient admission"
        ]
    })


def test_priority_order():

    agent = AdmissionPrioritisationAgent()

    queue = agent.reason(
        create_test_patients()
    )

    assert queue["priority"].tolist() == [
        1, 1, 2, 3, 5
    ]


def test_waiting_time_tie_break():

    agent = AdmissionPrioritisationAgent()

    queue = agent.reason(
        create_test_patients()
    )

    assert queue.iloc[0]["patient_id"] == (
        "TEST-004"
    )

    assert queue.iloc[1]["patient_id"] == (
        "TEST-002"
    )


def test_warning_alert():

    agent = AdmissionPrioritisationAgent()

    queue = agent.reason(
        create_test_patients()
    )

    patient = queue.loc[
        queue["patient_id"] == "TEST-001"
    ].iloc[0]

    assert patient["waiting_status"] == (
        "Warning"
    )


def test_critical_alert():

    agent = AdmissionPrioritisationAgent()

    queue = agent.reason(
        create_test_patients()
    )

    patient = queue.loc[
        queue["patient_id"] == "TEST-003"
    ].iloc[0]

    assert patient["waiting_status"] == (
        "Critical"
    )


def test_alert_generation():

    agent = AdmissionPrioritisationAgent()

    queue = agent.reason(
        create_test_patients()
    )

    alerts = agent.generate_alerts(queue)

    assert len(alerts) == 2


def test_metrics():

    agent = AdmissionPrioritisationAgent()

    queue = agent.reason(
        create_test_patients()
    )

    metrics = agent.calculate_metrics(queue)

    assert metrics["waiting_patients"] == 5

    assert metrics[
        "highest_urgency_patients"
    ] == 2

    assert metrics["icu_requests"] == 2

    assert metrics[
        "average_waiting_minutes"
    ] == 112.0


def test_empty_queue():

    agent = AdmissionPrioritisationAgent()

    empty = create_test_patients().iloc[0:0]

    queue = agent.reason(empty)

    metrics = agent.calculate_metrics(queue)

    assert queue.empty

    assert metrics["waiting_patients"] == 0

    assert metrics[
        "average_waiting_minutes"
    ] == 0.0


def test_priority_is_not_modified():

    agent = AdmissionPrioritisationAgent()

    original = create_test_patients()

    queue = agent.reason(original)

    expected = original.set_index(
        "patient_id"
    )["priority"].sort_index()

    actual = queue.set_index(
        "patient_id"
    )["priority"].sort_index()

    assert expected.equals(actual)
