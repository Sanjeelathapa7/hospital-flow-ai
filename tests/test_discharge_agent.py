
# Automated Tests for Agent 3

import numpy as np
import pandas as pd
import pytest

from core.state import create_initial_state

from agents.discharge_agent import (
    DischargePredictionAgent
)


class MockModel:

    def __init__(self, prediction=60.0):

        self.prediction = prediction

    def predict(self, features):

        return np.array([
            self.prediction
        ])


def make_agent(prediction=60.0):

    patients = pd.DataFrame({

        "patient_id": ["P001"],

        "age": [45],

        "clinical_complexity_category": [
            "Moderate"
        ]
    })

    config = {

        "numeric_features": [

            "age",
            "admission_hour",
            "admission_day_of_week",
            "admission_month",
            "is_weekend"
        ],

        "categorical_features": [

            "clinical_complexity_category"
        ]
    }

    return DischargePredictionAgent(

        model=MockModel(prediction),

        metadata={
            "model_name": "Test Model"
        },

        feature_config=config,

        patient_data=patients
    )


def make_patient(patient_id="P001"):

    return {

        "patient_id": patient_id,

        "encounter_id": "E001",

        "priority": 2,

        "waiting_time_minutes": 90
    }


def test_feature_preparation():

    agent = make_agent()

    features = agent.prepare_features(
        "P001",
        "2031-10-03 12:00:00"
    )

    assert features.shape == (1, 6)

    assert list(features.columns) == (
        agent.features
    )

    assert features.iloc[0][
        "admission_hour"
    ] == 12


def test_prediction_hours():

    agent = make_agent()

    result = agent.predict(
        make_patient(),
        "2031-10-03 12:00:00"
    )

    assert result[
        "predicted_los_hours"
    ] == 60.0


def test_prediction_days():

    agent = make_agent()

    result = agent.predict(
        make_patient(),
        "2031-10-03 12:00:00"
    )

    assert result[
        "predicted_los_days"
    ] == 2.5


def test_projected_stay_end():

    agent = make_agent()

    result = agent.predict(
        make_patient(),
        "2031-10-03 12:00:00"
    )

    assert result[
        "projected_stay_end"
    ] == "2031-10-06T00:00:00"


def test_negative_prediction_clipped():

    agent = make_agent(
        prediction=-10
    )

    result = agent.predict(
        make_patient(),
        "2031-10-03 12:00:00"
    )

    assert result[
        "predicted_los_hours"
    ] == 0.0


def test_invalid_prediction():

    agent = make_agent(
        prediction=np.nan
    )

    with pytest.raises(ValueError):

        agent.predict(
            make_patient(),
            "2031-10-03 12:00:00"
        )


def test_missing_patient():

    agent = make_agent()

    with pytest.raises(ValueError):

        agent.predict(
            make_patient("UNKNOWN"),
            "2031-10-03 12:00:00"
        )


def test_human_review_required():

    agent = make_agent()

    result = agent.predict(
        make_patient(),
        "2031-10-03 12:00:00"
    )

    assert result[
        "human_review_required"
    ] is True

    assert result[
        "autonomous_discharge_allowed"
    ] is False


def test_shared_state_update():

    agent = make_agent()

    state = create_initial_state(
        "2031-10-03 12:00:00"
    )

    state["prioritised_queue"] = [
        make_patient()
    ]

    result = agent.run(state)

    assert len(
        result["discharge_predictions"]
    ) == 1

    assert "AGENT-03" in (
        result["completed_agents"]
    )

    assert result["metrics"][
        "discharge_prediction"
    ]["successful_predictions"] == 1


def test_empty_queue():

    agent = make_agent()

    state = create_initial_state(
        "2031-10-03 12:00:00"
    )

    result = agent.run(state)

    assert result[
        "discharge_predictions"
    ] == []

    assert result["metrics"][
        "discharge_prediction"
    ]["successful_predictions"] == 0


def test_failed_prediction_alert():

    agent = make_agent()

    state = create_initial_state(
        "2031-10-03 12:00:00"
    )

    state["prioritised_queue"] = [

        make_patient("UNKNOWN")
    ]

    result = agent.run(state)

    assert result["metrics"][
        "discharge_prediction"
    ]["failed_predictions"] == 1

    assert any(

        alert["alert_type"]
        == "PREDICTION_UNAVAILABLE"

        for alert in result["alerts"]
    )


def test_agent_does_not_release_beds():

    agent = make_agent()

    state = create_initial_state(
        "2031-10-03 12:00:00"
    )

    state["beds"] = [

        {
            "bed_id": "B001",
            "status": "Occupied",
            "allocatable": False
        }
    ]

    state["prioritised_queue"] = [
        make_patient()
    ]

    result = agent.run(state)

    assert result["beds"][0][
        "status"
    ] == "Occupied"

    assert result["beds"][0][
        "allocatable"
    ] is False
