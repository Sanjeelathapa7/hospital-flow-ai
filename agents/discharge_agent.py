
# HospitalFlow AI
# Agent 3: Discharge Prediction Agent

import json
import math

import joblib
import numpy as np
import pandas as pd

from core.config import (
    ML_DIR,
    MODEL_DIR,
    OPERATIONAL_MODEL_PATH
)

from core.state import HospitalState
from core.data_loader import load_patients


class DischargePredictionAgent:

    def __init__(
        self,
        model=None,
        metadata=None,
        feature_config=None,
        patient_data=None
    ):

        self.name = "Discharge Prediction Agent"

        self.agent_id = "AGENT-03"

        self.goal = (
            "Estimate inpatient stay duration "
            "to support hospital resource planning."
        )

        # Optional dependencies allow unit testing
        # without loading the full dataset.

        if model is None:

            if not OPERATIONAL_MODEL_PATH.exists():

                raise FileNotFoundError(
                    f"Model not found: "
                    f"{OPERATIONAL_MODEL_PATH}"
                )

            self.model = joblib.load(
                OPERATIONAL_MODEL_PATH
            )

        else:

            self.model = model

        if metadata is None:

            with open(
                MODEL_DIR / "model_metadata.json"
            ) as file:

                self.metadata = json.load(file)

        else:

            self.metadata = metadata

        if feature_config is None:

            with open(
                ML_DIR / "feature_config.json"
            ) as file:

                self.feature_config = json.load(file)

        else:

            self.feature_config = feature_config

        self.numeric_features = (
            self.feature_config["numeric_features"]
        )

        self.categorical_features = (
            self.feature_config["categorical_features"]
        )

        self.features = (
            self.numeric_features
            + self.categorical_features
        )

        if patient_data is None:

            self.patient_data = (
                load_patients()
                .set_index("patient_id")
            )

        else:

            self.patient_data = (
                patient_data.set_index("patient_id")
            )

        if not self.patient_data.index.is_unique:

            raise ValueError(
                "Patient IDs must be unique."
            )

        trained_features = getattr(
            self.model,
            "feature_names_in_",
            None
        )

        if trained_features is not None:

            if list(trained_features) != self.features:

                raise ValueError(
                    "Model input features do not "
                    "match the saved feature configuration."
                )

    # ==========================================
    # 1. Perceive admission requests
    # ==========================================

    def perceive(self, state):

        return state.get(
            "prioritised_queue", []
        )

    # ==========================================
    # 2. Prepare model input features
    # ==========================================

    def prepare_features(
        self,
        patient_id,
        assumed_admission_time
    ):

        patient_id = str(patient_id)

        if patient_id not in self.patient_data.index:

            raise ValueError(
                f"Patient profile not found: "
                f"{patient_id}"
            )

        patient = self.patient_data.loc[
            patient_id
        ].to_dict()

        admission = pd.Timestamp(
            assumed_admission_time
        )

        if pd.isna(admission):

            raise ValueError(
                "Invalid assumed admission time."
            )

        patient["admission_hour"] = (
            admission.hour
        )

        patient["admission_day_of_week"] = (
admission.day_of_week        )

        patient["admission_month"] = (
            admission.month
        )

        patient["is_weekend"] = int(
            admission.day_of_week >= 5
        )

        missing_features = [

            feature
            for feature in self.features
            if feature not in patient
        ]

        if missing_features:

            raise ValueError(
                f"Missing model features: "
                f"{missing_features}"
            )

        features = pd.DataFrame([

            {
                feature: patient[feature]
                for feature in self.features
            }

        ])

        for column in self.numeric_features:

            features[column] = pd.to_numeric(
                features[column],
                errors="coerce"
            )

            features[column] = (
                features[column].replace(
                    [np.inf, -np.inf],
                    np.nan
                )
            )

        for column in self.categorical_features:

            features[column] = (
                features[column]
                .astype(object)
                .where(
                    features[column].notna(),
                    np.nan
                )
            )

        return features[self.features]

    # ==========================================
    # 3. Estimate inpatient length of stay
    # ==========================================

    def predict(
        self,
        patient,
        simulation_time
    ):

        patient_id = str(
            patient["patient_id"]
        )

        # The patient is still waiting for a bed.
        # Admission at the simulation time is
        # an explicit planning assumption.

        assumed_admission_time = pd.Timestamp(
            simulation_time
        )

        features = self.prepare_features(
            patient_id,
            assumed_admission_time
        )

        raw_prediction = float(
            self.model.predict(features)[0]
        )

        if not math.isfinite(raw_prediction):

            raise ValueError(
                "Model returned a non-finite prediction."
            )

        predicted_hours = max(
            0.0,
            raw_prediction
        )

        predicted_days = (
            predicted_hours / 24
        )

        projected_stay_end = (
            assumed_admission_time
            + pd.Timedelta(
                hours=predicted_hours
            )
        )

        return {

            "patient_id": patient_id,

            "encounter_id":
                str(patient["encounter_id"]),

            "model_name":
                self.metadata.get(
                    "model_name",
                    "Operational baseline"
                ),

            "assumed_admission_time":
                assumed_admission_time.isoformat(),

            "predicted_los_hours":
                round(predicted_hours, 2),

            "predicted_los_days":
                round(predicted_days, 2),

            "projected_stay_end":
                projected_stay_end.isoformat(),

            "prediction_status":
                "Planning Estimate",

            "prediction_scope":
                "Total inpatient length of stay",

            "reliability":
                "Limited",

            "human_review_required":
                True,

            "autonomous_discharge_allowed":
                False,

            "warning":
                "Population-level planning estimate. "
                "Not a clinical discharge decision."
        }

    # ==========================================
    # 4. Calculate agent metrics
    # ==========================================

    def calculate_metrics(
        self,
        predictions,
        failures
    ):

        hours = [

            prediction["predicted_los_hours"]

            for prediction in predictions
        ]

        return {

            "model_name":
                self.metadata.get(
                    "model_name",
                    "Operational baseline"
                ),

            "successful_predictions":
                len(predictions),

            "failed_predictions":
                len(failures),

            "average_predicted_los_hours":
                round(
                    float(np.mean(hours)), 2
                ) if hours else 0.0,

            "average_predicted_los_days":
                round(
                    float(np.mean(hours)) / 24,
                    2
                ) if hours else 0.0,

            "minimum_predicted_los_hours":
                min(hours) if hours else 0.0,

            "maximum_predicted_los_hours":
                max(hours) if hours else 0.0,

            "prediction_reliability":
                "Limited",

            "human_review_required":
                True
        }

    # ==========================================
    # 5. Execute Agent 3
    # ==========================================

    def run(
        self,
        state: HospitalState
    ) -> HospitalState:

        simulation_time = state[
            "simulation_time"
        ]

        patients = self.perceive(state)

        predictions = []

        failures = []

        for patient in patients:

            try:

                prediction = self.predict(
                    patient,
                    simulation_time
                )

                predictions.append(prediction)

            except (
                ValueError,
                KeyError,
                TypeError,
                OverflowError
            ) as error:

                failures.append({

                    "agent_id":
                        self.agent_id,

                    "alert_type":
                        "PREDICTION_UNAVAILABLE",

                    "severity":
                        "Warning",

                    "patient_id":
                        patient.get("patient_id"),

                    "encounter_id":
                        patient.get("encounter_id"),

                    "message":
                        str(error)
                })

        model_warning = {

            "agent_id": self.agent_id,

            "alert_type":
                "MODEL_RELIABILITY_LIMITATION",

            "severity":
                "Warning",

            "message":
                "The operational median baseline "
                "has limited individual predictive "
                "accuracy. Estimates must not trigger "
                "patient discharge or bed release."
        }

        state["discharge_predictions"] = (
            predictions
        )

        state["alerts"] = [

            alert
            for alert in state.get("alerts", [])

            if alert.get("agent_id")
            != self.agent_id

        ] + failures + [model_warning]

        state.setdefault(
            "metrics", {}
        )["discharge_prediction"] = (
            self.calculate_metrics(
                predictions,
                failures
            )
        )

        for receiver in [
            "AGENT-02",
            "AGENT-04"
        ]:

            state.setdefault(
                "agent_messages", []
            ).append({

                "sender":
                    self.agent_id,

                "receiver":
                    receiver,

                "message_type":
                    "LENGTH_OF_STAY_ESTIMATES",

                "prediction_count":
                    len(predictions),

                "failure_count":
                    len(failures),

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

def discharge_agent_node(
    state: HospitalState
) -> HospitalState:

    agent = DischargePredictionAgent()

    return agent.run(state)
