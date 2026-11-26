
# HospitalFlow AI
# Three-Agent Integration Demonstration

from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from core.config import DEFAULT_SIMULATION_TIME
from core.state import create_initial_state
from core.environment import HospitalEnvironment
from core.data_loader import load_patients

from agents.admission_agent import (
    AdmissionPrioritisationAgent
)

from agents.discharge_agent import (
    DischargePredictionAgent
)

from agents.bed_allocation_agent import (
    BedAllocationAgent
)


OUTPUT_DIR = (
    BASE_DIR / "outputs" / "agents" / "discharge"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("HOSPITALFLOW AI")
print("THREE-AGENT INTEGRATION")
print("=" * 60)

# ==========================================
# 1. Initialise hospital environment
# ==========================================

state = create_initial_state(
    simulation_time=DEFAULT_SIMULATION_TIME
)

environment = HospitalEnvironment()

state = environment.initialise(state)

print(
    "\nSimulation time:",
    state["simulation_time"]
)

# ==========================================
# 2. Execute Agent 1
# ==========================================

admission_agent = (
    AdmissionPrioritisationAgent()
)

state = admission_agent.run(state)

print("\nAGENT 1 METRICS")

print(
    json.dumps(
        state["metrics"]["admission"],
        indent=4
    )
)

# ==========================================
# 3. Execute Agent 3
# ==========================================

discharge_agent = (
    DischargePredictionAgent()
)

print("\nMODEL INFORMATION")

print(
    "Model:",
    discharge_agent.metadata.get("model_name")
)

print(
    "Input features:",
    len(discharge_agent.features)
)

print(
    "Historical test metrics:",
    discharge_agent.metadata.get("test_metrics")
)

state = discharge_agent.run(state)

print("\nAGENT 3 METRICS")

print(
    json.dumps(
        state["metrics"]["discharge_prediction"],
        indent=4
    )
)

predictions = pd.DataFrame(
    state["discharge_predictions"]
)

print("\nLENGTH OF STAY ESTIMATES")

if not predictions.empty:

    columns = [

        "patient_id",
        "encounter_id",
        "predicted_los_hours",
        "predicted_los_days",
        "projected_stay_end",
        "reliability"
    ]

    print(
        predictions[columns]
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "No successful predictions for "
        "the current admission queue."
    )

# ==========================================
# 4. Execute Agent 2
# ==========================================

allocation_agent = BedAllocationAgent()

state = allocation_agent.run(state)

print("\nAGENT 2 METRICS")

print(
    json.dumps(
        state["metrics"]["bed_allocation"],
        indent=4
    )
)

# ==========================================
# 5. Display inter-agent communication
# ==========================================

print("\nAGENT COMMUNICATION")

for message in state["agent_messages"]:

    print(
        json.dumps(
            message,
            indent=4
        )
    )

print("\nCOMPLETED AGENTS")

print(state["completed_agents"])

# ==========================================
# 6. Save historical snapshot results
# ==========================================

predictions.to_csv(
    OUTPUT_DIR / "discharge_predictions.csv",
    index=False
)

with open(
    OUTPUT_DIR / "discharge_metrics.json",
    "w"
) as file:

    json.dump(
        state["metrics"]["discharge_prediction"],
        file,
        indent=4
    )

with open(
    OUTPUT_DIR / "three_agent_state.json",
    "w"
) as file:

    json.dump(
        state,
        file,
        indent=4
    )

# ==========================================
# 7. Controlled prediction demonstration
# ==========================================

print("\nCONTROLLED PREDICTION DEMONSTRATION")

patients = load_patients().head(5)

scenario_state = create_initial_state(
    simulation_time=DEFAULT_SIMULATION_TIME,
    simulation_id="SIM-PREDICTION-001"
)

scenario_state["prioritised_queue"] = [

    {

        "patient_id":
            str(patient["patient_id"]),

        "encounter_id":
            f"SIM-PRED-{index + 1}",

        "priority": 3,

        "waiting_time_minutes": 60,

        "requested_care": "inpatient"
    }

    for index, (_, patient)
    in enumerate(patients.iterrows())
]

scenario_state = discharge_agent.run(
    scenario_state
)

scenario_predictions = pd.DataFrame(
    scenario_state["discharge_predictions"]
)

assert len(scenario_predictions) == 5

assert (
    scenario_predictions[
        "predicted_los_hours"
    ] >= 0
).all()

assert (
    scenario_predictions[
        "human_review_required"
    ]
).all()

assert not (
    scenario_predictions[
        "autonomous_discharge_allowed"
    ]
).any()

print(
    scenario_predictions[
        [
            "patient_id",
            "predicted_los_hours",
            "predicted_los_days",
            "projected_stay_end"
        ]
    ].to_string(index=False)
)

scenario_predictions.to_csv(
    OUTPUT_DIR / "controlled_predictions.csv",
    index=False
)

with open(
    OUTPUT_DIR / "controlled_prediction_state.json",
    "w"
) as file:

    json.dump(
        scenario_state,
        file,
        indent=4
    )

print("\nCONTROLLED PREDICTION TEST PASSED")

print("\nOutputs saved to:", OUTPUT_DIR)

print("\nAGENT 3 EXECUTION COMPLETED")
