
# HospitalFlow AI
# Four-Agent Integration Demonstration

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

from agents.coordination_agent import (
    CoordinationAgent
)


OUTPUT_DIR = (
    BASE_DIR / "outputs" / "agents" / "coordination"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_json(filename, data):

    with open(OUTPUT_DIR / filename, "w") as file:

        json.dump(data, file, indent=4)


print("=" * 60)
print("HOSPITALFLOW AI")
print("FOUR-AGENT INTEGRATION")
print("=" * 60)

# ==========================================
# 1. Historical hospital snapshot
# ==========================================

state = create_initial_state(
    simulation_time=DEFAULT_SIMULATION_TIME
)

environment = HospitalEnvironment()

admission_agent = AdmissionPrioritisationAgent()

discharge_agent = DischargePredictionAgent()

allocation_agent = BedAllocationAgent()

coordination_agent = CoordinationAgent()

state = environment.initialise(state)

state = admission_agent.run(state)

state = discharge_agent.run(state)

state = allocation_agent.run(state)

state = coordination_agent.run(state)

print("\nHISTORICAL SNAPSHOT METRICS")

print(
    json.dumps(state["metrics"], indent=4)
)

print("\nCOMPLETED AGENTS")

print(state["completed_agents"])

print("\nHISTORICAL ALERTS")

print(
    json.dumps(state["alerts"], indent=4)
)

save_json(
    "historical_four_agent_state.json",
    state
)

# ==========================================
# 2. Controlled hospital scenario
# ==========================================

print("\n" + "=" * 60)
print("CONTROLLED HOSPITAL SCENARIO")
print("=" * 60)

# Use real profiles from the synthetic dataset.
# Admission requests and bed requirements
# are explicitly fictional scenario inputs.

profiles = load_patients().head(3)

patient_ids = (
    profiles["patient_id"].astype(str).tolist()
)

scenario = create_initial_state(
    simulation_time="2031-11-07 12:00:00",
    simulation_id="SIM-COORD-001"
)

scenario["beds"] = [

    {
        "bed_id": "SIM-ICU-001",
        "bed_type": "ICU",
        "status": "Available",
        "allocatable": True
    },

    {
        "bed_id": "SIM-MEDW-001",
        "bed_type": "medical ward",
        "status": "Available",
        "allocatable": True
    }

]

# Add 18 occupied beds to simulate pressure.

for index in range(18):

    scenario["beds"].append({

        "bed_id": f"SIM-OCC-{index + 1:03d}",

        "bed_type": "medical ward",

        "status": "Occupied",

        "allocatable": False
    })

scenario["staffing"] = [

    {
        "department": "Emergency Department",
        "workload_category": "High"
    },

    {
        "department": "ICU",
        "workload_category": "Critical"
    }
]

# ==========================================
# 3. Agent 1 controlled reasoning
# ==========================================

queue = pd.DataFrame({

    "patient_id": patient_ids,

    "encounter_id": [
        "SIM-E001",
        "SIM-E002",
        "SIM-E003"
    ],

    "priority": [1, 2, 3],

    "waiting_time_minutes": [
        90, 150, 260
    ],

    "arrival_time": pd.to_datetime([
        "2031-11-07 09:00:00",
        "2031-11-07 08:00:00",
        "2031-11-07 06:00:00"
    ]),

    "disposition": [
        "icu admission",
        "inpatient admission",
        "inpatient admission"
    ],

    "requested_bed_type": [
        "ICU",
        "medical ward",
        "medical ward"
    ],

    "requirement_source": [
        "Synthetic scenario input",
        "Synthetic scenario input",
        "Synthetic scenario input"
    ]
})

prioritised = admission_agent.reason(queue)

scenario["waiting_patients"] = (
    admission_agent.to_records(queue)
)

scenario["prioritised_queue"] = (
    admission_agent.to_records(prioritised)
)

scenario["alerts"] = (
    admission_agent.generate_alerts(prioritised)
)

scenario["metrics"]["admission"] = (
    admission_agent.calculate_metrics(prioritised)
)

scenario["completed_agents"].append("AGENT-01")

# ==========================================
# 4. Execute Agents 3, 2 and 4
# ==========================================

scenario = discharge_agent.run(scenario)

scenario = allocation_agent.run(scenario)

scenario = coordination_agent.run(scenario)

print("\nCONTROLLED SCENARIO METRICS")

print(
    json.dumps(scenario["metrics"], indent=4)
)

print("\nAPPROVAL REQUESTS")

for request in scenario["approval_requests"]:

    print(
        json.dumps(request, indent=4)
    )

assert len(
    scenario["approval_requests"]
) == 2

assert len(
    scenario["unallocated_patients"]
) == 1

assert len(
    scenario["discharge_predictions"]
) == 3

save_json(
    "scenario_before_approval.json",
    scenario
)

# ==========================================
# 5. Simulate explicit human decisions
# ==========================================

requests = scenario["approval_requests"]

first_id = requests[0]["approval_id"]

second_id = requests[1]["approval_id"]

# These are scripted demonstration decisions.
# Actual interactive review comes in Streamlit.

scenario = coordination_agent.apply_decision(

    scenario,

    approval_id=first_id,

    decision="approve",

    reviewer="Simulation Administrator",

    reason="Compatible ICU bed confirmed."
)

scenario = coordination_agent.apply_decision(

    scenario,

    approval_id=second_id,

    decision="reject",

    reviewer="Simulation Administrator",

    reason="Additional operational review required."
)

# ==========================================
# 6. Verify approval outcomes
# ==========================================

assert len(
    scenario["approved_allocations"]
) == 1

assert len(
    scenario["rejected_allocations"]
) == 1

assert len(
    scenario["audit_log"]
) == 2

assert scenario["approval_status"] == "Completed"

approved = scenario["approved_allocations"][0]

approved_bed = next(

    bed for bed in scenario["beds"]

    if bed["bed_id"] == approved["bed_id"]
)

assert approved_bed["status"] == "Occupied"

assert approved_bed["allocatable"] is False

print("\nFINAL COORDINATION METRICS")

print(
    json.dumps(
        scenario["metrics"]["coordination"],
        indent=4
    )
)

print("\nAUDIT LOG")

print(
    json.dumps(
        scenario["audit_log"],
        indent=4
    )
)

save_json(
    "scenario_after_approval.json",
    scenario
)

pd.DataFrame(
    scenario["alerts"]
).to_csv(
    OUTPUT_DIR / "coordination_alerts.csv",
    index=False
)

pd.DataFrame(
    scenario["recommendations"]
).to_csv(
    OUTPUT_DIR / "operational_recommendations.csv",
    index=False
)

pd.DataFrame(
    scenario["approval_requests"]
).to_csv(
    OUTPUT_DIR / "approval_requests.csv",
    index=False
)

pd.DataFrame(
    scenario["audit_log"]
).to_csv(
    OUTPUT_DIR / "audit_log.csv",
    index=False
)

print("\nCONTROLLED APPROVAL SCENARIO PASSED")

print("\nOutputs saved to:", OUTPUT_DIR)

print("\nALL FOUR AGENTS EXECUTED SUCCESSFULLY")
