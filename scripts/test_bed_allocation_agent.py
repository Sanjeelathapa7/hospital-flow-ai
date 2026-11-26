
# HospitalFlow AI
# Agent 1 and Agent 2 Integration Demonstration

from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from core.config import DEFAULT_SIMULATION_TIME

from core.state import create_initial_state

from core.environment import HospitalEnvironment

from agents.admission_agent import (
    AdmissionPrioritisationAgent
)

from agents.bed_allocation_agent import (
    BedAllocationAgent
)


OUTPUT_DIR = (
    BASE_DIR / "outputs" / "agents" / "bed_allocation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("HOSPITALFLOW AI")
print("AGENT 1 AND AGENT 2 INTEGRATION")
print("=" * 60)

# ==========================================
# 1. Create shared hospital state
# ==========================================

state = create_initial_state(
    simulation_time=DEFAULT_SIMULATION_TIME
)

print(
    "\nSimulation time:",
    state["simulation_time"]
)

# ==========================================
# 2. Initialise hospital environment
# ==========================================

environment = HospitalEnvironment()

state = environment.initialise(state)

print("\nHOSPITAL ENVIRONMENT METRICS")

for key, value in state[
    "metrics"
]["environment"].items():

    print(f"{key}: {value}")

beds = pd.DataFrame(state["beds"])

print("\nBED STATUS BY BED TYPE")

print(
    pd.crosstab(
        beds["bed_type"],
        beds["status"]
    ).to_string()
)

# ==========================================
# 3. Execute Agent 1
# ==========================================

admission_agent = (
    AdmissionPrioritisationAgent()
)

state = admission_agent.run(state)

print("\nAGENT 1 METRICS")

for key, value in state[
    "metrics"
]["admission"].items():

    print(f"{key}: {value}")

# ==========================================
# 4. Execute Agent 2
# ==========================================

allocation_agent = BedAllocationAgent()

state = allocation_agent.run(state)

print("\nAGENT 2 METRICS")

for key, value in state[
    "metrics"
]["bed_allocation"].items():

    print(f"{key}: {value}")

# ==========================================
# 5. Display allocation recommendations
# ==========================================

recommendations = pd.DataFrame(
    state["allocation_recommendations"]
)

unallocated = pd.DataFrame(
    state["unallocated_patients"]
)

print("\nTOP 20 BED RECOMMENDATIONS")

if not recommendations.empty:

    columns = [
        "patient_id",
        "priority",
        "bed_id",
        "bed_type",
        "allocation_status"
    ]

    print(
        recommendations[columns]
        .head(20)
        .to_string(index=False)
    )

else:

    print("No allocation recommendations.")

print("\nUNALLOCATED PATIENTS")

if not unallocated.empty:

    print(
        unallocated[
            [
                "patient_id",
                "priority",
                "requested_care",
                "reason"
            ]
        ].head(20).to_string(index=False)
    )

else:

    print("No unallocated patients.")

# ==========================================
# 6. Validate recommendation safety
# ==========================================

if not recommendations.empty:

    assert not recommendations[
        "bed_id"
    ].duplicated().any()

    available_ids = set(
        beds.loc[
            beds["status"] == "Available",
            "bed_id"
        ]
    )

    assert set(
        recommendations["bed_id"]
    ).issubset(available_ids)

    assert (
        recommendations["allocation_status"]
        == "Pending Approval"
    ).all()

print("\nALLOCATION SAFETY CHECKS PASSED")

# ==========================================
# 7. Save historical snapshot outputs
# ==========================================

beds.to_csv(
    OUTPUT_DIR / "bed_inventory_snapshot.csv",
    index=False
)

recommendations.to_csv(
    OUTPUT_DIR / "allocation_recommendations.csv",
    index=False
)

unallocated.to_csv(
    OUTPUT_DIR / "unallocated_patients.csv",
    index=False
)

with open(
    OUTPUT_DIR / "allocation_metrics.json",
    "w"
) as file:

    json.dump(
        state["metrics"],
        file,
        indent=4
    )

with open(
    OUTPUT_DIR / "allocation_state.json",
    "w"
) as file:

    json.dump(
        state,
        file,
        indent=4
    )

# ==========================================
# 8. Controlled allocation scenario
# ==========================================

print("\nCONTROLLED SIMULATION SCENARIO")

# These are explicit fictional bed requests.
# They are not inferred clinical requirements.

scenario_queue = [

    {
        "patient_id": "SIM-P001",
        "encounter_id": "SIM-E001",
        "priority": 1,
        "waiting_time_minutes": 90,
        "requested_care": "ICU",
        "requested_bed_type": "ICU",
        "requirement_source":
            "Synthetic scenario input"
    },

    {
        "patient_id": "SIM-P002",
        "encounter_id": "SIM-E002",
        "priority": 2,
        "waiting_time_minutes": 120,
        "requested_care": "inpatient",
        "requested_bed_type": "medical ward",
        "requirement_source":
            "Synthetic scenario input"
    },

    {
        "patient_id": "SIM-P003",
        "encounter_id": "SIM-E003",
        "priority": 3,
        "waiting_time_minutes": 180,
        "requested_care": "inpatient",
        "requested_bed_type": "medical ward",
        "requirement_source":
            "Synthetic scenario input"
    }
]

scenario_beds = [

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

scenario_state = create_initial_state(
    simulation_time=DEFAULT_SIMULATION_TIME,
    simulation_id="SIM-CONTROLLED-001"
)

scenario_state["beds"] = scenario_beds

scenario_state["prioritised_queue"] = (
    scenario_queue
)

scenario_state = allocation_agent.run(
    scenario_state
)

print("\nCONTROLLED SCENARIO METRICS")

print(
    json.dumps(
        scenario_state["metrics"][
            "bed_allocation"
        ],
        indent=4
    )
)

assert len(
    scenario_state[
        "allocation_recommendations"
    ]
) == 2

assert len(
    scenario_state["unallocated_patients"]
) == 1

with open(
    OUTPUT_DIR / "controlled_scenario.json",
    "w"
) as file:

    json.dump(
        scenario_state,
        file,
        indent=4
    )

print("\nCONTROLLED SCENARIO PASSED")

print("\nOutputs saved to:", OUTPUT_DIR)

print("\nAGENT 2 EXECUTION COMPLETED")
