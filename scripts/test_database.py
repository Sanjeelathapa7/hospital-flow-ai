
# HospitalFlow AI
# Database Integration Demonstration

from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from langgraph.types import Command

from database.repository import (
    HospitalRepository
)

from workflows.hospital_workflow import (
    build_workflow
)

from workflows.scenarios import (
    create_scenario
)


OUTPUT_DIR = (
    BASE_DIR / "outputs" / "database"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 60)
    print("HOSPITALFLOW AI")
    print("DATABASE INTEGRATION TEST")
    print("=" * 60)

    repository = HospitalRepository()

    workflow = build_workflow()

    state = create_scenario(
        "bed_shortage"
    )

    simulation_id = state["simulation_id"]

    config = {
        "configurable": {
            "thread_id": simulation_id
        }
    }

    print(
        "\nSimulation ID:",
        simulation_id
    )

    # ==========================================
    # 1. Run until human review
    # ==========================================

    result = workflow.invoke(
        state,
        config=config
    )

    assert result.get("__interrupt__")

    paused_state = (
        workflow.get_state(config).values
    )

    saved_pending = repository.load_state(
        simulation_id
    )

    assert saved_pending is not None

    assert saved_pending["workflow_status"] == (
        "Awaiting Human Approval"
    )

    print("\nPENDING STATE SAVED")

    requests = repository.get_records(
        "bed_allocations",
        simulation_id
    )

    assert len(requests) == 2

    print(
        pd.DataFrame(requests)[
            [
                "approval_id",
                "patient_id",
                "bed_id",
                "approval_status"
            ]
        ].to_string(index=False)
    )

    # ==========================================
    # 2. Provide scripted human decisions
    # ==========================================

    decisions = []

    for index, request in enumerate(
        paused_state["approval_requests"]
    ):

        decisions.append({

            "approval_id":
                request["approval_id"],

            "decision":
                "approve"
                if index == 0
                else "reject",

            "reviewer":
                "Database Test Administrator",

            "reason":
                "Controlled database integration test."
        })

    workflow.invoke(
        Command(resume=decisions),
        config=config
    )

    # ==========================================
    # 3. Retrieve completed simulation
    # ==========================================

    saved_state = repository.load_state(
        simulation_id
    )

    assert saved_state is not None

    assert saved_state["workflow_status"] == (
        "Completed"
    )

    assert len(
        saved_state["approved_allocations"]
    ) == 1

    assert len(
        saved_state["rejected_allocations"]
    ) == 1

    print("\nFINAL SIMULATION SUMMARY")

    print(
        json.dumps(
            saved_state["final_summary"],
            indent=4
        )
    )

    # ==========================================
    # 4. Retrieve relational records
    # ==========================================

    tables = [

        "patient_queue",
        "bed_inventory",
        "discharge_predictions",
        "bed_allocations",
        "operational_alerts",
        "audit_logs",
        "agent_messages"
    ]

    print("\nCURRENT SIMULATION RECORDS")

    for table in tables:

        records = repository.get_records(
            table,
            simulation_id
        )

        print(
            f"{table}: {len(records)} records"
        )

        pd.DataFrame(records).to_csv(
            OUTPUT_DIR / f"{table}.csv",
            index=False
        )

    audit = repository.get_records(
        "audit_logs",
        simulation_id
    )

    assert len(audit) == 2

    # ==========================================
    # 5. Verify repeated saving
    # ==========================================

    repository.save_state(saved_state)

    repeated_audit = repository.get_records(
        "audit_logs",
        simulation_id
    )

    assert len(repeated_audit) == 2

    print("\nDUPLICATE SAVE TEST PASSED")

    # ==========================================
    # 6. Retrieve database history
    # ==========================================

    simulations = (
        repository.list_simulations()
    )

    print("\nSIMULATION HISTORY")

    print(
        pd.DataFrame(simulations)
        .to_string(index=False)
    )

    pd.DataFrame(simulations).to_csv(
        OUTPUT_DIR / "simulation_history.csv",
        index=False
    )

    summary = (
        repository.get_database_summary()
    )

    print("\nTOTAL DATABASE RECORDS")

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    with open(
        OUTPUT_DIR / "database_summary.json",
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        "\nDATABASE INTEGRATION TEST PASSED"
    )


if __name__ == "__main__":

    main()
