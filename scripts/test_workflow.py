
# HospitalFlow AI
# LangGraph Workflow Demonstration

from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

from langgraph.types import Command

from workflows.hospital_workflow import (
    build_workflow
)

from workflows.scenarios import (
    create_scenario,
    SCENARIOS
)


OUTPUT_DIR = BASE_DIR / "outputs" / "workflow"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_json(filename, data):

    with open(
        OUTPUT_DIR / filename, "w"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=str
        )


def run_scenario(workflow, scenario_name):

    print("\n" + "=" * 60)

    print(SCENARIOS[scenario_name].upper())

    print("=" * 60)

    state = create_scenario(scenario_name)

    config = {

        "configurable": {

            "thread_id":
                state["simulation_id"]
        }
    }

    # Execute graph until completion or interrupt.

    result = workflow.invoke(
        state,
        config=config
    )

    checkpoint = workflow.get_state(config)

    current_state = checkpoint.values

    print("\nWORKFLOW TRACE")

    print(
        " -> ".join(
            current_state["workflow_trace"]
        )
    )

    print("\nAPPROVAL REQUESTS")

    requests = current_state.get(
        "approval_requests", []
    )

    print(
        json.dumps(
            requests,
            indent=4
        )
    )

    save_json(
        f"{scenario_name}_before_review.json",
        current_state
    )

    if result.get("__interrupt__"):

        print(
            "\nWORKFLOW PAUSED FOR HUMAN REVIEW"
        )

        assert not current_state.get(
            "approved_allocations"
        )

        decisions = []

        for request in requests:

            # Scripted test decisions only.
            # Streamlit will collect real user input.

            decisions.append({

                "approval_id":
                    request["approval_id"],

                "decision": "approve",

                "reviewer":
                    "Simulation Test Administrator",

                "reason":
                    "Controlled scenario test approval."
            })

        workflow.invoke(

            Command(resume=decisions),

            config=config
        )

    final_state = (
        workflow.get_state(config).values
    )

    assert (
        final_state["workflow_status"]
        == "Completed"
    )

    assert len(
        final_state["completed_agents"]
    ) == 4

    approved_ids = [

        item["bed_id"]

        for item in final_state.get(
            "approved_allocations", []
        )
    ]

    assert len(approved_ids) == len(
        set(approved_ids)
    )

    print("\nFINAL SUMMARY")

    print(
        json.dumps(
            final_state["final_summary"],
            indent=4
        )
    )

    print("\nFINAL WORKFLOW TRACE")

    print(
        " -> ".join(
            final_state["workflow_trace"]
        )
    )

    save_json(
        f"{scenario_name}_final_state.json",
        final_state
    )

    print("\nSCENARIO PASSED")

    return final_state["final_summary"]


def main():

    workflow = build_workflow()

    summaries = []

    for scenario_name in SCENARIOS:

        summary = run_scenario(
            workflow,
            scenario_name
        )

        summaries.append(summary)

    save_json(
        "scenario_comparison.json",
        summaries
    )

    mermaid = (
        workflow.get_graph().draw_mermaid()
    )

    (OUTPUT_DIR / "workflow_diagram.mmd").write_text(
        mermaid
    )

    print("\nALL WORKFLOW SCENARIOS COMPLETED")

    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":

    main()
