
# LangGraph Workflow Tests

import pytest

from uuid import uuid4

from langgraph.types import Command

from workflows.hospital_workflow import (
    build_workflow,
    route_after_coordination
)

from workflows.scenarios import (
    create_scenario
)


def test_approval_routing():

    assert route_after_coordination({

        "approval_required": True

    }) == "human_review"


def test_no_approval_routing():

    assert route_after_coordination({

        "approval_required": False

    }) == "finalise"


def test_graph_contains_all_agents():

    workflow = build_workflow()

    nodes = workflow.get_graph().nodes

    for node in [

        "admission",
        "discharge",
        "allocation",
        "coordination"

    ]:

        assert node in nodes


@pytest.mark.parametrize(

    "scenario_name, expected_approvals, expected_waiting",

    [

        ("normal", 3, 0),

        ("bed_shortage", 2, 3),

        ("icu_crisis", 1, 4)
    ]
)

def test_complete_workflow(
    scenario_name,
    expected_approvals,
    expected_waiting
):

    workflow = build_workflow()

    state = create_scenario(scenario_name)

    config = {

        "configurable": {

            "thread_id": str(uuid4())
        }
    }

    result = workflow.invoke(
        state,
        config=config
    )

    assert result.get("__interrupt__")

    paused_state = (
        workflow.get_state(config).values
    )

    assert len(
        paused_state["approved_allocations"]
    ) == 0

    assert len(
        paused_state["approval_requests"]
    ) == expected_approvals

    decisions = [

        {

            "approval_id":
                request["approval_id"],

            "decision": "approve",

            "reviewer": "Test Reviewer",

            "reason": "Workflow integration test."
        }

        for request in paused_state[
            "approval_requests"
        ]
    ]

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

    assert len(
        final_state["approved_allocations"]
    ) == expected_approvals

    assert len(
        final_state["prioritised_queue"]
    ) == expected_waiting

    assert len(
        final_state["audit_log"]
    ) == expected_approvals

    assert final_state["workflow_trace"] == [

        "environment",
        "admission",
        "discharge",
        "allocation",
        "coordination",
        "human_review",
        "apply_decisions",
        "finalise"
    ]


def test_workflow_rejection():

    workflow = build_workflow()

    state = create_scenario("normal")

    config = {

        "configurable": {

            "thread_id": str(uuid4())
        }
    }

    workflow.invoke(
        state,
        config=config
    )

    paused_state = (
        workflow.get_state(config).values
    )

    decisions = [

        {

            "approval_id":
                request["approval_id"],

            "decision": "reject",

            "reviewer": "Test Reviewer",

            "reason": "Rejection scenario test."
        }

        for request in paused_state[
            "approval_requests"
        ]
    ]

    workflow.invoke(

        Command(resume=decisions),

        config=config
    )

    final_state = (
        workflow.get_state(config).values
    )

    assert len(
        final_state["approved_allocations"]
    ) == 0

    assert len(
        final_state["rejected_allocations"]
    ) == 3

    assert len(
        final_state["prioritised_queue"]
    ) == 3


def test_workflow_without_compatible_beds():

    workflow = build_workflow()

    state = create_scenario("normal")

    for bed in state["beds"]:

        bed["status"] = "Occupied"

        bed["allocatable"] = False

    config = {

        "configurable": {

            "thread_id": str(uuid4())
        }
    }

    result = workflow.invoke(
        state,
        config=config
    )

    assert not result.get("__interrupt__")

    assert result["workflow_status"] == "Completed"

    assert result["approval_status"] == "Not Required"

    assert len(result["prioritised_queue"]) == 3
