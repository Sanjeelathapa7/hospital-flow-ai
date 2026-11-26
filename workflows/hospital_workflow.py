
# HospitalFlow AI
# LangGraph Multi-Agent Orchestration

from copy import deepcopy

import pandas as pd

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.memory import (
    InMemorySaver
)

from langgraph.types import interrupt

from core.state import HospitalState

from core.environment import HospitalEnvironment

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


# ==========================================
# 1. Workflow tracing
# ==========================================

def trace(state, node_name):

    state.setdefault(
        "workflow_trace", []
    ).append(node_name)

    return state


# ==========================================
# 2. Initialise hospital environment
# ==========================================

def environment_node(state):

    state = deepcopy(state)

    if state.get("scenario_mode") == "historical":

        environment = HospitalEnvironment()

        state = environment.initialise(state)

    else:

        beds = state.get("beds", [])

        occupied = sum(
            bed["status"] == "Occupied"
            for bed in beds
        )

        state.setdefault(
            "metrics", {}
        )["environment"] = {

            "observed_capacity": len(beds),

            "occupied_beds": occupied,

            "available_beds": sum(
                bed["status"] == "Available"
                for bed in beds
            ),

            "occupancy_rate": round(
                occupied / len(beds) * 100, 2
            ) if beds else 0.0
        }

    state["workflow_status"] = "Running"

    return trace(state, "environment")


# ==========================================
# 3. Agent 1 node
# ==========================================

def admission_node(state):

    state = deepcopy(state)

    agent = AdmissionPrioritisationAgent()

    if state.get("scenario_mode") == "historical":

        state = agent.run(state)

    else:

        queue = pd.DataFrame(
            state.get("scenario_patients", [])
        )

        prioritised = agent.reason(queue)

        state["waiting_patients"] = (
            agent.to_records(queue)
        )

        state["prioritised_queue"] = (
            agent.to_records(prioritised)
        )

        state["alerts"] = (
            agent.generate_alerts(prioritised)
        )

        state.setdefault(
            "metrics", {}
        )["admission"] = (
            agent.calculate_metrics(prioritised)
        )

        state.setdefault(
            "completed_agents", []
        ).append("AGENT-01")

        state.setdefault(
            "agent_messages", []
        ).append({

            "sender": "AGENT-01",

            "receiver": "AGENT-02",

            "message_type":
                "PRIORITISED_PATIENT_QUEUE",

            "patient_count":
                len(prioritised),

            "simulation_time":
                state["simulation_time"]
        })

    return trace(state, "admission")


# ==========================================
# 4. Agent 3 node
# ==========================================

def discharge_node(state):

    state = deepcopy(state)

    agent = DischargePredictionAgent()

    state = agent.run(state)

    return trace(state, "discharge")


# ==========================================
# 5. Agent 2 node
# ==========================================

def allocation_node(state):

    state = deepcopy(state)

    agent = BedAllocationAgent()

    state = agent.run(state)

    return trace(state, "allocation")


# ==========================================
# 6. Agent 4 node
# ==========================================

def coordination_node(state):

    state = deepcopy(state)

    agent = CoordinationAgent()

    state = agent.run(state)

    return trace(state, "coordination")


# ==========================================
# 7. Conditional routing
# ==========================================

def route_after_coordination(state):

    if state.get("approval_required", False):

        return "human_review"

    return "finalise"


# ==========================================
# 8. Human-in-the-loop checkpoint
# ==========================================

def human_review_node(state):

    # LangGraph pauses execution here.
    # The workflow resumes only when decisions
    # are supplied using Command(resume=...).

    decisions = interrupt({

        "message":
            "Human allocation review required.",

        "simulation_id":
            state["simulation_id"],

        "approval_requests":
            state["approval_requests"],

        "alerts":
            state["alerts"]
    })

    state = deepcopy(state)

    state["human_decisions"] = decisions

    return trace(state, "human_review")


# ==========================================
# 9. Apply human decisions
# ==========================================

def apply_decisions_node(state):

    state = deepcopy(state)

    agent = CoordinationAgent()

    decisions = state.get(
        "human_decisions", []
    )

    if not isinstance(decisions, list):

        raise ValueError(
            "Human decisions must be a list."
        )

    pending_ids = {

        request["approval_id"]

        for request in state.get(
            "approval_requests", []
        )

        if request["approval_status"] == "Pending"
    }

    supplied_ids = [

        item["approval_id"]

        for item in decisions
    ]

    if (
        set(supplied_ids) != pending_ids
        or len(supplied_ids) != len(set(supplied_ids))
    ):

        raise ValueError(
            "Provide exactly one decision "
            "for every pending approval."
        )

    for item in decisions:

        state = agent.apply_decision(

            state,

            approval_id=item["approval_id"],

            decision=item["decision"],

            reviewer=item["reviewer"],

            reason=item["reason"]
        )

    return trace(state, "apply_decisions")


# ==========================================
# 10. Finalise workflow
# ==========================================

def finalise_node(state):

    state = deepcopy(state)

    beds = state.get("beds", [])

    occupied = sum(
        bed["status"] == "Occupied"
        for bed in beds
    )

    available = sum(

        bed["status"] == "Available"
        and bed.get("allocatable") is True

        for bed in beds
    )

    state["final_summary"] = {

        "simulation_id":
            state["simulation_id"],

        "scenario_mode":
            state.get("scenario_mode"),

        "completed_agents":
            len(state.get("completed_agents", [])),

        "total_beds":
            len(beds),

        "occupied_beds":
            occupied,

        "available_beds":
            available,

        "occupancy_rate": round(
            occupied / len(beds) * 100, 2
        ) if beds else 0.0,

        "patients_remaining":
            len(state.get("prioritised_queue", [])),

        "approved_allocations":
            len(state.get("approved_allocations", [])),

        "rejected_allocations":
            len(state.get("rejected_allocations", [])),

        "total_alerts":
            len(state.get("alerts", [])),

        "audit_events":
            len(state.get("audit_log", [])),

        "approval_status":
            state.get("approval_status")
    }

    state["workflow_status"] = "Completed"

    return trace(state, "finalise")


# ==========================================
# 11. Build LangGraph
# ==========================================

def build_workflow(checkpointer=None):

    builder = StateGraph(HospitalState)

    builder.add_node(
        "environment",
        environment_node
    )

    builder.add_node(
        "admission",
        admission_node
    )

    builder.add_node(
        "discharge",
        discharge_node
    )

    builder.add_node(
        "allocation",
        allocation_node
    )

    builder.add_node(
        "coordination",
        coordination_node
    )

    builder.add_node(
        "human_review",
        human_review_node
    )

    builder.add_node(
        "apply_decisions",
        apply_decisions_node
    )

    builder.add_node(
        "finalise",
        finalise_node
    )

    builder.add_edge(
        START, "environment"
    )

    builder.add_edge(
        "environment", "admission"
    )

    builder.add_edge(
        "admission", "discharge"
    )

    builder.add_edge(
        "discharge", "allocation"
    )

    builder.add_edge(
        "allocation", "coordination"
    )

    builder.add_conditional_edges(

        "coordination",

        route_after_coordination,

        {
            "human_review": "human_review",
            "finalise": "finalise"
        }
    )

    builder.add_edge(
        "human_review", "apply_decisions"
    )

    builder.add_edge(
        "apply_decisions", "finalise"
    )

    builder.add_edge(
        "finalise", END
    )

    if checkpointer is None:

        checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )
