
# HospitalFlow AI
# Shared Multi-Agent State

from typing import TypedDict, Any


class HospitalState(TypedDict, total=False):

    # Simulation information
    simulation_time: str
    simulation_id: str

    # NEW: Scenario configuration
    scenario_mode: str
    scenario_patients: list[dict[str, Any]]

    # Hospital environment
    patients: list[dict[str, Any]]
    beds: list[dict[str, Any]]
    staffing: list[dict[str, Any]]

    # Agent 1 outputs
    waiting_patients: list[dict[str, Any]]
    prioritised_queue: list[dict[str, Any]]

    # Agent 2 outputs
    available_beds: list[dict[str, Any]]
    allocation_recommendations: list[dict[str, Any]]
    unallocated_patients: list[dict[str, Any]]

    # Agent 3 outputs
    discharge_predictions: list[dict[str, Any]]

    # Agent 4 outputs
    alerts: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]

    # Human approval
    approval_required: bool
    approval_status: str
    approved_allocations: list[dict[str, Any]]
    rejected_allocations: list[dict[str, Any]]
    approval_requests: list[dict[str, Any]]

    # NEW: Human decisions
    human_decisions: list[dict[str, Any]]

    # System evaluation
    metrics: dict[str, Any]

    # Audit and workflow tracking
    agent_messages: list[dict[str, Any]]
    completed_agents: list[str]
    errors: list[str]
    audit_log: list[dict[str, Any]]

    # NEW: Workflow management
    workflow_status: str
    workflow_trace: list[str]

    # NEW: Final system summary
    final_summary: dict[str, Any]


def create_initial_state(
    simulation_time: str,
    simulation_id: str = "SIM-001"
) -> HospitalState:

    return HospitalState(

        # Simulation information
        simulation_time=simulation_time,
        simulation_id=simulation_id,

        # NEW: Initialise scenario configuration
        scenario_mode="historical",
        scenario_patients=[],

        # Hospital environment
        patients=[],
        beds=[],
        staffing=[],

        # Agent 1
        waiting_patients=[],
        prioritised_queue=[],

        # Agent 2
        available_beds=[],
        allocation_recommendations=[],
        unallocated_patients=[],

        # Agent 3
        discharge_predictions=[],

        # Agent 4
        alerts=[],
        recommendations=[],

        # Human approval
        approval_required=True,
        approval_status="Pending",
        approved_allocations=[],
        rejected_allocations=[],
        approval_requests=[],

        # NEW: Initialise human decisions
        human_decisions=[],

        # System evaluation
        metrics={},

        # Audit and workflow tracking
        agent_messages=[],
        completed_agents=[],
        errors=[],
        audit_log=[],

        # NEW: Initialise workflow management
        workflow_status="Initialised",
        workflow_trace=[],

        # NEW: Initialise final summary
        final_summary={}

    )
