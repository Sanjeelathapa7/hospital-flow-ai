
# HospitalFlow AI
# Shared Multi-Agent State

from typing import TypedDict, Any


class HospitalState(TypedDict, total=False):

    # Simulation information
    simulation_time: str
    simulation_id: str

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

    # NEW: Human approval requests
    approval_requests: list[dict[str, Any]]

    # System evaluation
    metrics: dict[str, Any]

    # Audit and workflow tracking
    agent_messages: list[dict[str, Any]]
    completed_agents: list[str]
    errors: list[str]

    # NEW: System audit log
    audit_log: list[dict[str, Any]]


def create_initial_state(
    simulation_time: str,
    simulation_id: str = "SIM-001"
) -> HospitalState:

    return HospitalState(

        simulation_time=simulation_time,
        simulation_id=simulation_id,

        patients=[],
        beds=[],
        staffing=[],

        waiting_patients=[],
        prioritised_queue=[],

        available_beds=[],
        allocation_recommendations=[],
        unallocated_patients=[],

        discharge_predictions=[],

        alerts=[],
        recommendations=[],

        approval_required=True,
        approval_status="Pending",

        approved_allocations=[],
        rejected_allocations=[],

        # NEW: Initialise approval requests
        approval_requests=[],

        metrics={},

        agent_messages=[],
        completed_agents=[],
        errors=[],

        # NEW: Initialise audit log
        audit_log=[]

    )
