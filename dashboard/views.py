
# HospitalFlow AI
# Dashboard Pages

import json

from pathlib import Path
from uuid import uuid4
from datetime import datetime, time

import pandas as pd
import streamlit as st
import plotly.express as px

from langgraph.types import Command

from core.state import create_initial_state

from workflows.scenarios import (
    create_scenario,
    SCENARIOS
)

from dashboard.components import (

    render_table,

    render_hospital_metrics,

    render_bed_occupancy,

    render_bed_distribution,

    render_priority_distribution,

    render_agent_status,

    render_alert_summary,

    render_download
)


BASE_DIR = Path(__file__).resolve().parents[1]


# ==========================================
# 1. Hospital overview
# ==========================================

def overview_page(state):

    st.header("Hospital Overview")

    if not state:

        st.info(
            "Start a simulation from the "
            "Simulation Centre."
        )

        st.markdown(
            "### About HospitalFlow AI"
        )

        st.write(
            "An intelligent multi-agent system "
            "for hospital patient prioritisation, "
            "bed allocation, length-of-stay "
            "estimation and operational coordination."
        )

        render_agent_status({})

        return

    st.caption(

        f"Simulation: {state['simulation_id']}"

        f" | Time: {state['simulation_time']}"

        f" | Scenario: {state.get('scenario_mode')}"
    )

    render_hospital_metrics(state)

    left, right = st.columns(2)

    with left:

        render_bed_occupancy(state)

    with right:

        render_bed_distribution(state)

    st.subheader("Patient Demand")

    render_priority_distribution(state)

    st.subheader("Agent Execution Status")

    render_agent_status(state)

    st.subheader("Operational Alerts")

    render_alert_summary(state)


# ==========================================
# 2. Simulation centre
# ==========================================

def simulation_page(workflow, repository):

    st.header("Simulation Centre")

    st.write(
        "Configure a hospital scenario and "
        "execute the four-agent LangGraph workflow."
    )

    if st.session_state.get(
        "awaiting_approval", False
    ):

        st.warning(
            "An existing simulation is awaiting "
            "human approval. Complete its review "
            "before starting another simulation."
        )

        return

    scenario_options = {

        "historical": "Historical Dataset Snapshot",

        **SCENARIOS
    }

    scenario = st.selectbox(

        "Hospital scenario",

        options=list(scenario_options.keys()),

        format_func=lambda value: (
            scenario_options[value]
        ),

        index=2
    )

    descriptions = {

        "historical":
            "Reconstructs hospital conditions "
            "from the synthetic operational dataset.",

        "normal":
            "Three admission requests with "
            "sufficient compatible beds.",

        "bed_shortage":
            "Five admission requests competing "
            "for two available beds.",

        "icu_crisis":
            "Critical ICU demand with "
            "no available ICU beds."
    }

    st.info(descriptions[scenario])

    left, right = st.columns(2)

    with left:

        selected_date = st.date_input(

            "Simulation date",

            value=datetime(2031, 11, 7).date(),

            min_value=datetime(
                2031, 11, 4
            ).date(),

            max_value=datetime(
                2031, 12, 30
            ).date()
        )

    with right:

        selected_time = st.time_input(

            "Simulation time",

            value=time(12, 0)
        )

    simulation_time = datetime.combine(

        selected_date,

        selected_time

    ).strftime("%Y-%m-%d %H:%M:%S")

    st.caption(
        "Dates are restricted to the later "
        "synthetic dataset period to avoid "
        "using a model trained on future records "
        "relative to the simulation clock."
    )

    if st.button(

        "Run Multi-Agent Simulation",

        type="primary",

        use_container_width=True
    ):

        try:

            if scenario == "historical":

                state = create_initial_state(

                    simulation_time=simulation_time,

                    simulation_id=(
                        f"SIM-{uuid4().hex[:10]}"
                    )
                )

                state["scenario_mode"] = (
                    "historical"
                )

            else:

                state = create_scenario(

                    scenario_name=scenario,

                    simulation_time=simulation_time
                )

            config = {

                "configurable": {

                    "thread_id":
                        state["simulation_id"]
                }
            }

            with st.spinner(

                "Executing hospital agents..."
            ):

                result = workflow.invoke(

                    state,

                    config=config
                )

            current_state = (

                workflow.get_state(config).values
            )

            awaiting = bool(

                result.get("__interrupt__")
            )

            if awaiting:

                current_state = (

                    repository.load_state(
                        state["simulation_id"]
                    )

                    or current_state
                )

            st.session_state["current_state"] = (
                current_state
            )

            st.session_state["workflow_config"] = (
                config
            )

            st.session_state["awaiting_approval"] = (
                awaiting
            )

            if awaiting:

                st.warning(
                    "Workflow paused. Open the "
                    "Approval Centre to review "
                    "allocation requests."
                )

            else:

                st.success(
                    "Simulation completed."
                )

            render_agent_status(current_state)

            render_hospital_metrics(current_state)

        except Exception as error:

            st.error(
                "Simulation execution failed."
            )

            st.exception(error)


# ==========================================
# 3. Patient queue
# ==========================================

def patient_queue_page(state):

    st.header("Patient Admission Queue")

    st.caption(
        "Agent 1 — Admission Prioritisation"
    )

    queue = state.get(
        "prioritised_queue", []
    )

    columns = st.columns(3)

    columns[0].metric(

        "Waiting patients",

        len(queue)
    )

    columns[1].metric(

        "Highest urgency",

        sum(
            item.get("priority") == 1
            for item in queue
        )
    )

    columns[2].metric(

        "Average waiting time",

        f"{sum(item.get('waiting_time_minutes', 0) for item in queue) / len(queue):.1f} min"
        if queue else "0 min"
    )

    render_priority_distribution(state)

    selected_priorities = st.multiselect(

        "Filter patient priority",

        options=[1, 2, 3, 4, 5],

        default=[1, 2, 3, 4, 5]
    )

    filtered = [

        item

        for item in queue

        if item.get("priority")
        in selected_priorities
    ]

    render_table(

        filtered,

        preferred_columns=[

            "queue_position",

            "patient_id",

            "encounter_id",

            "priority",

            "priority_label",

            "requested_care",

            "requested_bed_type",

            "waiting_time_minutes",

            "waiting_status"
        ]
    )

    render_download(

        filtered,

        "patient_queue.csv",

        "queue_download"
    )

    st.info(
        "Priority 1 represents the highest urgency. "
        "The system uses recorded or explicitly "
        "simulated priority values rather than "
        "performing autonomous clinical triage."
    )


# ==========================================
# 4. Bed management
# ==========================================

def bed_management_page(state):

    st.header("Bed Availability and Allocation")

    st.caption(
        "Agent 2 — Constraint-Based Allocation"
    )

    render_hospital_metrics(state)

    render_bed_distribution(state)

    beds = state.get("beds", [])

    statuses = sorted({

        item.get("status", "Unknown")

        for item in beds
    })

    bed_types = sorted({

        item.get("bed_type", "Unknown")

        for item in beds
    })

    left, right = st.columns(2)

    with left:

        selected_status = st.multiselect(

            "Bed status",

            options=statuses,

            default=statuses
        )

    with right:

        selected_types = st.multiselect(

            "Bed type",

            options=bed_types,

            default=bed_types
        )

    filtered = [

        item

        for item in beds

        if item.get("status") in selected_status

        and item.get("bed_type") in selected_types
    ]

    st.subheader("Bed Inventory")

    render_table(

        filtered,

        preferred_columns=[

            "bed_id",

            "bed_type",

            "status",

            "allocatable",

            "assigned_patient_id",

            "assigned_encounter_id"
        ]
    )

    render_download(

        filtered,

        "bed_inventory.csv",

        "bed_download"
    )

    st.subheader("Allocation Recommendations")

    render_table(

        state.get(
            "allocation_recommendations", []
        ),

        preferred_columns=[

            "patient_id",

            "encounter_id",

            "priority",

            "bed_id",

            "bed_type",

            "allocation_status",

            "reason"
        ]
    )

    st.subheader("Unallocated Patients")

    render_table(

        state.get(
            "unallocated_patients", []
        )
    )


# ==========================================
# 5. Discharge prediction
# ==========================================

def discharge_prediction_page(state):

    st.header("Discharge Planning Estimates")

    st.caption(
        "Agent 3 — Length-of-Stay Prediction"
    )

    st.warning(
        "These estimates support educational "
        "capacity planning only. They do not "
        "determine clinical discharge readiness "
        "or automatically release beds."
    )

    predictions = state.get(
        "discharge_predictions", []
    )

    if not predictions:

        st.info(
            "No length-of-stay predictions "
            "available for this simulation."
        )

        return

    dataframe = pd.DataFrame(predictions)

    if "predicted_los_hours" in dataframe:

        values = pd.to_numeric(

            dataframe["predicted_los_hours"],

            errors="coerce"
        )

        columns = st.columns(3)

        columns[0].metric(

            "Prediction records",

            len(dataframe)
        )

        columns[1].metric(

            "Average estimated LOS",

            f"{values.mean():.1f} hours"

            if values.notna().any()

            else "Unavailable"
        )

        columns[2].metric(

            "Average estimated days",

            f"{values.mean() / 24:.2f}"

            if values.notna().any()

            else "Unavailable"
        )

        if values.notna().any():

            figure = px.histogram(

                dataframe.assign(
                    predicted_los_hours=values
                ),

                x="predicted_los_hours",

                title="Estimated Length of Stay",

                labels={

                    "predicted_los_hours":
                        "Estimated LOS (hours)"
                },

                template="plotly_dark"
            )

            st.plotly_chart(

                figure,

                use_container_width=True
            )

    render_table(predictions)

    render_download(

        predictions,

        "discharge_predictions.csv",

        "prediction_download"
    )

    st.caption(
        "The operational model is a median "
        "baseline because the enhanced ML model "
        "did not generalise reliably to the "
        "later test period."
    )


# ==========================================
# 6. Human approval centre
# ==========================================

def approval_page(
    state,
    workflow,
    repository
):

    st.header("Human Approval Centre")

    st.caption(
        "Agent 4 — Human-in-the-Loop Oversight"
    )

    pending = [

        item

        for item in state.get(
            "approval_requests", []
        )

        if item.get("approval_status")
        == "Pending"
    ]

    columns = st.columns(3)

    columns[0].metric(

        "Pending approvals",

        len(pending)
    )

    columns[1].metric(

        "Approved allocations",

        len(
            state.get(
                "approved_allocations", []
            )
        )
    )

    columns[2].metric(

        "Rejected allocations",

        len(
            state.get(
                "rejected_allocations", []
            )
        )
    )

    if pending:

        if not st.session_state.get(
            "awaiting_approval", False
        ):

            st.warning(
                "This is a saved simulation record. "
                "Its live workflow checkpoint "
                "is not available for resumption."
            )

            render_table(pending)

            return

        st.warning(
            "The workflow is paused until "
            "all pending requests are reviewed."
        )

        with st.form("human_review_form"):

            reviewer = st.text_input(

                "Reviewer name",

                placeholder="Enter reviewer name"
            )

            decisions = []

            for request in pending:

                approval_id = request[
                    "approval_id"
                ]

                with st.container(border=True):

                    st.markdown(

                        f"### Request {approval_id}"
                    )

                    left, right = st.columns(2)

                    with left:

                        st.write(

                            "Patient:",

                            request.get("patient_id")
                        )

                        st.write(

                            "Encounter:",

                            request.get("encounter_id")
                        )

                        st.write(

                            "Priority:",

                            request.get("priority")
                        )

                    with right:

                        st.write(

                            "Recommended bed:",

                            request.get("bed_id")
                        )

                        st.write(

                            "Bed type:",

                            request.get("bed_type")
                        )

                    decision = st.radio(

                        "Allocation decision",

                        options=[

                            "Select decision",

                            "Approve",

                            "Reject"
                        ],

                        key=f"decision_{approval_id}",

                        horizontal=True
                    )

                    reason = st.text_area(

                        "Decision justification",

                        key=f"reason_{approval_id}"
                    )

                    decisions.append({

                        "approval_id": approval_id,

                        "decision":
                            decision.lower(),

                        "reviewer": reviewer.strip(),

                        "reason": reason.strip()
                    })

            submitted = st.form_submit_button(

                "Submit Human Decisions",

                type="primary",

                use_container_width=True
            )

        if submitted:

            if not reviewer.strip():

                st.error(
                    "Reviewer name is required."
                )

                return

            if any(

                item["decision"]

                not in ["approve", "reject"]

                for item in decisions
            ):

                st.error(
                    "Select a decision for "
                    "every request."
                )

                return

            if any(

                not item["reason"]

                for item in decisions
            ):

                st.error(
                    "Provide a justification "
                    "for every decision."
                )

                return

            try:

                config = st.session_state[
                    "workflow_config"
                ]

                with st.spinner(

                    "Applying human decisions..."
                ):

                    workflow.invoke(

                        Command(resume=decisions),

                        config=config
                    )

                final_state = (

                    workflow.get_state(config).values
                )

                if final_state.get(
                    "workflow_status"
                ) != "Completed":

                    raise RuntimeError(
                        "Workflow has not completed."
                    )

                saved_state = repository.load_state(

                    final_state["simulation_id"]
                )

                if not saved_state:

                    raise RuntimeError(
                        "Final state was not persisted."
                    )

                st.session_state[
                    "current_state"
                ] = saved_state

                st.session_state[
                    "awaiting_approval"
                ] = False

                st.session_state[
                    "approval_success"
                ] = True

                st.rerun()

            except Exception as error:

                st.error(
                    "Unable to complete approval."
                )

                st.exception(error)

    else:

        st.info(
            "No pending approval requests."
        )

    st.subheader("Approved Allocations")

    render_table(

        state.get(
            "approved_allocations", []
        )
    )

    st.subheader("Rejected Allocations")

    render_table(

        state.get(
            "rejected_allocations", []
        )
    )


# ==========================================
# 7. Agent activity and coordination
# ==========================================

def agent_activity_page(state):

    st.header("Agent Activity and Coordination")

    render_agent_status(state)

    st.subheader("Workflow Execution Trace")

    trace = state.get(
        "workflow_trace", []
    )

    if trace:

        st.code(

            " → ".join(trace),

            language=None
        )

    st.subheader("Operational Alert Summary")

    render_alert_summary(state)

    alerts = state.get("alerts", [])

    if alerts:

        severity = st.selectbox(

            "Alert severity",

            options=[

                "All",

                "Critical",

                "Warning",

                "Info"
            ]
        )

        filtered = [

            item

            for item in alerts

            if severity == "All"

            or str(

                item.get("severity", "")

            ).lower() == severity.lower()
        ]

        render_table(filtered)

    else:

        st.success(
            "No operational alerts recorded."
        )

    st.subheader("Inter-Agent Communication")

    messages = state.get(
        "agent_messages", []
    )

    render_table(messages)

    render_download(

        messages,

        "agent_messages.csv",

        "messages_download"
    )

    st.subheader("Operational Recommendations")

    recommendations = state.get(
        "recommendations", []
    )

    if recommendations:

        st.json(recommendations)

    else:

        st.info(
            "No additional recommendations."
        )

    st.subheader("Agent Performance Metrics")

    st.json(

        state.get("metrics", {})
    )


# ==========================================
# 8. Database and audit history
# ==========================================

def database_page(repository):

    st.header("Database and Audit History")

    summary = repository.get_database_summary()

    columns = st.columns(4)

    columns[0].metric(

        "Simulation runs",

        summary["simulation_runs"]
    )

    columns[1].metric(

        "Allocation records",

        summary["bed_allocations"]
    )

    columns[2].metric(

        "Audit events",

        summary["audit_logs"]
    )

    columns[3].metric(

        "Agent messages",

        summary["agent_messages"]
    )

    simulations = repository.list_simulations()

    st.subheader("Saved Simulations")

    render_table(simulations)

    if not simulations:

        return

    simulation_ids = [

        item["simulation_id"]

        for item in simulations
    ]

    selected_id = st.selectbox(

        "Select saved simulation",

        options=simulation_ids
    )

    tables = [

        "patient_queue",

        "bed_inventory",

        "discharge_predictions",

        "bed_allocations",

        "operational_alerts",

        "audit_logs",

        "agent_messages"
    ]

    selected_table = st.selectbox(

        "Select database table",

        options=tables
    )

    records = repository.get_records(

        selected_table,

        selected_id
    )

    st.subheader(

        selected_table.replace("_", " ").title()
    )

    render_table(records)

    render_download(

        records,

        f"{selected_id}_{selected_table}.csv",

        "database_download"
    )

    saved_state = repository.load_state(
        selected_id
    )

    if saved_state:

        with st.expander(

            "View saved simulation summary"
        ):

            st.json(

                saved_state.get(
                    "final_summary", {}
                )
            )

        st.download_button(

            "Download complete simulation JSON",

            data=json.dumps(

                saved_state,

                indent=4,

                default=str
            ),

            file_name=f"{selected_id}.json",

            mime="application/json"
        )


# ==========================================
# 9. Machine learning evaluation
# ==========================================

def model_evaluation_page():

    st.header("Machine Learning Evaluation")

    st.write(
        "Comparison of the original and "
        "enhanced length-of-stay experiments."
    )

    experiments = {

        "Original Experiment":

            BASE_DIR / "outputs" / "evaluation",

        "Enhanced Experiment":

            BASE_DIR / "outputs" / "evaluation_v2"
    }

    for title, directory in experiments.items():

        st.subheader(title)

        comparison_file = (

            directory / "model_comparison.csv"
        )

        test_file = (

            directory / "test_metrics.csv"
        )

        if comparison_file.exists():

            dataframe = pd.read_csv(
                comparison_file
            )

            st.dataframe(

                dataframe,

                hide_index=True,

                use_container_width=True
            )

            figure = px.bar(

                dataframe,

                x="Model",

                y="MAE_hours",

                color="Model",

                title="Validation MAE Comparison",

                template="plotly_dark"
            )

            figure.update_layout(

                showlegend=False,

                yaxis_title="MAE (hours)"
            )

            st.plotly_chart(

                figure,

                use_container_width=True
            )

        else:

            st.warning(
                f"Missing file: {comparison_file.name}"
            )

        if test_file.exists():

            st.markdown("**Later Test Results**")

            st.dataframe(

                pd.read_csv(test_file),

                hide_index=True,

                use_container_width=True
            )

    st.subheader("Model Selection Policy")

    policy_file = (

        BASE_DIR

        / "outputs"

        / "evaluation_v2"

        / "model_selection_policy.json"
    )

    if policy_file.exists():

        with open(policy_file) as file:

            policy = json.load(file)

        st.json(policy)

    st.warning(
        "The enhanced model improved validation "
        "performance but performed worse than "
        "the median baseline on the later test "
        "period. Therefore, it is retained for "
        "experimental comparison rather than "
        "operational use."
    )

    st.caption(
        "The dataset is synthetic. These results "
        "do not establish clinical validity."
    )

    st.subheader("Evaluation Figures")

    figures = [

        "06_model_mae_comparison.png",

        "07_model_rmse_comparison.png",

        "09_actual_vs_predicted.png",

        "10_residual_distribution.png",

        "17_experiment_mae_comparison.png",

        "18_experiment_r2_comparison.png",

        "19_model_generalisation.png",

        "20_monthly_los_distribution.png"
    ]

    columns = st.columns(2)

    for index, filename in enumerate(figures):

        path = (

            BASE_DIR

            / "outputs"

            / "figures"

            / filename
        )

        if path.exists():

            with columns[index % 2]:

                st.image(

                    str(path),

                    caption=filename,

                    use_container_width=True
                )
