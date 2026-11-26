
# HospitalFlow AI
# Main Streamlit Application

import streamlit as st

from core.config import DISCLAIMER

from database.repository import (
    HospitalRepository
)

from workflows.hospital_workflow import (
    build_workflow
)

from dashboard.styles import (

    apply_styles,

    render_header
)
from dashboard.system_evaluation import (
    system_evaluation_page
)
from dashboard.views import (

    overview_page,

    simulation_page,

    patient_queue_page,

    bed_management_page,

    discharge_prediction_page,

    approval_page,

    agent_activity_page,

    database_page,

    model_evaluation_page
)


# ==========================================
# 1. Page configuration
# ==========================================

st.set_page_config(

    page_title="HospitalFlow AI",

    page_icon="🏥",

    layout="wide",

    initial_sidebar_state="expanded"
)

apply_styles()


# ==========================================
# 2. Initialise application session
# ==========================================

if "workflow" not in st.session_state:

    st.session_state["workflow"] = (
        build_workflow()
    )

if "repository" not in st.session_state:

    st.session_state["repository"] = (
        HospitalRepository()
    )

if "current_state" not in st.session_state:

    st.session_state["current_state"] = {}

if "awaiting_approval" not in st.session_state:

    st.session_state["awaiting_approval"] = False

if "workflow_config" not in st.session_state:

    st.session_state["workflow_config"] = None


workflow = st.session_state["workflow"]

repository = st.session_state["repository"]

state = st.session_state["current_state"]


# ==========================================
# 3. Sidebar navigation
# ==========================================

with st.sidebar:

    st.title("🏥 HospitalFlow AI")

    st.caption(
        "Intelligent Hospital Operations"
    )

    st.divider()

    page = st.radio(

        "Navigation",

        [

            "Hospital Overview",

            "Simulation Centre",

            "Patient Queue",

            "Bed Management",

            "Discharge Prediction",

            "Approval Centre",

            "Agent Activity",

            "Database & Audit",

"Model Evaluation",

"System Evaluation"
        ],

        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### System Status")

    st.success("SQLite repository initialised")

    st.success("LangGraph initialised")

    if st.session_state["awaiting_approval"]:

        st.warning("Awaiting human approval")

    elif state.get("workflow_status") == "Completed":

        st.success("Workflow completed")

    else:

        st.info("Ready for simulation")

    if state:

        st.caption("Active simulation")

        st.code(

            state.get(
                "simulation_id", "Unavailable"
            ),

            language=None
        )

    st.divider()

    st.markdown("### Project Information")

    st.caption(
        "CN7050 — Intelligent Systems"
    )

    st.caption(
        "MSc Artificial Intelligence"
    )

    st.caption(
        "University of East London"
    )

    st.caption(
        "HospitalFlow AI · Version 1.0"
    )


# ==========================================
# 4. Main application header
# ==========================================

render_header()

st.caption(DISCLAIMER)

if st.session_state.pop(

    "approval_success", False

):

    st.success(
        "Human decisions recorded successfully. "
        "The workflow has completed and the "
        "results have been saved to SQLite."
    )


# ==========================================
# 5. Page routing
# ==========================================

if page == "Hospital Overview":

    overview_page(state)

elif page == "Simulation Centre":

    simulation_page(

        workflow,

        repository
    )

elif page == "Database & Audit":

    database_page(repository)

elif page == "Model Evaluation":

    model_evaluation_page()

elif page == "System Evaluation":

    system_evaluation_page()

elif not state:

    st.info(

        "No active simulation. Open the "
        "Simulation Centre and run a scenario."
    )

elif page == "Patient Queue":

    patient_queue_page(state)

elif page == "Bed Management":

    bed_management_page(state)

elif page == "Discharge Prediction":

    discharge_prediction_page(state)

elif page == "Approval Centre":

    approval_page(

        state,

        workflow,

        repository
    )

elif page == "Agent Activity":

    agent_activity_page(state)


# ==========================================
# 6. Footer
# ==========================================

st.divider()

st.caption(

    "HospitalFlow AI | CN7050 Intelligent Systems "

    "| Educational simulation using synthetic data"
)
