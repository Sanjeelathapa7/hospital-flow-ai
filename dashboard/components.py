
# HospitalFlow AI
# Reusable Dashboard Components

from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px


BED_COLOURS = {

    "Occupied": "#38BDF8",

    "Available": "#22C55E",

    "Cleaning": "#F59E0B",

    "Awaiting Cleaning": "#FB923C",

    "Unknown": "#64748B"
}


PRIORITY_COLOURS = {

    "1": "#EF4444",

    "2": "#F97316",

    "3": "#EAB308",

    "4": "#38BDF8",

    "5": "#22C55E"
}


AGENTS = [

    (
        "AGENT-01",
        "Admission Prioritisation",
        "Rule-based intelligent reasoning"
    ),

    (
        "AGENT-02",
        "Bed Availability and Allocation",
        "Constraint-based resource matching"
    ),

    (
        "AGENT-03",
        "Discharge Prediction",
        "Length-of-stay planning estimation"
    ),

    (
        "AGENT-04",
        "Coordination and Human Approval",
        "Operational monitoring and oversight"
    )
]


# ==========================================
# 1. Reusable data table
# ==========================================

def render_table(
    records,
    preferred_columns=None,
    empty_message="No records available."
):

    if not records:

        st.info(empty_message)

        return

    dataframe = pd.DataFrame(records)

    if preferred_columns:

        selected = [

            column

            for column in preferred_columns

            if column in dataframe.columns
        ]

        if selected:

            dataframe = dataframe[selected]

    st.dataframe(

        dataframe,

        use_container_width=True,

        hide_index=True
    )


# ==========================================
# 2. Hospital overview metrics
# ==========================================

def render_hospital_metrics(state):

    beds = state.get("beds", [])

    total = len(beds)

    occupied = sum(

        bed.get("status") == "Occupied"

        for bed in beds
    )

    available = sum(

        bed.get("status") == "Available"

        and bed.get("allocatable") is True

        for bed in beds
    )

    waiting = len(

        state.get("prioritised_queue", [])
    )

    occupancy = (

        occupied / total * 100

        if total else 0
    )

    columns = st.columns(5)

    columns[0].metric(
        "Total beds", total
    )

    columns[1].metric(
        "Occupied beds", occupied
    )

    columns[2].metric(
        "Available beds", available
    )

    columns[3].metric(
        "Occupancy rate",
        f"{occupancy:.1f}%"
    )

    columns[4].metric(
        "Waiting patients", waiting
    )


# ==========================================
# 3. Bed occupancy chart
# ==========================================

def render_bed_occupancy(state):

    beds = state.get("beds", [])

    if not beds:

        st.info("No bed inventory available.")

        return

    counts = Counter(

        bed.get("status", "Unknown")

        for bed in beds
    )

    dataframe = pd.DataFrame({

        "Status": list(counts.keys()),

        "Beds": list(counts.values())
    })

    figure = px.pie(

        dataframe,

        names="Status",

        values="Beds",

        hole=0.65,

        color="Status",

        color_discrete_map=BED_COLOURS,

        title="Hospital Bed Occupancy"
    )

    figure.update_traces(

        textposition="inside",

        textinfo="percent+value"
    )

    figure.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        height=350,

        margin=dict(
            l=10, r=10, t=50, b=10
        )
    )

    st.plotly_chart(

        figure,

        use_container_width=True
    )


# ==========================================
# 4. Bed type distribution
# ==========================================

def render_bed_distribution(state):

    beds = state.get("beds", [])

    if not beds:

        st.info("No bed records available.")

        return

    dataframe = pd.DataFrame(beds)

    grouped = (

        dataframe.groupby(

            ["bed_type", "status"]

        ).size().reset_index(name="Beds")
    )

    figure = px.bar(

        grouped,

        x="bed_type",

        y="Beds",

        color="status",

        color_discrete_map=BED_COLOURS,

        barmode="stack",

        title="Capacity by Bed Type"
    )

    figure.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        xaxis_title="Bed type",

        height=350
    )

    st.plotly_chart(

        figure,

        use_container_width=True
    )


# ==========================================
# 5. Patient priority distribution
# ==========================================

def render_priority_distribution(state):

    queue = state.get(
        "prioritised_queue", []
    )

    if not queue:

        st.info("No patients currently waiting.")

        return

    dataframe = pd.DataFrame(queue)

    counts = (

        dataframe["priority"]

        .value_counts()

        .reindex(range(1, 6), fill_value=0)

        .sort_index()

        .rename_axis("Priority")

        .reset_index(name="Patients")
    )

    counts["Priority"] = (

        counts["Priority"].astype(str)
    )

    figure = px.bar(

        counts,

        x="Priority",

        y="Patients",

        color="Priority",

        color_discrete_map=PRIORITY_COLOURS,

        text_auto=True,

        title="Waiting Patients by Priority"
    )

    figure.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        showlegend=False,

        height=320
    )

    st.plotly_chart(

        figure,

        use_container_width=True
    )


# ==========================================
# 6. Agent execution status
# ==========================================

def render_agent_status(state):

    completed = state.get(
        "completed_agents", []
    )

    columns = st.columns(2)

    for index, agent in enumerate(AGENTS):

        agent_id, name, description = agent

        with columns[index % 2]:

            with st.container(border=True):

                st.markdown(f"**{name}**")

                st.caption(agent_id)

                st.write(description)

                if agent_id in completed:

                    st.success("Completed")

                else:

                    st.info("Not executed")


# ==========================================
# 7. Operational alert summary
# ==========================================

def render_alert_summary(state):

    alerts = state.get("alerts", [])

    critical = sum(

        str(
            item.get("severity", "")
        ).lower() == "critical"

        for item in alerts
    )

    warning = sum(

        str(
            item.get("severity", "")
        ).lower() == "warning"

        for item in alerts
    )

    columns = st.columns(3)

    columns[0].metric(
        "Total alerts", len(alerts)
    )

    columns[1].metric(
        "Critical alerts", critical
    )

    columns[2].metric(
        "Warnings", warning
    )


# ==========================================
# 8. CSV download
# ==========================================

def render_download(
    records,
    filename,
    key
):

    if not records:

        return

    dataframe = pd.DataFrame(records)

    csv_data = dataframe.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="Download CSV",

        data=csv_data,

        file_name=filename,

        mime="text/csv",

        key=key
    )
