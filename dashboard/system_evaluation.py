from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = (
    ROOT / "outputs" / "system_evaluation"
)

FIGURE_DIR = EVALUATION_DIR / "figures"

METHODS = ["FCFS", "HospitalFlow AI"]

COLORS = {
    "FCFS": "#64748B",
    "HospitalFlow AI": "#2563EB",
}


# ==========================================
# 1. Load evaluation data
# ==========================================

def load_evaluation_data():

    files = {
        "scenarios": "scenario_performance.csv",
        "overall": "overall_system_comparison.csv",
        "stress": "stress_test_results.csv",
        "stress_summary": "stress_test_summary.csv",
        "stress_overall": "stress_test_overall.csv",
        "paired": "stress_test_paired_comparison.csv",
        "outcomes": "stress_test_outcomes.csv",
        "scalability": "scalability_summary.csv",
        "safety": "stress_test_safety.csv",
        "historical": "historical_queue_summary.csv",
    }

    data = {}

    for name, filename in files.items():

        path = EVALUATION_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"Missing evaluation file: {filename}"
            )

        data[name] = pd.read_csv(path)

    return data


# ==========================================
# 2. Performance overview
# ==========================================

def render_performance_overview(data):

    overall = data["stress_overall"]

    fcfs = overall.loc[
        overall["method"].eq("FCFS")
    ].iloc[0]

    ai = overall.loc[
        overall["method"].eq("HospitalFlow AI")
    ].iloc[0]

    outcomes = data["outcomes"].iloc[0]

    safety_violations = int(
        data["safety"]["violations"].sum()
    )

    improvement = (
        ai["overall_high_priority_rate"]
        - fcfs["overall_high_priority_rate"]
    )

    additional_urgent = int(
        ai["urgent_patients_selected"]
        - fcfs["urgent_patients_selected"]
    )

    st.subheader("Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Stress scenarios",
        f"{int(outcomes['total_scenarios']):,}",
    )

    col2.metric(
        "Patient requests per method",
        f"{int(ai['total_patients']):,}",
    )

    col3.metric(
        "Urgent allocation rate",
        f"{ai['overall_high_priority_rate']:.2f}%",
        delta=f"{improvement:.2f} pp vs FCFS",
    )

    col4.metric(
        "Detected safety violations",
        safety_violations,
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Additional urgent requests selected",
        f"{additional_urgent:,}",
    )

    col6.metric(
        "Overall allocation rate",
        f"{ai['overall_allocation_rate']:.2f}%",
    )

    col7.metric(
        "Better scenarios",
        int(outcomes["ai_better_scenarios"]),
    )

    col8.metric(
        "Mean allocation execution time",
        f"{ai['mean_execution_time_ms']:.2f} ms",
    )

    st.caption(
        "Results represent controlled synthetic "
        "allocation experiments, not real patients "
        "or clinical outcomes."
    )

    plot_data = overall.melt(
        id_vars="method",
        value_vars=[
            "overall_allocation_rate",
            "overall_high_priority_rate",
        ],
        var_name="metric",
        value_name="rate",
    )

    plot_data["metric"] = plot_data[
        "metric"
    ].replace({
        "overall_allocation_rate": "Overall allocation",
        "overall_high_priority_rate": "Urgent allocation",
    })

    fig = px.bar(
        plot_data,
        x="metric",
        y="rate",
        color="method",
        barmode="group",
        color_discrete_map=COLORS,
        text_auto=".2f",
        title="Overall Comparative Performance",
        labels={
            "metric": "Performance metric",
            "rate": "Allocation rate (%)",
            "method": "Allocation method",
        },
    )

    fig.update_yaxes(range=[0, 110])

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        overall,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================
# 3. Controlled scenario evaluation
# ==========================================

def render_controlled_scenarios(data):

    st.subheader("Controlled Hospital Scenarios")

    df = data["scenarios"]

    selected = st.selectbox(
        "Select hospital scenario",
        ["All scenarios"]
        + df["scenario"].unique().tolist(),
        key="evaluation_scenario",
    )

    filtered = df.copy()

    if selected != "All scenarios":

        filtered = filtered[
            filtered["scenario"].eq(selected)
        ]

    metric_options = {
        "High-priority allocation rate":
            "high_priority_allocation_rate",

        "Overall allocation rate":
            "allocation_rate_percent",

        "Recommended allocations":
            "recommended_allocations",

        "Remaining patients":
            "remaining_patients",

        "Projected bed occupancy":
            "projected_occupancy_percent",

        "Execution time (ms)":
            "execution_time_ms",
    }

    metric_label = st.selectbox(
        "Select performance metric",
        list(metric_options.keys()),
        key="scenario_metric",
    )

    metric = metric_options[metric_label]

    fig = px.bar(
        filtered,
        x="scenario",
        y=metric,
        color="method",
        barmode="group",
        color_discrete_map=COLORS,
        text_auto=".2f",
        title=metric_label,
        labels={
            "scenario": "Hospital scenario",
            metric: metric_label,
            "method": "Allocation method",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Both methods use identical bed compatibility "
        "and availability constraints. Their principal "
        "difference is patient ordering."
    )

    st.caption(
        "Projected occupancy assumes recommendations "
        "are approved. It is not observed occupancy."
    )


# ==========================================
# 4. Randomised stress testing
# ==========================================

def render_stress_testing(data):

    st.subheader("Randomised Stress Testing")

    summary = data["stress_summary"]

    paired = data["paired"]

    outcomes = data["outcomes"].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean improvement",
        f"{outcomes['mean_urgent_improvement_pp']:.2f} pp",
    )

    col2.metric(
        "Median improvement",
        f"{outcomes['median_urgent_improvement_pp']:.2f} pp",
    )

    col3.metric(
        "Minimum improvement",
        f"{outcomes['minimum_improvement_pp']:.2f} pp",
    )

    left, right = st.columns(2)

    with left:

        demand = st.selectbox(
            "Patient demand",
            sorted(
                summary["patient_demand"].unique()
            ),
            key="stress_demand",
        )

    with right:

        availability = st.selectbox(
            "Bed availability (%)",
            sorted(
                summary[
                    "bed_availability_percent"
                ].unique()
            ),
            key="stress_availability",
        )

    filtered = summary[
        summary["patient_demand"].eq(demand)
        & summary[
            "bed_availability_percent"
        ].eq(availability)
    ]

    fig = px.bar(
        filtered,
        x="method",
        y="mean_high_priority_rate",
        color="method",
        color_discrete_map=COLORS,
        error_y="std_high_priority_rate",
        text_auto=".2f",
        title=(
            f"Urgent Allocation: {demand} Requests, "
            f"{availability:g}% Bed Availability"
        ),
        labels={
            "method": "Allocation method",
            "mean_high_priority_rate":
                "Mean urgent allocation rate (%)",
        },
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Error bars represent standard deviation "
        "across random seeds, not confidence intervals."
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown("#### Paired Improvement Distribution")

    fig = px.histogram(
        paired,
        x="urgent_allocation_improvement_pp",
        nbins=15,
        title="Urgent Allocation Improvement Over FCFS",
        labels={
            "urgent_allocation_improvement_pp":
                "Improvement (percentage points)",
        },
        color_discrete_sequence=["#2563EB"],
    )

    fig.add_vline(
        x=paired[
            "urgent_allocation_improvement_pp"
        ].mean(),
        line_dash="dash",
        line_color="#DC2626",
        annotation_text="Mean improvement",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    matrix = paired.pivot_table(
        index="patient_demand",
        columns="bed_availability_percent",
        values="urgent_allocation_improvement_pp",
        aggfunc="mean",
    )

    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-100,
        zmax=100,
        title="Improvement by Demand and Bed Availability",
        labels={
            "x": "Bed availability (%)",
            "y": "Patient demand",
            "color": "Improvement (pp)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("#### Paired Scenario Outcomes")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "HospitalFlow AI better",
        int(outcomes["ai_better_scenarios"]),
    )

    col2.metric(
        "Equal",
        int(outcomes["equal_scenarios"]),
    )

    col3.metric(
        "FCFS better",
        int(outcomes["ai_worse_scenarios"]),
    )


# ==========================================
# 5. Scalability analysis
# ==========================================

def render_scalability(data):

    st.subheader("Scalability Analysis")

    df = data["scalability"]

    fig = px.line(
        df,
        x="patient_demand",
        y="mean_execution_time_ms",
        color="method",
        markers=True,
        color_discrete_map=COLORS,
        title="Mean Allocation Execution Time",
        labels={
            "patient_demand": "Patient requests",
            "mean_execution_time_ms":
                "Execution time (ms)",
            "method": "Allocation method",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    fig = px.line(
        df,
        x="patient_demand",
        y="max_execution_time_ms",
        color="method",
        markers=True,
        color_discrete_map=COLORS,
        title="Maximum Recorded Execution Time",
        labels={
            "patient_demand": "Patient requests",
            "max_execution_time_ms":
                "Execution time (ms)",
            "method": "Allocation method",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Execution times measure the allocation "
        "implementations only. They exclude complete "
        "LangGraph execution, database persistence "
        "and dashboard rendering. FCFS also performs "
        "additional copying and timestamp conversion."
    )


# ==========================================
# 6. Allocation safety verification
# ==========================================

def render_safety(data):

    st.subheader("Allocation Safety Verification")

    safety = data["safety"].copy()

    safety.columns = [
        "Safety check",
        "Detected violations",
    ]

    safety["Safety check"] = (
        safety["Safety check"]
        .astype(str)
        .str.replace("_", " ")
        .str.capitalize()
    )

    total = int(
        safety["Detected violations"].sum()
    )

    if total == 0:

        st.success(
            "All tested allocation constraints "
            "were satisfied."
        )

    else:

        st.error(
            f"{total} allocation violations detected."
        )

    for _, row in safety.iterrows():

        left, right = st.columns([4, 1])

        left.write(row["Safety check"])

        violations = int(
            row["Detected violations"]
        )

        if violations == 0:

            right.success("0")

        else:

            right.error(str(violations))

    st.dataframe(
        safety,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Zero detected violations demonstrate "
        "constraint compliance in tested scenarios. "
        "They do not establish real-world clinical safety."
    )


# ==========================================
# 7. Critical evaluation
# ==========================================

def render_critical_evaluation(data):

    st.subheader("Critical Evaluation and Limitations")

    overall = data["stress_overall"].copy()

    overall["High-priority selected"] = (
        overall["urgent_patients_selected"]
    )

    overall["Lower-priority selected"] = (
        overall["total_recommendations"]
        - overall["urgent_patients_selected"]
    )

    plot_data = overall.melt(
        id_vars="method",
        value_vars=[
            "High-priority selected",
            "Lower-priority selected",
        ],
        var_name="priority_group",
        value_name="requests",
    )

    fig = px.bar(
        plot_data,
        x="method",
        y="requests",
        color="priority_group",
        barmode="stack",
        text_auto=True,
        color_discrete_map={
            "High-priority selected": "#2563EB",
            "Lower-priority selected": "#94A3B8",
        },
        title="Patient Prioritisation Trade-Off",
        labels={
            "method": "Allocation method",
            "requests": "Recommended allocations",
            "priority_group": "Priority group",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("""
    #### Key findings

    - Urgency-based ordering improves the selection
      of high-priority patient requests.
    - Both algorithms remain constrained by available
      compatible beds.
    - Prioritisation redistributes existing capacity
      rather than creating additional capacity.

    #### Limitations

    - Strict urgency ordering may disadvantage
      lower-priority patients over time.
    - The experiments use controlled synthetic
      patient requests.
    - Static allocation snapshots cannot establish
      actual reductions in waiting time.
    - Reused random seeds introduce dependence
      between experimental conditions.
    - Allocation benchmarks do not measure the
      complete four-agent workflow.
    - Discharge predictions remain planning estimates
      and must not trigger autonomous discharge.

    #### Future improvements

    - Clinician-governed waiting-time priority ageing.
    - Discrete-event hospital flow simulation.
    - Independent external dataset validation.
    - Improved discharge prediction generalisation.
    - Concurrent allocation and reservation testing.
    """)

    st.divider()

    st.markdown("#### Historical Dataset Limitation")

    historical = data["historical"].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Admission requests",
        f"{int(historical['total_admission_requests']):,}",
    )

    col2.metric(
        "Zero-duration intervals",
        f"{int(historical['zero_waiting_intervals']):,}",
    )

    col3.metric(
        "Positive waiting intervals",
        int(historical["positive_waiting_intervals"]),
    )

    st.warning(
        "The historical dataset contains no measurable "
        "post-admission-decision waiting intervals under "
        "the implemented queue definition. Controlled "
        "scenarios were therefore used to evaluate "
        "allocation prioritisation."
    )


# ==========================================
# 8. Evaluation figure gallery
# ==========================================

def render_figure_gallery():

    st.subheader("Evaluation Figure Gallery")

    figures = sorted(
        FIGURE_DIR.glob("*.png")
    )

    if not figures:

        st.warning(
            "No evaluation figures found. "
            "Run scripts/visualise_system_evaluation.py"
        )

        return

    st.caption(
        f"{len(figures)} evaluation figures available."
    )

    selected = st.selectbox(
        "Select evaluation figure",
        figures,
        format_func=lambda path: (
            path.stem.replace("_", " ").title()
        ),
        key="system_evaluation_figure",
    )

    st.image(
        str(selected),
        use_container_width=True,
    )

    st.download_button(
        label="Download selected figure",
        data=selected.read_bytes(),
        file_name=selected.name,
        mime="image/png",
        key="download_evaluation_figure",
    )


# ==========================================
# 9. Download evaluation results
# ==========================================

def render_downloads():

    st.subheader("Export Evaluation Results")

    st.caption(
        "Download experimental results for academic "
        "reporting and reproducibility."
    )

    files = sorted(
        EVALUATION_DIR.glob("*.csv")
    )

    for index, path in enumerate(files):

        left, right = st.columns([3, 1])

        left.write(
            path.stem.replace("_", " ").title()
        )

        right.download_button(
            label="Download CSV",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            key=f"evaluation_download_{index}",
        )


# ==========================================
# 10. Main system evaluation page
# ==========================================

def system_evaluation_page():

    st.title("System Evaluation")

    st.caption(
        "Comparative Performance, Scalability "
        "and Allocation Constraint Verification"
    )

    try:

        data = load_evaluation_data()

    except FileNotFoundError as error:

        st.error(str(error))

        st.info(
            "Generate the evaluation results using "
            "the following terminal commands."
        )

        st.code(
            "python scripts/investigate_historical_queue.py\n"
            "python scripts/evaluate_system.py\n"
            "python evaluation/stress_testing.py\n"
            "python scripts/visualise_system_evaluation.py",
            language="bash",
        )

        return

    tabs = st.tabs([
        "Overview",
        "Controlled Scenarios",
        "Stress Testing",
        "Scalability",
        "Safety",
        "Critical Evaluation",
        "Figures",
        "Downloads",
    ])

    with tabs[0]:
        render_performance_overview(data)

    with tabs[1]:
        render_controlled_scenarios(data)

    with tabs[2]:
        render_stress_testing(data)

    with tabs[3]:
        render_scalability(data)

    with tabs[4]:
        render_safety(data)

    with tabs[5]:
        render_critical_evaluation(data)

    with tabs[6]:
        render_figure_gallery()

    with tabs[7]:
        render_downloads()