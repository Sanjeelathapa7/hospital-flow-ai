from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "outputs" / "system_evaluation"

FIGURE_DIR = DATA_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# VISUALISATION CONFIGURATION
# -------------------------------------------------

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
})

METHODS = ["FCFS", "HospitalFlow AI"]

PALETTE = {
    "FCFS": "#64748B",
    "HospitalFlow AI": "#2563EB",
}


# -------------------------------------------------
# LOAD EVALUATION RESULTS
# -------------------------------------------------

def load_results():

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
    }

    results = {}

    for name, filename in files.items():

        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required evaluation file missing: {path}"
            )

        results[name] = pd.read_csv(path)

        print(
            f"Loaded {filename}: "
            f"{len(results[name])} records"
        )

    return results


# -------------------------------------------------
# SAVE FIGURES
# -------------------------------------------------

def save_figure(fig, filename):

    path = FIGURE_DIR / filename

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


def label_bars(ax, decimals=1, suffix=""):

    for container in ax.containers:

        if hasattr(container, "datavalues"):

            labels = [
                f"{value:.{decimals}f}{suffix}"
                if np.isfinite(value) else ""
                for value in container.datavalues
            ]

            ax.bar_label(
                container,
                labels=labels,
                padding=3,
                fontsize=9,
            )


# -------------------------------------------------
# FIGURE 21: SCENARIO URGENT ALLOCATION
# -------------------------------------------------

def plot_scenario_priority(df):

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df,
        x="scenario",
        y="high_priority_allocation_rate",
        hue="method",
        hue_order=METHODS,
        palette=PALETTE,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(
        "High-Priority Allocation Across Controlled Scenarios"
    )

    ax.set_xlabel("Hospital scenario")
    ax.set_ylabel("High-priority allocation rate (%)")
    ax.set_ylim(0, 115)

    ax.tick_params(axis="x", rotation=20)

    label_bars(ax, suffix="%")

    ax.legend(title="Allocation method")

    save_figure(
        fig,
        "21_scenario_priority_comparison.png",
    )


# -------------------------------------------------
# FIGURE 22: OVERALL ALLOCATION RATE
# -------------------------------------------------

def plot_scenario_allocation(df):

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df,
        x="scenario",
        y="allocation_rate_percent",
        hue="method",
        hue_order=METHODS,
        palette=PALETTE,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(
        "Overall Bed Allocation Across Controlled Scenarios"
    )

    ax.set_xlabel("Hospital scenario")
    ax.set_ylabel("Recommended allocation rate (%)")
    ax.set_ylim(0, 115)

    ax.tick_params(axis="x", rotation=20)

    label_bars(ax, suffix="%")

    save_figure(
        fig,
        "22_scenario_allocation_comparison.png",
    )


# -------------------------------------------------
# FIGURE 23: OVERALL STRESS TEST COMPARISON
# -------------------------------------------------

def plot_stress_overall(df):

    plot_data = df.melt(
        id_vars="method",
        value_vars=[
            "overall_allocation_rate",
            "overall_high_priority_rate",
        ],
        var_name="metric",
        value_name="rate",
    )

    plot_data["metric"] = plot_data["metric"].replace({
        "overall_allocation_rate": "Overall allocation",
        "overall_high_priority_rate": "Urgent allocation",
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.barplot(
        data=plot_data,
        x="metric",
        y="rate",
        hue="method",
        hue_order=METHODS,
        palette=PALETTE,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(
        "Overall Performance Across 150 Stress Scenarios"
    )

    ax.set_xlabel("")
    ax.set_ylabel("Allocation rate (%)")
    ax.set_ylim(0, 100)

    label_bars(ax, decimals=2, suffix="%")

    save_figure(
        fig,
        "23_stress_overall_comparison.png",
    )


# -------------------------------------------------
# FIGURE 24: URGENT PATIENT SELECTION
# -------------------------------------------------

def plot_urgent_patients(df):

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        data=df,
        x="method",
        y="urgent_patients_selected",
        hue="method",
        palette=PALETTE,
        legend=False,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(
        "High-Priority Patient Requests Selected"
    )

    ax.set_xlabel("Allocation method")
    ax.set_ylabel("Number of selected requests")

    ax.set_ylim(
        0,
        df["urgent_patients_selected"].max() * 1.2,
    )

    label_bars(ax, decimals=0)

    save_figure(
        fig,
        "24_urgent_patient_selection.png",
    )


# -------------------------------------------------
# FIGURE 25: SCALABILITY
# -------------------------------------------------

def plot_scalability(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="patient_demand",
        y="mean_execution_time_ms",
        hue="method",
        hue_order=METHODS,
        palette=PALETTE,
        marker="o",
        linewidth=2.5,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(
        "Allocation Algorithm Scalability"
    )

    ax.set_xlabel("Number of patient requests")
    ax.set_ylabel("Mean execution time (ms)")

    ax.set_xticks(
        sorted(df["patient_demand"].unique())
    )

    ax.set_ylim(bottom=0)

    save_figure(
        fig,
        "25_scalability_comparison.png",
    )


# -------------------------------------------------
# FIGURE 26: PAIRED IMPROVEMENT DISTRIBUTION
# -------------------------------------------------

def plot_improvement_distribution(df):

    improvement = df[
        "urgent_allocation_improvement_pp"
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        improvement,
        bins=15,
        color="#2563EB",
        edgecolor="white",
        alpha=0.85,
    )

    mean = improvement.mean()

    ax.axvline(
        mean,
        color="#DC2626",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean:.2f} pp",
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
    )

    ax.set_title(
        "Distribution of Urgent Allocation Improvements"
    )

    ax.set_xlabel(
        "HospitalFlow AI improvement over FCFS "
        "(percentage points)"
    )

    ax.set_ylabel("Number of scenarios")
    ax.legend()

    save_figure(
        fig,
        "26_paired_improvement_distribution.png",
    )


# -------------------------------------------------
# FIGURE 27: PERFORMANCE BY BED AVAILABILITY
# -------------------------------------------------

def plot_availability_performance(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="bed_availability_percent",
        y="high_priority_allocation_rate",
        hue="method",
        hue_order=METHODS,
        palette=PALETTE,
        marker="o",
        linewidth=2.5,
        estimator="mean",
        errorbar=None,
        ax=ax,
    )

    ax.set_title(
        "Urgent Allocation Under Different Bed Availability Levels"
    )

    ax.set_xlabel("Available proportion of bed inventory (%)")

    ax.set_ylabel(
        "Mean high-priority allocation rate (%)"
    )

    ax.set_xticks(
        sorted(df["bed_availability_percent"].unique())
    )

    ax.set_ylim(0, 105)

    save_figure(
        fig,
        "27_bed_availability_performance.png",
    )


# -------------------------------------------------
# FIGURE 28: IMPROVEMENT HEATMAP
# -------------------------------------------------

def plot_improvement_heatmap(df):

    matrix = df.pivot_table(
        index="patient_demand",
        columns="bed_availability_percent",
        values="urgent_allocation_improvement_pp",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        center=0,
        vmin=-100,
        vmax=100,
        linewidths=0.5,
        cbar_kws={
            "label": "Improvement (percentage points)"
        },
        ax=ax,
    )

    ax.set_title(
        "Mean Urgent Allocation Improvement by Demand and Capacity"
    )

    ax.set_xlabel("Bed availability (%)")
    ax.set_ylabel("Patient demand")

    save_figure(
        fig,
        "28_priority_improvement_heatmap.png",
    )


# -------------------------------------------------
# FIGURE 29: PAIRED SCENARIO OUTCOMES
# -------------------------------------------------

def plot_paired_outcomes(df):

    record = df.iloc[0]

    labels = [
        "HospitalFlow AI better",
        "Equal performance",
        "FCFS better",
    ]

    values = [
        record["ai_better_scenarios"],
        record["equal_scenarios"],
        record["ai_worse_scenarios"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        labels,
        values,
        color=[
            "#16A34A",
            "#94A3B8",
            "#DC2626",
        ],
    )

    ax.bar_label(
        bars,
        fmt="%.0f",
        padding=5,
    )

    ax.set_title(
        "Paired Scenario Outcomes: Urgent Allocation"
    )

    ax.set_xlabel("Number of scenarios")
    ax.set_xlim(0, max(values) * 1.15)
    ax.invert_yaxis()

    save_figure(
        fig,
        "29_paired_scenario_outcomes.png",
    )


# -------------------------------------------------
# FIGURE 30: SAFETY VERIFICATION
# -------------------------------------------------

def plot_safety(df):

    names = (
        df.iloc[:, 0]
        .astype(str)
        .str.replace("_", " ")
        .str.capitalize()
    )

    values = df["violations"]

    fig, ax = plt.subplots(figsize=(10, 5))

    positions = np.arange(len(names))

    ax.scatter(
        values,
        positions,
        color="#16A34A",
        s=120,
        zorder=3,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(names)

    ax.set_xlim(
        -0.1,
        max(1, values.max() + 1),
    )

    for position, value in zip(positions, values):

        ax.annotate(
            str(value),
            (value, position),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
        )

    ax.set_title(
        "Allocation Safety Verification Across Stress Tests"
    )

    ax.set_xlabel("Detected violations")
    ax.invert_yaxis()

    save_figure(
        fig,
        "30_allocation_safety_verification.png",
    )


# -------------------------------------------------
# FIGURE 31: ALLOCATION TRADE-OFF
# -------------------------------------------------

def plot_priority_tradeoff(df):

    plot_data = df.copy()

    plot_data["High-priority selected"] = (
        plot_data["urgent_patients_selected"]
    )

    plot_data["Lower-priority selected"] = (
        plot_data["total_recommendations"]
        - plot_data["urgent_patients_selected"]
    )

    plot_data = plot_data.set_index("method")

    fig, ax = plt.subplots(figsize=(9, 6))

    plot_data.loc[
        METHODS,
        [
            "High-priority selected",
            "Lower-priority selected",
        ],
    ].plot(
        kind="bar",
        stacked=True,
        color=["#2563EB", "#CBD5E1"],
        rot=0,
        ax=ax,
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            label_type="center",
            fmt="%.0f",
        )

    ax.set_title(
        "Resource Distribution and Prioritisation Trade-Off"
    )

    ax.set_xlabel("Allocation method")
    ax.set_ylabel("Recommended allocations")

    ax.legend(title="Patient priority group")

    save_figure(
        fig,
        "31_priority_allocation_tradeoff.png",
    )


# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------

def main():

    print("=" * 70)
    print("HOSPITALFLOW AI - SYSTEM EVALUATION VISUALISATION")
    print("=" * 70)

    results = load_results()

    print("\nGenerating evaluation figures...\n")

    plot_scenario_priority(results["scenarios"])

    plot_scenario_allocation(results["scenarios"])

    plot_stress_overall(results["stress_overall"])

    plot_urgent_patients(results["stress_overall"])

    plot_scalability(results["scalability"])

    plot_improvement_distribution(results["paired"])

    plot_availability_performance(results["stress"])

    plot_improvement_heatmap(results["paired"])

    plot_paired_outcomes(results["outcomes"])

    plot_safety(results["safety"])

    plot_priority_tradeoff(results["stress_overall"])

    print("\n" + "=" * 70)
    print("VISUALISATION COMPLETED")
    print("=" * 70)

    print("Total figures generated: 11")

    print("Figures saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()