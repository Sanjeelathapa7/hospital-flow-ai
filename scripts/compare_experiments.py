
# HospitalFlow AI
# Final Experimental Comparison

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 1. Load both experiments
# ==========================================

v1 = pd.read_csv(
    OUTPUT_DIR / "evaluation" /
    "model_comparison.csv"
)

v2 = pd.read_csv(
    OUTPUT_DIR / "evaluation_v2" /
    "model_comparison.csv"
)

v1_test = pd.read_csv(
    OUTPUT_DIR / "evaluation" /
    "test_metrics.csv"
)

v2_test = pd.read_csv(
    OUTPUT_DIR / "evaluation_v2" /
    "test_metrics.csv"
)

v1["Experiment"] = "Version 1"
v2["Experiment"] = "Version 2"

combined = pd.concat(
    [v1, v2],
    ignore_index=True
)

# ==========================================
# 2. Compare validation MAE
# ==========================================

plt.figure(figsize=(12, 7))

sns.barplot(
    data=combined,
    x="MAE_hours",
    y="Model",
    hue="Experiment"
)

plt.title(
    "Validation MAE: Original vs Enhanced Features"
)

plt.xlabel("Mean Absolute Error (Hours)")
plt.ylabel("Regression Model")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "17_experiment_mae_comparison.png",
    dpi=300
)

plt.close()

# ==========================================
# 3. Compare validation R2
# ==========================================

plt.figure(figsize=(12, 7))

sns.barplot(
    data=combined,
    x="R2",
    y="Model",
    hue="Experiment"
)

plt.axvline(
    0,
    color="black",
    linewidth=1
)

plt.title(
    "Validation R²: Original vs Enhanced Features"
)

plt.xlabel("R² Score")
plt.ylabel("Regression Model")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "18_experiment_r2_comparison.png",
    dpi=300
)

plt.close()

# ==========================================
# 4. Compare validation and test errors
# ==========================================

comparison = pd.DataFrame([
    {
        "Experiment": "Version 1",
        "Stage": "Validation",
        "MAE": v1.iloc[0]["MAE_hours"]
    },
    {
        "Experiment": "Version 1",
        "Stage": "Testing",
        "MAE": v1_test.iloc[0]["MAE_hours"]
    },
    {
        "Experiment": "Version 2",
        "Stage": "Validation",
        "MAE": v2.iloc[0]["MAE_hours"]
    },
    {
        "Experiment": "Version 2",
        "Stage": "Testing",
        "MAE": v2_test.iloc[0]["MAE_hours"]
    }
])

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=comparison,
    x="Stage",
    y="MAE",
    hue="Experiment"
)

for container in ax.containers:

    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )

plt.title(
    "Model Generalisation Across Evaluation Periods"
)

plt.ylabel("Mean Absolute Error (Hours)")
plt.xlabel("Evaluation Stage")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "19_model_generalisation.png",
    dpi=300
)

plt.close()

# ==========================================
# 5. Investigate temporal target distribution
# ==========================================

df = pd.read_csv(
    BASE_DIR / "data" / "processed" /
    "ml_v2" / "discharge_training_data.csv",
    usecols=[
        "admission_time",
        "length_of_stay_hours"
    ]
)

df["admission_time"] = pd.to_datetime(
    df["admission_time"],
    format="ISO8601"
)

df["month"] = (
    df["admission_time"]
    .dt.to_period("M")
    .astype(str)
)

monthly = df.groupby("month")[
    "length_of_stay_hours"
].agg([
    "count",
    "mean",
    "median",
    "std"
]).reset_index()

monthly.to_csv(
    OUTPUT_DIR / "evaluation_v2" /
    "monthly_los_summary.csv",
    index=False
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly["month"],
    monthly["mean"],
    marker="o",
    label="Mean LOS"
)

plt.plot(
    monthly["month"],
    monthly["median"],
    marker="s",
    label="Median LOS"
)

plt.xticks(rotation=45)

plt.title("Monthly Length-of-Stay Distribution")

plt.xlabel("Synthetic Operational Month")
plt.ylabel("Length of Stay (Hours)")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "20_monthly_los_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 6. Save final experiment comparison
# ==========================================

comparison.to_csv(
    OUTPUT_DIR / "evaluation_v2" /
    "final_experiment_comparison.csv",
    index=False
)

combined.to_csv(
    OUTPUT_DIR / "evaluation_v2" /
    "all_model_results.csv",
    index=False
)

# ==========================================
# 7. Record operational model policy
# ==========================================

policy = {
    "operational_model": "Dummy Regressor",
    "operational_model_path":
        "outputs/models/best_discharge_model.joblib",
    "experimental_model":
        "Histogram Gradient Boosting",
    "experimental_model_path":
        "outputs/models_v2/best_discharge_model.joblib",
    "reason":
        "Enhanced model failed to outperform the "
        "median baseline on the later test period.",
    "human_review_required": True,
    "autonomous_discharge_allowed": False,
    "intended_use": "Educational simulation only"
}

with open(
    OUTPUT_DIR / "evaluation_v2" /
    "model_selection_policy.json",
    "w"
) as file:

    json.dump(policy, file, indent=4)

# ==========================================
# 8. Display results
# ==========================================

print("\nFINAL EXPERIMENT COMPARISON")
print(comparison.to_string(index=False))

print("\nMONTHLY LOS SUMMARY")
print(monthly.round(2).to_string(index=False))

print("\nOperational model: Dummy Regressor")

print(
    "Experimental model: "
    "Histogram Gradient Boosting"
)

print("\nFINAL MODEL COMPARISON COMPLETED")
