
# HospitalFlow AI
# Phase 5.2 - Model Evaluation and Visualisation

from pathlib import Path
import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Define project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
MODEL_DIR = BASE_DIR / "outputs" / "models"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 2. Load model evaluation results
# ==========================================

comparison = pd.read_csv(
    EVAL_DIR / "model_comparison.csv"
)

predictions = pd.read_csv(
    EVAL_DIR / "test_predictions.csv"
)

test_metrics = pd.read_csv(
    EVAL_DIR / "test_metrics.csv"
)

with open(
    MODEL_DIR / "model_metadata.json",
    "r"
) as file:

    metadata = json.load(file)

best_model = metadata["model_name"]

print("Selected model:", best_model)

def save_figure(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", filename)

# ==========================================
# 3. MAE comparison
# ==========================================

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=comparison,
    x="MAE_hours",
    y="Model",
    color="steelblue"
)

for container in ax.containers:

    ax.bar_label(container, fmt="%.2f", padding=3)

plt.title("Validation MAE Comparison")
plt.xlabel("Mean Absolute Error (Hours)")
plt.ylabel("Regression Model")

plt.xlim(
    0,
    comparison["MAE_hours"].max() * 1.15
)

save_figure("06_model_mae_comparison.png")

# ==========================================
# 4. RMSE comparison
# ==========================================

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=comparison,
    x="RMSE_hours",
    y="Model",
    color="darkorange"
)

for container in ax.containers:

    ax.bar_label(container, fmt="%.2f", padding=3)

plt.title("Validation RMSE Comparison")
plt.xlabel("Root Mean Squared Error (Hours)")
plt.ylabel("Regression Model")

plt.xlim(
    0,
    comparison["RMSE_hours"].max() * 1.15
)

save_figure("07_model_rmse_comparison.png")

# ==========================================
# 5. R-squared comparison
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="R2",
    y="Model",
    color="seagreen"
)

plt.axvline(0, color="black", linewidth=1)

plt.title("Validation R-Squared Comparison")
plt.xlabel("R² Score")
plt.ylabel("Regression Model")

save_figure("08_model_r2_comparison.png")

# ==========================================
# 6. Actual vs predicted LOS
# ==========================================

sample = predictions.sample(
    n=min(5000, len(predictions)),
    random_state=42
)

plt.figure(figsize=(9, 7))

plt.scatter(
    sample["actual_los_hours"],
    sample["predicted_los_hours"],
    alpha=0.25,
    s=15
)

max_value = max(
    sample["actual_los_hours"].max(),
    sample["predicted_los_hours"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    "r--",
    label="Perfect prediction"
)

plt.title(
    f"Actual vs Predicted LOS - {best_model}"
)

plt.xlabel("Actual Length of Stay (Hours)")
plt.ylabel("Predicted Length of Stay (Hours)")
plt.legend()

save_figure("09_actual_vs_predicted.png")

# ==========================================
# 7. Residual distribution
# ==========================================

predictions["residual"] = (
    predictions["actual_los_hours"]
    - predictions["predicted_los_hours"]
)

plt.figure(figsize=(10, 6))

sns.histplot(
    predictions["residual"],
    bins=60,
    kde=True,
    color="steelblue"
)

plt.axvline(
    0,
    color="red",
    linestyle="--"
)

plt.title("Test Prediction Residual Distribution")
plt.xlabel("Actual Minus Predicted LOS (Hours)")
plt.ylabel("Frequency")

save_figure("10_residual_distribution.png")

# ==========================================
# 8. Actual and predicted LOS distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    predictions["actual_los_hours"],
    bins=60,
    color="steelblue",
    alpha=0.5,
    label="Actual LOS"
)

sns.histplot(
    predictions["predicted_los_hours"],
    bins=60,
    color="darkorange",
    alpha=0.5,
    label="Predicted LOS"
)

plt.title("Actual and Predicted LOS Distribution")
plt.xlabel("Length of Stay (Hours)")
plt.ylabel("Frequency")
plt.legend()

save_figure("11_los_distribution.png")

# ==========================================
# 9. Training time comparison
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="Training_time_seconds",
    y="Model",
    color="mediumpurple"
)

plt.title("Model Training Time Comparison")
plt.xlabel("Training Time (Seconds)")
plt.ylabel("Regression Model")

save_figure("12_training_time_comparison.png")

# ==========================================
# 10. Feature importance
# ==========================================

pipeline = joblib.load(
    MODEL_DIR / "best_discharge_model.joblib"
)

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps["model"]

feature_names = (
    preprocessor.get_feature_names_out()
)

if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    importance_label = "Impurity-based Importance"

elif hasattr(model, "coef_"):

    importance = np.abs(model.coef_).ravel()

    importance_label = "Absolute Coefficient Magnitude"

else:

    importance = None

if importance is not None:

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_df.to_csv(
        EVAL_DIR / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=importance_df.head(15),
        x="importance",
        y="feature",
        color="teal"
    )

    plt.title(
        f"Top 15 Model Features - {best_model}"
    )

    plt.xlabel(importance_label)
    plt.ylabel("Transformed Feature")

    save_figure("13_feature_importance.png")

else:

    print(
        "Feature importance is not available "
        "for the selected model."
    )

# ==========================================
# 11. Display final evaluation results
# ==========================================

print("\nVALIDATION RESULTS")
print(comparison.round(4).to_string(index=False))

print("\nTEST RESULTS")
print(test_metrics.round(4).to_string(index=False))

print("\nMODEL EVALUATION COMPLETED")
