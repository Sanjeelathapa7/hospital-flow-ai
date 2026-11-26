
# Project: HospitalFlow AI
# Phase 5.1 - Machine Learning Model Training

from pathlib import Path
from time import perf_counter
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. Define project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

ML_DIR = BASE_DIR / "data" / "processed" / "ml_v2"

MODEL_DIR = BASE_DIR / "outputs" / "models_v2"

EVALUATION_DIR = (
    BASE_DIR / "outputs" / "evaluation_v2"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ==========================================
# 2. Load dataset and feature configuration
# ==========================================

df = pd.read_csv(
    ML_DIR / "discharge_training_data.csv",
    dtype={
        "patient_id": "string",
        "encounter_id": "string"
    }
)

with open(
    ML_DIR / "feature_config.json",
    "r"
) as file:
    config = json.load(file)

TARGET = config["target"]

numeric_features = config["numeric_features"]

categorical_features = config[
    "categorical_features"
]

FEATURES = numeric_features + categorical_features

print("Dataset shape:", df.shape)
print("Input features:", len(FEATURES))
print("Prediction target:", TARGET)

# ==========================================
# 3. Validate model inputs
# ==========================================

required_columns = FEATURES + [
    TARGET,
    "patient_id",
    "encounter_id",
    "admission_time",
    "discharge_time"
]

missing_columns = (
    set(required_columns) - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

for column in ["admission_time", "discharge_time"]:

    df[column] = pd.to_datetime(
        df[column],
        format="ISO8601",
        errors="raise"
    )

if df[
    ["admission_time", "discharge_time"]
].isna().any().any():
    raise ValueError("Missing timestamps detected.")

if df.duplicated(
    ["patient_id", "encounter_id"]
).any():
    raise ValueError("Duplicate encounters detected.")

if not (
    np.isfinite(df[TARGET])
    & (df[TARGET] > 0)
).all():
    raise ValueError("Invalid target values detected.")

df = df.sort_values(
    "admission_time"
).reset_index(drop=True)

# ==========================================
# 4. Chronological data splitting
# ==========================================

n = len(df)

validation_start = df.iloc[
    int(n * 0.70)
]["admission_time"]

test_start = df.iloc[
    int(n * 0.85)
]["admission_time"]

# Only use completed stays whose outcomes
# were known before the next evaluation period.

train = df.loc[
    (df["admission_time"] < validation_start)
    & (df["discharge_time"] < validation_start)
].copy()

validation = df.loc[
    (df["admission_time"] >= validation_start)
    & (df["admission_time"] < test_start)
    & (df["discharge_time"] < test_start)
].copy()

test = df.loc[
    df["admission_time"] >= test_start
].copy()

if any(
    split.empty
    for split in [train, validation, test]
):
    raise ValueError("An evaluation split is empty.")

excluded_rows = n - (
    len(train) + len(validation) + len(test)
)

print("\n" + "=" * 60)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 60)

print("Training records:", len(train))
print("Validation records:", len(validation))
print("Testing records:", len(test))

print(
    "Boundary-crossing encounters excluded:",
    excluded_rows
)

print("Validation begins:", validation_start)
print("Testing begins:", test_start)

# Check repeated patients across periods

train_patients = set(train["patient_id"])
validation_patients = set(validation["patient_id"])
test_patients = set(test["patient_id"])

print(
    "Train/test patient overlap:",
    len(train_patients & test_patients)
)

print(
    "Train/validation patient overlap:",
    len(train_patients & validation_patients)
)

split_summary = pd.DataFrame([
    {
        "split": name,
        "records": len(data),
        "unique_patients": data["patient_id"].nunique(),
        "start_date": data["admission_time"].min(),
        "end_date": data["admission_time"].max()
    }
    for name, data in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test)
    ]
])

split_summary.to_csv(
    EVALUATION_DIR / "data_split_summary.csv",
    index=False
)

# ==========================================
# 5. Separate inputs and targets
# ==========================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# ==========================================
# 6. Create preprocessing pipeline
# ==========================================

def create_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
SimpleImputer(
    strategy="median",
    add_indicator=True
)        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

# ==========================================
# 7. Define regression models
# ==========================================


models = {

    "Dummy Regressor": DummyRegressor(
        strategy="median"
    ),

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=20,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        loss="absolute_error",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        random_state=RANDOM_STATE
    ),

    "Histogram Gradient Boosting":
        HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
}


# ==========================================
# 8. Evaluation function
# ==========================================

def calculate_metrics(actual, predicted):

    return {
        "MAE_hours": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE_hours": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "R2": float(
            r2_score(actual, predicted)
        )
    }

# ==========================================
# 9. Train and validate each model
# ==========================================

results = []
trained_models = {}

validation_predictions = validation[
    ["patient_id", "encounter_id", TARGET]
].copy()

for name, model in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            model
        )
    ])

    start_time = perf_counter()

    pipeline.fit(X_train, y_train)

    training_time = perf_counter() - start_time

    start_time = perf_counter()

    predictions = pipeline.predict(X_val)

    # A length of stay cannot be negative.
    predictions = np.maximum(predictions, 0)

    prediction_time = perf_counter() - start_time

    metrics = calculate_metrics(
        y_val,
        predictions
    )

    result = {
        "Model": name,
        **metrics,
        "Training_time_seconds": training_time,
        "Prediction_time_seconds": prediction_time
    }

    results.append(result)

    trained_models[name] = pipeline

    validation_predictions[name] = predictions

    print("MAE:", round(metrics["MAE_hours"], 4))
    print("RMSE:", round(metrics["RMSE_hours"], 4))
    print("R2:", round(metrics["R2"], 4))

    print(
        "Training time:",
        round(training_time, 2),
        "seconds"
    )

# ==========================================
# 10. Compare validation results
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE_hours"
).reset_index(drop=True)

results_df.to_csv(
    EVALUATION_DIR / "model_comparison.csv",
    index=False
)

validation_predictions.to_csv(
    EVALUATION_DIR / "validation_predictions.csv",
    index=False
)

print("\n" + "=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

# ==========================================
# 11. Select best model using validation MAE
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_pipeline = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# ==========================================
# 12. Refit selected model before testing
# ==========================================

# Include all historical encounters with
# completed outcomes before the test period.

development = df.loc[
    (df["admission_time"] < test_start)
    & (df["discharge_time"] < test_start)
].copy()

best_pipeline.fit(
    development[FEATURES],
    development[TARGET]
)

# ==========================================
# 13. Evaluate independent test dataset
# ==========================================

test_predictions = np.maximum(
    best_pipeline.predict(X_test),
    0
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions
)

test_results = {
    "Model": best_model_name,
    **test_metrics,
    "Test_records": len(test),
    "Development_records": len(development)
}

# Compare performance with a median baseline.

baseline_predictions = np.full(
    len(test),
    development[TARGET].median()
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions
)

test_comparison = pd.DataFrame([
    test_results,
    {
        "Model": "Median Baseline",
        **baseline_metrics,
        "Test_records": len(test),
        "Development_records": len(development)
    }
])

test_comparison.to_csv(
    EVALUATION_DIR / "test_metrics.csv",
    index=False
)

# ==========================================
# 14. Evaluate previously unseen patients
# ==========================================

development_patients = set(
    development["patient_id"]
)

unseen_mask = ~test["patient_id"].isin(
    development_patients
)

unseen_count = int(unseen_mask.sum())

if unseen_count >= 2:

    unseen_metrics = calculate_metrics(
        y_test.loc[unseen_mask],
        test_predictions[unseen_mask.to_numpy()]
    )

    unseen_results = {
        "Records": unseen_count,
        **unseen_metrics
    }

else:

    unseen_results = {
        "Records": unseen_count,
        "Note": "Insufficient unseen patient records"
    }

with open(
    EVALUATION_DIR / "unseen_patient_metrics.json",
    "w"
) as file:

    json.dump(unseen_results, file, indent=4)

# ==========================================
# 15. Save test predictions
# ==========================================

prediction_df = test[
    [
        "patient_id",
        "encounter_id",
        "admission_time",
        TARGET
    ]
].copy()

prediction_df = prediction_df.rename(
    columns={TARGET: "actual_los_hours"}
)

prediction_df["predicted_los_hours"] = (
    test_predictions
)

prediction_df["absolute_error_hours"] = np.abs(
    prediction_df["actual_los_hours"]
    - prediction_df["predicted_los_hours"]
)

prediction_df["unseen_patient"] = (
    unseen_mask.to_numpy()
)

prediction_df.to_csv(
    EVALUATION_DIR / "test_predictions.csv",
    index=False
)

# ==========================================
# 16. Save best model
# ==========================================

model_path = (
    MODEL_DIR / "best_discharge_model.joblib"
)

joblib.dump(best_pipeline, model_path)

model_metadata = {
    "model_name": best_model_name,
    "target": TARGET,
    "features": FEATURES,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "random_state": RANDOM_STATE,
    "selection_metric": "Validation MAE",
    "validation_start": str(validation_start),
    "test_start": str(test_start),
    "test_metrics": test_metrics,
    "prediction_unit": "hours",
    "minimum_prediction": 0,
    "dataset_type": "synthetic",
    "intended_use": "Educational simulation only"
}

with open(
    MODEL_DIR / "model_metadata.json",
    "w"
) as file:

    json.dump(model_metadata, file, indent=4)

# ==========================================
# 17. Display final results
# ==========================================

print("\n" + "=" * 60)
print("INDEPENDENT TEST RESULTS")
print("=" * 60)

print(
    test_comparison.round(4).to_string(index=False)
)

print("\nUnseen patient evaluation:")
print(unseen_results)

print("\nBest model saved to:")
print(model_path)

print("\nMODEL TRAINING COMPLETED")
