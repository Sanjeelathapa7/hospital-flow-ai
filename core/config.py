
# HospitalFlow AI
# Central System Configuration

from pathlib import Path

# ==========================================
# 1. Project information
# ==========================================

PROJECT_NAME = "HospitalFlow AI"

PROJECT_TITLE = (
    "Multi-Agent Hospital Patient Flow "
    "and Bed Management System"
)

VERSION = "1.0.0"

RANDOM_STATE = 42

# ==========================================
# 2. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

AGENT_DATA_DIR = PROCESSED_DIR / "agents"

ML_DIR = PROCESSED_DIR / "ml"

ML_V2_DIR = PROCESSED_DIR / "ml_v2"

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

EXPERIMENTAL_MODEL_DIR = OUTPUT_DIR / "models_v2"

DATABASE_DIR = BASE_DIR / "database"

DATABASE_PATH = DATABASE_DIR / "hospital_flow.db"

# ==========================================
# 3. Agent dataset paths
# ==========================================

PATIENT_FILE = AGENT_DATA_DIR / "patient_features.csv"

ADMISSION_FILE = AGENT_DATA_DIR / "admission_events.csv"

BED_FILE = AGENT_DATA_DIR / "bed_events.csv"

DISCHARGE_FILE = AGENT_DATA_DIR / "discharge_events.csv"

COORDINATION_FILE = (
    AGENT_DATA_DIR / "coordination_events.csv"
)

# ==========================================
# 4. Model configuration
# ==========================================

OPERATIONAL_MODEL_PATH = (
    MODEL_DIR / "best_discharge_model.joblib"
)

EXPERIMENTAL_MODEL_PATH = (
    EXPERIMENTAL_MODEL_DIR /
    "best_discharge_model.joblib"
)

# ==========================================
# 5. Hospital capacity
# ==========================================

EXPECTED_BED_CAPACITY = {
    "medical_ward": 680,
    "surgical_ward": 220,
    "shared": 260,
    "private": 130,
    "ICU": 175,
    "step_down": 55,
    "isolation": 130,
    "ED_observation": 40,
    "pediatric": 30
}

EXPECTED_TOTAL_BEDS = sum(
    EXPECTED_BED_CAPACITY.values()
)

# ==========================================
# 6. Admission priority configuration
# ==========================================

PRIORITY_LABELS = {
    1: "Highest urgency",
    2: "High urgency",
    3: "Moderate urgency",
    4: "Low urgency",
    5: "Lowest urgency"
}

# ==========================================
# 7. Operational alert thresholds
# ==========================================

BED_OCCUPANCY_WARNING = 0.85

BED_OCCUPANCY_CRITICAL = 0.95

WAITING_TIME_WARNING_MINUTES = 120

WAITING_TIME_CRITICAL_MINUTES = 240

# ==========================================
# 8. Simulation configuration
# ==========================================

DEFAULT_SIMULATION_TIME = "2031-10-03 12:00:00"

HUMAN_APPROVAL_REQUIRED = True

AUTONOMOUS_DISCHARGE_ALLOWED = False

# ==========================================
# 9. Ethical and safety configuration
# ==========================================

DATASET_TYPE = "Synthetic"

DISCLAIMER = (
    "HospitalFlow AI is an educational simulation "
    "using synthetic hospital data. "
    "It is not intended for clinical use."
)
