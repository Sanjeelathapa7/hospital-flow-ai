
import pandas as pd
from pathlib import Path

# Define dataset location
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

# Load patient records
patients = pd.read_csv(RAW_DIR / "cross_sectional.csv")

# Load a sample of hospital events
events = pd.read_csv(
    RAW_DIR / "longitudinal.csv",
    nrows=10000
)

# Load data dictionary
dictionary = pd.read_csv(
    RAW_DIR / "DATA_DICTIONARY (1).csv"
)

# Inspect patient dataset
print("\n--- PATIENT DATASET ---")
print("Shape:", patients.shape)
print("\nColumns:")
print(patients.columns.tolist())
print("\nFirst 5 records:")
print(patients.head())

# Inspect hospital event dataset
print("\n--- HOSPITAL EVENT DATASET ---")
print("Sample shape:", events.shape)
print("\nColumns:")
print(events.columns.tolist())
print("\nFirst 5 records:")
print(events.head())

# Inspect event categories
print("\n--- EVENT TYPES ---")
print(events["event_type"].value_counts())

# Inspect data dictionary
print("\n--- DATA DICTIONARY ---")
print(dictionary.head())
