
# HospitalFlow AI
# Shared Hospital Environment

import json
import pandas as pd

from core.config import EXPECTED_TOTAL_BEDS

from core.data_loader import (
    load_bed_events,
    load_coordination_events
)


class HospitalEnvironment:

    def __init__(self):

        self.bed_events = load_bed_events()

        self.management_events = (
            self.bed_events.loc[
                self.bed_events["event_type"]
                == "BED_MANAGEMENT"
            ].copy()
        )

        self.inventory = self.build_inventory()

    # ==========================================
    # 1. Build physical bed inventory
    # ==========================================

    def build_inventory(self):

        inventory = self.management_events[
            ["bed_id", "bed_type"]
        ].dropna().drop_duplicates()

        if inventory["bed_id"].duplicated().any():

            raise ValueError(
                "Conflicting physical bed types."
            )

        inventory = inventory.sort_values(
            "bed_id"
        ).reset_index(drop=True)

        return inventory

    # ==========================================
    # 2. Reconstruct bed states
    # ==========================================

    def get_bed_snapshot(self, simulation_time):

        now = pd.Timestamp(simulation_time)

        events = self.management_events

        snapshot = self.inventory.copy()

        occupied = events.loc[
            (events["occupancy_status"] == "Occupied")
            & (events["assignment_time"] <= now)
            & (events["release_time"] > now)
        ]

        occupied_ids = set(
            occupied["bed_id"].dropna()
        )

        # Check ICU occupancy independently.

        icu = self.bed_events.loc[
            self.bed_events["event_type"] == "ICU_STAY"
        ]

        active_icu = icu.loc[
            (icu["admission_time"] <= now)
            & (icu["discharge_time"] > now)
        ]

        icu_occupied_ids = set(
            active_icu["ICU_bed_id"]
            .dropna()
            .astype(str)
        )

        occupied_ids.update(icu_occupied_ids)

        # Beds currently being cleaned.

        cleaning = events.loc[
            (events["cleaning_start_time"] <= now)
            & (events["cleaning_end_time"] > now)
        ]

        cleaning_ids = set(
            cleaning["bed_id"].dropna()
        )

        # Beds released but awaiting cleaning.

        pending_cleaning = events.loc[
            (events["release_time"] <= now)
            & (events["cleaning_start_time"] > now)
        ]

        pending_cleaning_ids = set(
            pending_cleaning["bed_id"].dropna()
        )

        # Latest recorded completed cleaning.

        completed_cleaning = events.loc[
            events["cleaning_end_time"] <= now
        ].groupby("bed_id")[
            "cleaning_end_time"
        ].max()

        snapshot["last_cleaning_end"] = (
            snapshot["bed_id"].map(
                completed_cleaning
            )
        )

        # Latest observed assignment.

        past_assignments = events.loc[
            events["assignment_time"] <= now
        ].groupby("bed_id")[
            "assignment_time"
        ].max()

        snapshot["last_assignment"] = (
            snapshot["bed_id"].map(
                past_assignments
            )
        )

        # Unknown is the conservative default.

        snapshot["status"] = "Unknown"

        verified_clean = (
            snapshot["last_cleaning_end"].notna()
            & (
                snapshot["last_assignment"].isna()
                | (
                    snapshot["last_cleaning_end"]
                    >= snapshot["last_assignment"]
                )
            )
        )

        snapshot.loc[
            verified_clean,
            "status"
        ] = "Available"

        snapshot.loc[
            snapshot["bed_id"].isin(
                pending_cleaning_ids
            ),
            "status"
        ] = "Awaiting Cleaning"

        snapshot.loc[
            snapshot["bed_id"].isin(cleaning_ids),
            "status"
        ] = "Cleaning"

        snapshot.loc[
            snapshot["bed_id"].isin(occupied_ids),
            "status"
        ] = "Occupied"

        # ICU episodes outside the bed-management
        # inventory are excluded from allocation.

        snapshot["simulation_time"] = str(now)

        snapshot["allocatable"] = (
            snapshot["status"] == "Available"
        )

        return snapshot

    # ==========================================
    # 3. Observe active staffing records
    # ==========================================

    def get_staffing_snapshot(self, simulation_time):

        now = pd.Timestamp(simulation_time)

        events = load_coordination_events()

        staffing = events.loc[
            (events["event_type"] == "STAFFING_SHIFT")
            & (events["shift_start"] <= now)
            & (events["shift_end"] > now)
        ].copy()

        columns = [
            "department",
            "staff_role",
            "shift_type",
            "scheduled_staff_count",
            "active_staff_count",
            "workload_category"
        ]

        return staffing[columns]

    # ==========================================
    # 4. Calculate environmental metrics
    # ==========================================

    def calculate_metrics(self, snapshot):

        total = len(snapshot)

        occupied = int(
            (snapshot["status"] == "Occupied").sum()
        )

        return {
            "documented_capacity":
                EXPECTED_TOTAL_BEDS,

            "observed_capacity": total,

            "excluded_unobserved_beds":
                EXPECTED_TOTAL_BEDS - total,

            "occupied_beds": occupied,

            "available_beds": int(
                (snapshot["status"] == "Available").sum()
            ),

            "cleaning_beds": int(
                (snapshot["status"] == "Cleaning").sum()
            ),

            "awaiting_cleaning_beds": int(
                (
                    snapshot["status"]
                    == "Awaiting Cleaning"
                ).sum()
            ),

            "unknown_beds": int(
                (snapshot["status"] == "Unknown").sum()
            ),

            "occupancy_rate": round(
                occupied / total * 100, 2
            ) if total else 0.0
        }

    # ==========================================
    # 5. Convert to serialisable records
    # ==========================================

    @staticmethod
    def to_records(df):

        return json.loads(
            df.to_json(
                orient="records",
                date_format="iso"
            )
        )

    # ==========================================
    # 6. Initialise shared environment
    # ==========================================

    def initialise(self, state):

        simulation_time = state["simulation_time"]

        beds = self.get_bed_snapshot(
            simulation_time
        )

        staffing = self.get_staffing_snapshot(
            simulation_time
        )

        state["beds"] = self.to_records(beds)

        state["staffing"] = self.to_records(
            staffing
        )

        state.setdefault(
            "metrics", {}
        )["environment"] = self.calculate_metrics(
            beds
        )

        return state
