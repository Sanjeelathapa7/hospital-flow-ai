
CREATE TABLE IF NOT EXISTS simulation_runs (

    simulation_id TEXT PRIMARY KEY,
    simulation_time TEXT NOT NULL,
    scenario_mode TEXT NOT NULL,
    workflow_status TEXT NOT NULL,
    approval_status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    final_summary_json TEXT,
    state_json TEXT NOT NULL

);


CREATE TABLE IF NOT EXISTS patient_queue (

    simulation_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    encounter_id TEXT NOT NULL,
    priority INTEGER CHECK(priority BETWEEN 1 AND 5),
    queue_position INTEGER,
    waiting_time_minutes REAL,
    requested_care TEXT,
    requested_bed_type TEXT,
    queue_status TEXT NOT NULL,

    PRIMARY KEY (
        simulation_id,
        patient_id,
        encounter_id
    ),

    FOREIGN KEY (simulation_id)
        REFERENCES simulation_runs(simulation_id)
        ON DELETE CASCADE

);


CREATE TABLE IF NOT EXISTS bed_inventory (

    simulation_id TEXT NOT NULL,
    bed_id TEXT NOT NULL,
    bed_type TEXT NOT NULL,
    status TEXT NOT NULL,
    allocatable INTEGER NOT NULL
        CHECK(allocatable IN (0, 1)),
    assigned_patient_id TEXT,
    assigned_encounter_id TEXT,

    PRIMARY KEY (simulation_id, bed_id),

    FOREIGN KEY (simulation_id)
        REFERENCES simulation_runs(simulation_id)
        ON DELETE CASCADE

);


CREATE TABLE IF NOT EXISTS discharge_predictions (

    simulation_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    encounter_id TEXT NOT NULL,
    model_name TEXT,
    predicted_los_hours REAL
        CHECK(predicted_los_hours >= 0),
    predicted_los_days REAL,
    assumed_admission_time TEXT,
    projected_stay_end TEXT,
    reliability TEXT,
    human_review_required INTEGER,
    autonomous_discharge_allowed INTEGER
        CHECK(autonomous_discharge_allowed = 0),

    PRIMARY KEY (
        simulation_id,
        patient_id,
        encounter_id
    ),

    FOREIGN KEY (simulation_id)
        REFERENCES simulation_runs(simulation_id)
        ON DELETE CASCADE

);


CREATE TABLE IF NOT EXISTS bed_allocations (

    simulation_id TEXT NOT NULL,
    approval_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    encounter_id TEXT NOT NULL,
    bed_id TEXT NOT NULL,
    bed_type TEXT,
    approval_status TEXT NOT NULL
        CHECK(
            approval_status IN (
                'Pending',
                'Approved',
                'Rejected'
            )
        ),
    reviewer TEXT,
    decision_reason TEXT,
    decision_time TEXT,
    predicted_los_hours REAL,

    PRIMARY KEY (simulation_id, approval_id),

    FOREIGN KEY (simulation_id)
        REFERENCES simulation_runs(simulation_id)
        ON DELETE CASCADE,

    FOREIGN KEY (simulation_id, bed_id)
        REFERENCES bed_inventory(simulation_id, bed_id)

);


CREATE TABLE IF NOT EXISTS operational_alerts (

    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id TEXT NOT NULL,
    agent_id TEXT,
    alert_type TEXT,
    severity TEXT,
    message TEXT,
    patient_id TEXT,
    bed_id TEXT,
    department TEXT,
    details_json TEXT,

    FOREIGN KEY (simulation_id)
        REFERENCES simulation_runs(simulation_id)
        ON DELETE CASCADE

);


CREATE TABLE IF NOT EXISTS audit_logs (

    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    approval_id TEXT NOT NULL,
    patient_id TEXT,
    encounter_id TEXT,
    bed_id TEXT,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reason TEXT NOT NULL,
    decision_time TEXT NOT NULL,

    UNIQUE (simulation_id, approval_id),

    FOREIGN KEY (simulation_id, approval_id)
        REFERENCES bed_allocations(
            simulation_id,
            approval_id
        )

);


CREATE TABLE IF NOT EXISTS agent_messages (

    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id TEXT NOT NULL,
    sender TEXT,
    receiver TEXT,
    message_type TEXT,
    simulation_time TEXT,
    payload_json TEXT,

    FOREIGN KEY (simulation_id)
        REFERENCES simulation_runs(simulation_id)
        ON DELETE CASCADE

);


-- Prevent multiple approved allocations
-- for the same bed within one simulation.

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_unique_approved_bed

ON bed_allocations (
    simulation_id,
    bed_id
)

WHERE approval_status = 'Approved';


-- Prevent multiple approved allocations
-- for the same patient encounter.

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_unique_approved_encounter

ON bed_allocations (
    simulation_id,
    patient_id,
    encounter_id
)

WHERE approval_status = 'Approved';


CREATE INDEX IF NOT EXISTS idx_alert_severity

ON operational_alerts (
    simulation_id,
    severity
);


CREATE INDEX IF NOT EXISTS idx_patient_priority

ON patient_queue (
    simulation_id,
    priority
);


CREATE INDEX IF NOT EXISTS idx_bed_status

ON bed_inventory (
    simulation_id,
    status
);
