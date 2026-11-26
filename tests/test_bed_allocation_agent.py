
# Automated Tests for Agent 2

from agents.bed_allocation_agent import (
    BedAllocationAgent
)

from core.state import create_initial_state


def make_patient(
    patient_id,
    priority=3,
    bed_type="medical ward",
    care="inpatient",
    wait=60
):

    return {

        "patient_id": patient_id,

        "encounter_id": f"ENC-{patient_id}",

        "priority": priority,

        "waiting_time_minutes": wait,

        "requested_care": care,

        "requested_bed_type": bed_type
    }


def make_bed(
    bed_id,
    bed_type="medical ward",
    status="Available"
):

    return {

        "bed_id": bed_id,

        "bed_type": bed_type,

        "status": status,

        "allocatable":
            status == "Available"
    }


def test_available_bed_allocation():

    agent = BedAllocationAgent()

    patients = [make_patient("P001")]

    beds = [make_bed("B001")]

    allocated, waiting = agent.allocate(
        patients, beds
    )

    assert len(allocated) == 1

    assert len(waiting) == 0


def test_no_duplicate_allocation():

    agent = BedAllocationAgent()

    patients = [
        make_patient("P001"),
        make_patient("P002")
    ]

    beds = [make_bed("B001")]

    allocated, waiting = agent.allocate(
        patients, beds
    )

    assert len(allocated) == 1

    assert len(waiting) == 1


def test_priority_respected():

    agent = BedAllocationAgent()

    patients = [
        make_patient("P001", priority=5),
        make_patient("P002", priority=1)
    ]

    beds = [make_bed("B001")]

    allocated, waiting = agent.allocate(
        patients, beds
    )

    assert allocated[0]["patient_id"] == "P002"


def test_waiting_time_tie_break():

    agent = BedAllocationAgent()

    patients = [
        make_patient("P001", wait=30),
        make_patient("P002", wait=180)
    ]

    beds = [make_bed("B001")]

    allocated, waiting = agent.allocate(
        patients, beds
    )

    assert allocated[0]["patient_id"] == "P002"


def test_icu_compatibility():

    agent = BedAllocationAgent()

    patients = [

        make_patient(
            "P001",
            bed_type="ICU",
            care="ICU"
        )
    ]

    beds = [
        make_bed("B001", "medical ward"),
        make_bed("ICU001", "ICU")
    ]

    allocated, waiting = agent.allocate(
        patients, beds
    )

    assert allocated[0]["bed_id"] == "ICU001"


def test_incompatible_bed_rejected():

    agent = BedAllocationAgent()

    patients = [

        make_patient(
            "P001",
            bed_type="ICU",
            care="ICU"
        )
    ]

    beds = [make_bed("B001")]

    allocated, waiting = agent.allocate(
        patients, beds
    )

    assert len(allocated) == 0

    assert len(waiting) == 1


def test_occupied_bed_excluded():

    agent = BedAllocationAgent()

    state = create_initial_state(
        "2031-10-03 12:00:00"
    )

    state["beds"] = [
        make_bed(
            "B001",
            status="Occupied"
        )
    ]

    assert agent.perceive(state) == []


def test_cleaning_bed_excluded():

    agent = BedAllocationAgent()

    state = create_initial_state(
        "2031-10-03 12:00:00"
    )

    state["beds"] = [
        make_bed(
            "B001",
            status="Cleaning"
        )
    ]

    assert agent.perceive(state) == []


def test_unknown_bed_excluded():

    agent = BedAllocationAgent()

    state = create_initial_state(
        "2031-10-03 12:00:00"
    )

    state["beds"] = [
        make_bed(
            "B001",
            status="Unknown"
        )
    ]

    assert agent.perceive(state) == []


def test_missing_bed_requirements():

    agent = BedAllocationAgent()

    patients = [
        make_patient(
            "P001",
            bed_type=None
        )
    ]

    allocated, waiting = agent.allocate(
        patients,
        [make_bed("B001")]
    )

    assert len(allocated) == 0

    assert len(waiting) == 1


def test_human_approval_required():

    agent = BedAllocationAgent()

    allocated, waiting = agent.allocate(
        [make_patient("P001")],
        [make_bed("B001")]
    )

    assert allocated[0][
        "human_approval_required"
    ] is True

    assert allocated[0][
        "allocation_status"
    ] == "Pending Approval"


def test_empty_queue():

    agent = BedAllocationAgent()

    allocated, waiting = agent.allocate(
        [],
        [make_bed("B001")]
    )

    assert allocated == []

    assert waiting == []


def test_duplicate_encounter():

    agent = BedAllocationAgent()

    patient = make_patient("P001")

    allocated, waiting = agent.allocate(
        [patient, patient.copy()],
        [
            make_bed("B001"),
            make_bed("B002")
        ]
    )

    assert len(allocated) == 1


def test_no_available_beds():

    agent = BedAllocationAgent()

    allocated, waiting = agent.allocate(
        [make_patient("P001")],
        []
    )

    assert allocated == []

    assert len(waiting) == 1
