from evaluation.fcfs_baseline import FCFSBaseline


def make_patient(
    patient_id,
    priority,
    arrival,
    bed_type="medical ward",
):

    return {
        "patient_id": patient_id,
        "encounter_id": f"ENC-{patient_id}",
        "priority": priority,
        "arrival_time": arrival,
        "waiting_time_minutes": 60,
        "requested_care": (
            "ICU"
            if bed_type == "ICU"
            else "inpatient"
        ),
        "requested_bed_type": bed_type,
    }


def make_bed(
    bed_id,
    bed_type="medical ward",
    status="Available",
):

    return {
        "bed_id": bed_id,
        "bed_type": bed_type,
        "status": status,
        "allocatable": status == "Available",
    }


def test_earliest_arrival_selected():

    patients = [
        make_patient(
            "P002", 1, "2031-11-07 10:00"
        ),
        make_patient(
            "P001", 4, "2031-11-07 08:00"
        ),
    ]

    beds = [make_bed("B001")]

    allocated, waiting = (
        FCFSBaseline().allocate(patients, beds)
    )

    assert len(allocated) == 1
    assert allocated[0]["patient_id"] == "P001"
    assert len(waiting) == 1


def test_incompatible_bed_rejected():

    patients = [
        make_patient(
            "P001", 1,
            "2031-11-07 08:00",
            "ICU",
        )
    ]

    beds = [make_bed("B001", "medical ward")]

    allocated, waiting = (
        FCFSBaseline().allocate(patients, beds)
    )

    assert len(allocated) == 0
    assert len(waiting) == 1


def test_occupied_bed_rejected():

    patients = [
        make_patient(
            "P001", 1, "2031-11-07 08:00"
        )
    ]

    beds = [
        make_bed(
            "B001", status="Occupied"
        )
    ]

    allocated, waiting = (
        FCFSBaseline().allocate(patients, beds)
    )

    assert len(allocated) == 0
    assert len(waiting) == 1


def test_no_duplicate_bed_allocation():

    patients = [
        make_patient(
            "P001", 2, "2031-11-07 08:00"
        ),
        make_patient(
            "P002", 1, "2031-11-07 09:00"
        ),
    ]

    beds = [make_bed("B001")]

    allocated, waiting = (
        FCFSBaseline().allocate(patients, beds)
    )

    assert len(allocated) == 1
    assert len(waiting) == 1


def test_empty_patient_queue():

    allocated, waiting = (
        FCFSBaseline().allocate(
            [],
            [make_bed("B001")],
        )
    )

    assert allocated == []
    assert waiting == []


def test_no_available_beds():

    patients = [
        make_patient(
            "P001", 1, "2031-11-07 08:00"
        )
    ]

    allocated, waiting = (
        FCFSBaseline().allocate(patients, [])
    )

    assert allocated == []
    assert len(waiting) == 1


def test_human_approval_required():

    patients = [
        make_patient(
            "P001", 1, "2031-11-07 08:00"
        )
    ]

    allocated, _ = FCFSBaseline().allocate(
        patients,
        [make_bed("B001")],
    )

    assert allocated[0][
        "human_approval_required"
    ] is True

    assert allocated[0][
        "allocation_status"
    ] == "Pending Approval"


def test_original_bed_inventory_unchanged():

    patients = [
        make_patient(
            "P001", 1, "2031-11-07 08:00"
        )
    ]

    beds = [make_bed("B001")]

    FCFSBaseline().allocate(patients, beds)

    assert beds[0]["status"] == "Available"
    assert beds[0]["allocatable"] is True