import json
from types import SimpleNamespace

from app.video_service import VideoService
from test_realtime import CAR_IN_CELL_1, blank_frame, make_service


def test_set_services_updates_state_and_survives_a_restart(make_detector, tmp_path):
    service = make_service(make_detector())
    service.settings.services_state_path = tmp_path / "servicios.json"

    assert service.services() == {"espacios": True, "placas": True}
    assert service.set_services(plates=False) == {"espacios": True, "placas": False}

    snapshot = service.state.snapshot()
    assert snapshot["servicios"] == {"espacios": True, "placas": False}
    assert snapshot["placa"]["estado"] == "detenido"
    assert json.loads(service.settings.services_state_path.read_text()) == {
        "espacios": True,
        "placas": False,
    }
    assert VideoService._load_services(service.settings.services_state_path) == (True, False)


def test_local_change_is_saved_in_the_platform_first(make_detector, tmp_path):
    service = make_service(make_detector())
    service.settings.services_state_path = tmp_path / "servicios.json"
    sent = []
    service.backend = SimpleNamespace(set_services=lambda spaces, plates: sent.append((spaces, plates)))

    service.set_services(spaces=False)

    assert sent == [(False, None)]
    assert service.services() == {"espacios": False, "placas": True}


def test_platform_switches_are_applied_without_echoing_them_back(make_detector, tmp_path):
    service = make_service(make_detector())
    service.settings.services_state_path = tmp_path / "servicios.json"
    service.backend = SimpleNamespace(set_services=lambda *args: 1 / 0)

    service._apply_services(True, False)

    assert service.services() == {"espacios": True, "placas": False}
    assert service.state.snapshot()["placa"]["estado"] == "detenido"


def test_services_default_to_on_without_a_saved_file(tmp_path):
    assert VideoService._load_services(tmp_path / "no_existe.json") == (True, True)


def test_plates_off_does_not_start_a_plate_read(make_detector):
    service = make_service(make_detector(), detections=[CAR_IN_CELL_1])
    service.vehicle_was_detected = False  # el vehículo recién llegó: normalmente se leería
    service._read_plate = lambda images: None
    service.plates_enabled = False

    service._process_detection(blank_frame())

    assert service.plate_thread is None


def test_plates_on_starts_a_plate_read_when_a_vehicle_arrives(make_detector):
    service = make_service(make_detector(), detections=[CAR_IN_CELL_1])
    service.vehicle_was_detected = False
    service._read_plate = lambda images: None

    service._process_detection(blank_frame())

    assert service.plate_thread is not None


def test_spaces_off_keeps_the_last_occupancy_and_stops_reporting(make_detector):
    service = make_service(make_detector(), detections=[CAR_IN_CELL_1])
    service.backend = SimpleNamespace(report_occupancy=lambda *args: 1 / 0)
    service.spaces_enabled = False

    service._process_detection(blank_frame())

    assert service.state.snapshot()["celdas"] == {}
    assert service._report_occupancy() is False


def test_state_update_keeps_the_services_flags(make_detector):
    service = make_service(make_detector())
    service.state.update_services({"espacios": False, "placas": True})

    service.state.update({"celdas": {}})

    assert service.state.snapshot()["servicios"] == {"espacios": False, "placas": True}
