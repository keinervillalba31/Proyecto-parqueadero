import time
from types import SimpleNamespace

from app.backend_client import BackendError, PlateEntryResult
from app.plate_reader import PlateReader
from test_realtime import make_service


def entry_result(outcome: str, plate: str = "ABC123") -> PlateEntryResult:
    return PlateEntryResult(
        outcome=outcome,
        plate=plate,
        assignment_id=None,
        parking_space_id=None,
        parking_id=None,
        space_number=None,
        parking_name=None,
        notified=outcome == "PROMPTED",
        raw={},
    )


def make_plate_service(make_detector, outcomes: dict[str, str]):
    """Servicio con dos lecturas para confirmar y un backend que responde según `outcomes`."""
    service = make_service(make_detector())
    service.settings.plate_cooldown_seconds = 30
    service.settings.plate_recheck_seconds = 4
    service.plates = PlateReader("", "", confirmation_reads=2, cooldown_seconds=30)
    service.registered_vehicles = {}
    service.first_read_checks = {}
    sent: list[str] = []

    def sync_plate_detection(plate: str) -> PlateEntryResult:
        sent.append(plate)
        return entry_result(outcomes.get(plate, "NOT_REGISTERED"), plate)

    service.backend = SimpleNamespace(sync_plate_detection=sync_plate_detection)
    service.plates.read_all = lambda image: list(image)  # la "imagen" ya trae sus placas
    return service, sent


def read(service, *plates: str) -> None:
    service._read_plate([list(plates)])
    for worker in list(service_threads(service)):
        worker.join(timeout=2)


def service_threads(service):
    import threading

    return [t for t in threading.enumerate() if t.name in ("backend-first-read", "backend-sync")]


def test_first_read_of_a_registered_plate_is_confirmed_without_waiting(make_detector):
    service, sent = make_plate_service(make_detector, {"ABC123": "PROMPTED"})

    read(service, "ABC123")

    plate = service.state.snapshot()["placa"]
    assert sent == ["ABC123"]
    assert plate["valor"] == "ABC123"
    assert plate["backend_estado"] == "prompted"
    assert service.plates.in_cooldown("ABC123")


def test_a_recognized_plate_is_not_sent_again_while_it_stays_in_front_of_the_camera(make_detector):
    service, sent = make_plate_service(make_detector, {"ABC123": "PROMPTED"})

    for _ in range(4):
        read(service, "ABC123")

    assert sent == ["ABC123"]


def test_plate_with_a_spot_is_asked_again_soon_so_a_release_is_noticed(make_detector):
    outcomes = {"ABC123": "ALREADY_ASSIGNED"}
    service, sent = make_plate_service(make_detector, outcomes)
    service.settings.plate_recheck_seconds = 0.05

    read(service, "ABC123")
    assert service.state.snapshot()["placa"]["backend_estado"] == "already_assigned"

    # El estudiante libera el puesto; la placa sigue frente a la cámara.
    outcomes["ABC123"] = "PROMPTED"
    time.sleep(0.1)
    read(service, "ABC123")

    assert sent == ["ABC123", "ABC123"]
    assert service.state.snapshot()["placa"]["backend_estado"] == "prompted"


def test_unknown_plate_still_needs_repeated_reads(make_detector):
    service, sent = make_plate_service(make_detector, {})

    read(service, "ZZZ999")
    assert service.state.snapshot()["placa"]["valor"] is None
    assert service.state.snapshot()["placa"]["estado"] == "confirmando"

    read(service, "ZZZ999")
    plate = service.state.snapshot()["placa"]
    assert plate["valor"] == "ZZZ999"
    assert plate["backend_estado"] == "not_registered"


def test_a_misread_does_not_block_the_real_plate(make_detector):
    # El OCR confunde un carácter la primera vez; la lectura buena llega después.
    service, sent = make_plate_service(make_detector, {"ABC123": "PROMPTED"})

    read(service, "ABC128")
    assert service.state.snapshot()["placa"]["valor"] is None

    read(service, "ABC123")
    assert service.state.snapshot()["placa"]["valor"] == "ABC123"


def test_backend_down_falls_back_to_repeated_reads(make_detector):
    service, _ = make_plate_service(make_detector, {})

    def failing(plate: str):
        raise BackendError("sin conexión")

    service.backend = SimpleNamespace(sync_plate_detection=failing)

    read(service, "ABC123")
    assert service.state.snapshot()["placa"]["valor"] is None
    read(service, "ABC123")
    plate = service.state.snapshot()["placa"]
    assert plate["valor"] == "ABC123"
    assert plate["backend_estado"] == "error"


def test_without_backend_nothing_is_consulted(make_detector):
    service, sent = make_plate_service(make_detector, {"ABC123": "PROMPTED"})
    service.backend = None

    assert service._check_first_read("ABC123") is None
    assert sent == []
