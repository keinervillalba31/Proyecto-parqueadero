import threading
import time
from types import SimpleNamespace

from app.state import RuntimeState
from app.video_service import VideoService


def cells(**occupied: bool) -> dict:
    return {
        "celdas": {
            name: {"ocupado": value, "estado": "ocupado" if value else "libre"}
            for name, value in occupied.items()
        }
    }


def make_service(detector) -> VideoService:
    service = VideoService.__new__(VideoService)
    service.settings = SimpleNamespace(reservation_timeout_seconds=600)
    service.parking = detector
    service.reservations = {}
    service.reservations_lock = threading.Lock()
    return service


def test_reservation_lifecycle(make_detector):
    service = make_service(make_detector())

    assert service._assign_space("ABC123", cells(Celda_1=False, Celda_2=False)) == "Celda_1"
    # La misma placa conserva su celda y otra placa recibe la siguiente.
    assert service._assign_space("ABC123", cells(Celda_1=False, Celda_2=False)) == "Celda_1"
    assert service._assign_space("XYZ987", cells(Celda_1=False, Celda_2=False)) == "Celda_2"

    state = cells(Celda_1=True, Celda_2=False)
    service._update_reservations(state)
    assert service.reservations["ABC123"]["estacionado"] is True
    assert set(state["reservas"]) == {"ABC123", "XYZ987"}

    # El vehículo sale: la reserva se libera.
    service._update_reservations(cells(Celda_1=False, Celda_2=False))
    assert "ABC123" not in service.reservations
    assert service.reserved_cells() == {"Celda_2": "XYZ987"}


def test_reservation_expires_if_vehicle_never_parks(make_detector):
    service = make_service(make_detector())
    service._assign_space("ABC123", cells(Celda_1=False, Celda_2=False))
    service.reservations["ABC123"]["desde"] = time.time() - 601

    service._update_reservations(cells(Celda_1=False, Celda_2=False))
    assert service.reservations == {}


def test_parking_update_does_not_overwrite_plate():
    state = RuntimeState(total_spaces=2)
    state.update_plate({"valor": "ABC123", "estado": "autorizado"})
    state.update({"ocupadas": 1, "celdas": {}})
    assert state.snapshot()["placa"]["valor"] == "ABC123"
