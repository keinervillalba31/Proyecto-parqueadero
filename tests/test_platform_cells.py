import threading
from types import SimpleNamespace

import numpy as np
import pytest

from app.backend_client import BackendClient, BackendError
from app.state import RuntimeState
from app.video_service import VideoService
from test_backend_client import FakeResponse, FakeSession, make_settings

CAR = 2


def vehicle_at(x1: int, y1: int, x2: int, y2: int):
    return (CAR, 0.9, (x1, y1, x2, y2))


def frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# Mitad izquierda de la imagen, en coordenadas relativas.
LEFT_HALF = {"points": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]], "puesto_id": 7, "puesto": "A-01"}


def test_relative_cells_are_scaled_to_the_real_frame_size(make_detector):
    detector = make_detector()
    detector.set_relative_cells({"A-01": LEFT_HALF})

    # Auto con la base en x=50 (mitad izquierda de un cuadro de 200 de ancho).
    detector.vehicles._detect_with_tiles = lambda f: [vehicle_at(30, 50, 70, 90)]
    state, _ = detector.analyze(frame(200, 100))
    assert set(state["celdas"]) == {"A-01"}
    assert state["celdas"]["A-01"]["ocupado"] is True
    assert state["celdas"]["A-01"]["puesto"] == "A-01"
    assert state["celdas"]["A-01"]["puesto_id"] == 7

    # El mismo auto a x=150 cae en la mitad derecha: ya no ocupa la celda.
    detector.vehicles._detect_with_tiles = lambda f: [vehicle_at(130, 50, 170, 90)]
    state, _ = detector.analyze(frame(200, 100))
    assert state["celdas"]["A-01"]["ocupado"] is False


def test_cells_follow_a_change_of_video_resolution(make_detector):
    detector = make_detector()
    detector.set_relative_cells({"A-01": LEFT_HALF})
    detector.vehicles._detect_with_tiles = lambda f: [vehicle_at(130, 50, 170, 90)]
    detector.analyze(frame(200, 100))

    # Con el doble de ancho, x=150 queda dentro de la mitad izquierda (0 a 200).
    state, _ = detector.analyze(frame(400, 100))
    assert state["celdas"]["A-01"]["ocupado"] is True


def test_reloading_cells_keeps_the_state_of_cells_that_remain(make_detector):
    detector = make_detector(occupancy_frames=3)
    detector.set_relative_cells({"A-01": LEFT_HALF})
    detector.vehicles._detect_with_tiles = lambda f: [vehicle_at(30, 50, 70, 90)]
    for _ in range(3):
        detector.analyze(frame(200, 100))

    # El administrador agrega otra celda: A-01 no debe "parpadear" a libre.
    detector.set_relative_cells({
        "A-01": LEFT_HALF,
        "A-02": {"points": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]], "puesto_id": 8, "puesto": "A-02"},
    })
    state, _ = detector.analyze(frame(200, 100))
    assert state["celdas"]["A-01"]["ocupado"] is True
    assert state["celdas"]["A-02"]["ocupado"] is False
    assert state["total_celdas"] == 2


def make_service(make_detector, backend) -> VideoService:
    service = VideoService.__new__(VideoService)
    service.settings = SimpleNamespace(backend_camera_id=1)
    service.backend = backend
    service.parking = make_detector()
    service.state = RuntimeState(total_spaces=2)
    service.cells_version = None
    return service


class FakeBackend:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error

    def get_camera_cells(self, camera_id):
        if self.error:
            raise self.error
        return self.data


PLATFORM_DATA = {
    "version": "100-1",
    "cells": [{"label": "A-01", "parkingSpaceId": 7, "parkingSpaceNumber": "A-01", "points": LEFT_HALF["points"]}],
}


def test_refresh_applies_platform_cells_once_per_version(make_detector):
    service = make_service(make_detector, FakeBackend(PLATFORM_DATA))

    assert service._refresh_cells() is True
    assert service.cells_version == "100-1"
    assert service.state.snapshot()["fuente"]["celdas_origen"] == "plataforma"
    # Misma versión: no se vuelve a aplicar.
    assert service._refresh_cells() is False


def test_refresh_keeps_local_cells_when_platform_has_none(make_detector):
    service = make_service(make_detector, FakeBackend({"version": "0-0", "cells": []}))

    assert service._refresh_cells() is False
    assert service.state.snapshot()["fuente"]["celdas_origen"] == "local"
    assert set(service.parking.cells) == {"Celda_1", "Celda_2"}


def test_refresh_survives_a_backend_error(make_detector):
    service = make_service(make_detector, FakeBackend(error=BackendError("caído")))

    assert service._refresh_cells() is False
    assert service.cells_version is None


@pytest.fixture
def client():
    backend = BackendClient(make_settings())
    session = FakeSession()
    backend.session = session
    return backend, session


def test_get_camera_cells_returns_the_data(client):
    backend, session = client
    session.script["/api/monitoring/cameras/3/cells"] = FakeResponse(200, {"success": True, "data": PLATFORM_DATA})

    assert backend.get_camera_cells(3) == PLATFORM_DATA


def test_get_camera_cells_raises_when_the_camera_does_not_exist(client):
    backend, session = client
    session.script["/api/monitoring/cameras/9/cells"] = FakeResponse(404, {}, text="no existe")

    with pytest.raises(BackendError):
        backend.get_camera_cells(9)
