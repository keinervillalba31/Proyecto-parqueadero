import threading
import time
from types import SimpleNamespace

import pytest

from app.state import RuntimeState
from app.video_service import VideoService


class FakeCapture:
    def __init__(self, name: str) -> None:
        self.name = name
        self.released = False

    def release(self) -> None:
        self.released = True


def make_service(monkeypatch, available: set[str], fallback: str = "respaldo.mp4", retry: float = 10):
    """Servicio sin YOLO; `available` son las fuentes que "se pueden abrir"."""
    service = VideoService.__new__(VideoService)
    service.settings = SimpleNamespace(
        video_source="rtsp://camara",
        video_fallback=fallback,
        video_camera_retry_seconds=retry,
    )
    service.state = RuntimeState(total_spaces=2)
    service.active_source = service.settings.video_source
    service.using_fallback = False
    service.last_primary_attempt = 0.0
    service.probe_lock = threading.Lock()
    service.probe_thread = None
    service.probed_capture = None

    def fake_open(source: str) -> FakeCapture:
        if source not in available:
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
        return FakeCapture(source)

    monkeypatch.setattr(VideoService, "_open_capture", staticmethod(fake_open))
    return service


def test_uses_primary_camera_when_available(monkeypatch):
    service = make_service(monkeypatch, available={"rtsp://camara", "respaldo.mp4"})

    capture = service._open_source()

    assert capture.name == "rtsp://camara"
    assert service.using_fallback is False
    assert service.state.snapshot()["fuente"]["modo"] == "principal"


def test_falls_back_to_looping_video_when_camera_is_down(monkeypatch):
    service = make_service(monkeypatch, available={"respaldo.mp4"})

    capture = service._open_source()

    assert capture.name == "respaldo.mp4"
    assert service.using_fallback is True
    fuente = service.state.snapshot()["fuente"]
    assert fuente["modo"] == "respaldo"
    assert fuente["origen"] == "respaldo.mp4"


def test_raises_when_camera_is_down_and_there_is_no_fallback(monkeypatch):
    service = make_service(monkeypatch, available=set(), fallback="")

    with pytest.raises(RuntimeError):
        service._open_source()


def test_does_not_probe_camera_before_retry_interval(monkeypatch):
    service = make_service(monkeypatch, available={"respaldo.mp4"}, retry=60)
    capture = service._open_source()

    assert service._maybe_switch_to_primary(capture) is capture
    assert service.probe_thread is None


def test_switches_to_camera_as_soon_as_it_responds(monkeypatch):
    available = {"respaldo.mp4"}
    service = make_service(monkeypatch, available=available, retry=0)
    fallback_capture = service._open_source()

    # La cámara sigue caída: se intenta en segundo plano y se queda en el respaldo.
    assert service._maybe_switch_to_primary(fallback_capture) is fallback_capture
    service.probe_thread.join(timeout=2)
    assert service._maybe_switch_to_primary(fallback_capture) is fallback_capture
    assert service.using_fallback is True

    # La cámara vuelve: en el siguiente reintento se cambia y se suelta el respaldo.
    available.add("rtsp://camara")
    service.last_primary_attempt = 0.0
    service._maybe_switch_to_primary(fallback_capture)
    service.probe_thread.join(timeout=2)
    switched = service._maybe_switch_to_primary(fallback_capture)

    assert switched.name == "rtsp://camara"
    assert fallback_capture.released is True
    assert service.using_fallback is False
    assert service.state.snapshot()["fuente"]["modo"] == "principal"


def test_source_info_survives_parking_updates():
    state = RuntimeState(total_spaces=2)
    state.update_source({"modo": "respaldo", "origen": "respaldo.mp4"})
    state.update({"ocupadas": 1, "celdas": {}})

    assert state.snapshot()["fuente"]["modo"] == "respaldo"
