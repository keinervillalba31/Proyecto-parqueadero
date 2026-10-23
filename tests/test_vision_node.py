import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.vision_node import CameraConfig, VisionNode


class FakeWorker:
    """Lo que el nodo usa de una cámara, sin video ni YOLO."""

    def __init__(self, settings, *, shared_model=None, backend=None, role="MIXED", manage_services=True):
        self.settings = settings
        self.shared_model = shared_model
        self.backend = backend
        self.role = role
        self.manage_services = manage_services
        self.started = False
        self.stopped = False
        self.services = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def _apply_services(self, spaces=None, plates=None):
        self.services.append((spaces, plates))

    def summary(self):
        return {"id": self.settings.backend_camera_id, "tipo": self.role, "en_linea": True, "error": None,
                "fps_video": 10.0, "fps_analisis": 0.3}


class FakeModel:
    loads = 0

    def __init__(self, path):
        FakeModel.loads += 1
        self.device = "cpu"
        self.calls = 0
        self.busy_seconds = 0.0


def write_cameras(path: Path, cameras, services=None):
    path.write_text(json.dumps({"cameras": cameras, "services": services or {}}), encoding="utf-8")


@pytest.fixture
def node_factory(tmp_path):
    FakeModel.loads = 0
    config = tmp_path / "camaras.json"

    def factory(cameras, services=None, **overrides):
        write_cameras(config, cameras, services)
        settings = replace(Settings(), node_cameras_file=config, node_data_dir=tmp_path / "datos",
                           backend_sync_enabled=False, yolo_tile_grid=2, stream_max_fps=0, **overrides)
        node = VisionNode(settings, model_factory=FakeModel, worker_factory=FakeWorker)
        return node, config

    return factory


OCCUPANCY = {"id": 4, "name": "Archivos", "type": "OCCUPANCY", "sourceUrl": "rtsp://archivos", "parkingId": 2}
ENTRY = {"id": 7, "name": "Entrada norte", "type": "ENTRY", "sourceUrl": "rtsp://entrada", "parkingId": 3}


def test_each_camera_gets_its_own_worker_and_all_share_one_model(node_factory, tmp_path):
    node, _ = node_factory([OCCUPANCY, ENTRY])
    node.start()

    assert set(node.workers) == {4, 7}
    assert FakeModel.loads == 1
    assert node.workers[4].shared_model is node.workers[7].shared_model
    assert all(worker.started and not worker.manage_services for worker in node.workers.values())

    occupancy = node.workers[4].settings
    assert occupancy.video_source == "rtsp://archivos"
    assert occupancy.backend_camera_id == 4 and occupancy.backend_parking_id == 2
    assert occupancy.cells_reference_path == tmp_path / "datos" / "4" / "referencia.jpg"
    # Sin GPU no se recorta: con recortes el modelo no alcanza para varias cámaras.
    assert occupancy.analysis_interval_seconds == 3.0 and occupancy.yolo_tile_grid == 1
    entry = node.workers[7].settings
    # En la entrada el vehículo está cerca: sin recortes y con análisis más seguido.
    assert entry.yolo_tile_grid == 1 and entry.analysis_interval_seconds == 0.5
    assert node.workers[7].role == "ENTRY"
    node.stop()


def test_cameras_are_added_removed_and_rebuilt_when_the_configuration_changes(node_factory):
    node, config = node_factory([OCCUPANCY, ENTRY])
    node.start()
    first_entry = node.workers[7]

    moved = {**OCCUPANCY, "sourceUrl": "rtsp://archivos-nueva"}
    write_cameras(config, [moved, {"id": 9, "type": "OCCUPANCY", "sourceUrl": "videos/otro.mp4"}])
    assert node.sync()

    assert set(node.workers) == {4, 9}
    assert first_entry.stopped
    assert node.workers[4].settings.video_source == "rtsp://archivos-nueva"
    assert node.workers[9].started
    node.stop()


def test_turning_a_camera_off_does_not_restart_it(node_factory):
    node, config = node_factory([OCCUPANCY])
    node.start()
    worker = node.workers[4]

    write_cameras(config, [{**OCCUPANCY, "detectionEnabled": False}], {"spacesEnabled": True, "platesEnabled": True})
    node.sync()

    assert node.workers[4] is worker and not worker.stopped
    assert worker.services[-1] == (False, False)
    node.stop()


def test_general_switches_apply_to_every_camera(node_factory):
    node, _ = node_factory([OCCUPANCY, ENTRY], {"spacesEnabled": False, "platesEnabled": True})
    node.start()

    assert node.workers[4].services[-1] == (False, True)
    assert node.workers[7].services[-1] == (False, True)
    node.stop()


def test_routes_without_a_number_use_the_occupancy_camera(node_factory):
    node, _ = node_factory([ENTRY, OCCUPANCY])
    node.start()

    assert node.default_worker() is node.workers[4]
    assert node.worker(99) is None
    node.stop()


def test_invalid_cameras_are_skipped():
    assert CameraConfig.parse({"id": 1}) is None
    assert CameraConfig.parse({"sourceUrl": "rtsp://x"}) is None
    assert CameraConfig.parse({"id": 1, "sourceUrl": "rtsp://x", "type": "raro"}).role == "OCCUPANCY"


def test_a_missing_configuration_keeps_what_is_running(node_factory):
    node, config = node_factory([OCCUPANCY])
    node.start()
    config.write_text("{no es json", encoding="utf-8")

    assert node.sync() is False
    assert set(node.workers) == {4}
    assert node.last_sync_error
    node.stop()


class FakeBackend:
    def __init__(self):
        self.heartbeats = []

    def get_node_config(self, code):
        return {"services": {"spacesEnabled": True, "platesEnabled": True}, "cameras": [OCCUPANCY]}

    def node_heartbeat(self, code, payload):
        self.heartbeats.append((code, payload))


def test_the_node_takes_its_cameras_from_the_platform_and_reports_them(tmp_path):
    FakeModel.loads = 0
    backend = FakeBackend()
    settings = replace(Settings(), node_code="nodo-norte", node_data_dir=tmp_path, node_public_url="http://10.0.0.5:8000",
                       node_heartbeat_seconds=0.1)
    node = VisionNode(settings, backend=backend, model_factory=FakeModel, worker_factory=FakeWorker)
    node.start()
    try:
        assert set(node.workers) == {4}
        import time
        for _ in range(40):
            if backend.heartbeats:
                break
            time.sleep(0.05)
        code, payload = backend.heartbeats[0]
        assert code == "nodo-norte"
        assert payload["baseUrl"] == "http://10.0.0.5:8000"
        assert payload["cameras"] == [{"id": 4, "online": True, "error": None, "fpsVideo": 10.0, "fpsAnalysis": 0.3}]
    finally:
        node.stop()


def test_without_node_settings_it_is_the_single_camera_service(tmp_path):
    settings = replace(Settings(), node_code="", node_cameras_file=None, backend_camera_id=4)
    node = VisionNode(settings, model_factory=FakeModel, worker_factory=FakeWorker)
    node.start()

    assert node.mode == "unica"
    assert set(node.workers) == {4}
    # La cámara única maneja sus interruptores como siempre y no comparte modelo.
    assert node.workers[4].manage_services and node.workers[4].shared_model is None
    node.stop()


def test_an_entry_camera_never_detects_spaces(tmp_path):
    from types import SimpleNamespace

    from app.state import RuntimeState
    from app.video_service import VideoService

    camera = VideoService.__new__(VideoService)
    camera.role = "ENTRY"
    camera.settings = SimpleNamespace(services_state_path=tmp_path / "servicios.json")
    camera.state = RuntimeState(total_spaces=0)
    camera.spaces_enabled = False
    camera.plates_enabled = False

    assert camera._apply_services(spaces=True, plates=True) == {"espacios": False, "placas": True}


def test_tiles_are_used_with_a_gpu_or_when_the_camera_asks(node_factory):
    node, _ = node_factory([OCCUPANCY, {**ENTRY, "id": 8, "type": "OCCUPANCY", "tileGrid": 2}])
    node.start()
    assert node.workers[4].settings.yolo_tile_grid == 1
    assert node.workers[8].settings.yolo_tile_grid == 2
    node.stop()

    gpu, _ = node_factory([OCCUPANCY])
    gpu._model_factory = lambda path: type("Gpu", (), {"device": "cuda", "calls": 0, "busy_seconds": 0.0})()
    gpu.start()
    assert gpu.workers[4].settings.yolo_tile_grid == 2
    gpu.stop()
