"""Detección automática de puestos en el servicio: observar, terminar, descartar (sin el modelo real)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from admin.simular_puestos import CENTER, HEIGHT, VIEWS, WIDTH, camera, simulate
from app import cell_detection
from app.cell_detection import CellDetection

FRAME = np.zeros((HEIGHT, WIDTH, 3), np.uint8)


class FakeModel:
    """Devuelve, cuadro a cuadro, las siluetas de un parqueadero simulado."""

    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def __call__(self, frame, **options):
        items = self.frames[self.calls % len(self.frames)]
        self.calls += 1
        polygons = [item.polygon.astype(np.float32) for item in items]
        classes = np.array([item.class_id for item in items], float)
        masks = SimpleNamespace(xy=polygons) if polygons else None
        boxes = SimpleNamespace(cls=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: classes)))
        return [SimpleNamespace(masks=masks, boxes=boxes)]


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(cell_detection, "MIN_EVERY_SECONDS", 0.001)


def detection(frames, reference=None):
    model = FakeModel(frames)
    return CellDetection(
        frame_source=lambda: FRAME,
        reference_source=lambda: reference,
        model_path=cell_detection.Path("no-existe.pt"),
        dwell_minutes=0.0,
        model_factory=lambda: model,
    ), model


def wait_for(job, states, seconds=20.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if job.status()["estado"] in states:
            return job.status()
        time.sleep(0.02)
    raise AssertionError(f"La detección quedó en {job.status()['estado']}")


def scene_frames():
    position, focal = VIEWS["cenital"]
    return simulate(camera(position, CENTER, focal), occupancy=0.8, frames=12, seed=1).frames


def test_observing_then_finishing_proposes_the_cells():
    job, model = detection(scene_frames())
    job.start(minutes=10, every_seconds=0.01)
    assert job.status()["estado"] == "observando"
    deadline = time.time() + 10
    while model.calls < 12 and time.time() < deadline:
        time.sleep(0.02)
    job.finish()
    status = wait_for(job, {"lista", "error"})
    assert status["estado"] == "lista", status["error"]
    proposal = status["propuesta"]
    assert proposal["observadas"] >= 20
    assert proposal["ancho"] == WIDTH and proposal["alto"] == HEIGHT
    for cell in proposal["celdas"]:
        assert len(cell["puntos"]) == 4
        assert all(0.0 <= value <= 1.0 for point in cell["puntos"] for value in point)
    assert job.reference_jpeg() is not None


def test_the_observation_ends_by_itself_when_the_time_is_up():
    job, _ = detection(scene_frames())
    job.start(minutes=0.5 / 60, every_seconds=0.01)
    assert wait_for(job, {"lista", "error"})["estado"] == "lista"


def test_only_one_observation_at_a_time():
    job, _ = detection(scene_frames())
    job.start(minutes=10, every_seconds=0.01)
    with pytest.raises(RuntimeError):
        job.start(minutes=10, every_seconds=0.01)
    job.discard()
    assert job.status()["estado"] == "inactiva"
    assert job.status()["propuesta"] is None


def test_the_proposal_uses_the_reference_of_the_current_cells():
    reference = np.full((HEIGHT, WIDTH, 3), 7, np.uint8)
    job, _ = detection(scene_frames(), reference=reference)
    job.start(minutes=10, every_seconds=0.01)
    wait_for(job, {"observando"})
    deadline = time.time() + 5
    while job.status()["cuadros"] == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert np.array_equal(job.reference, reference)
    job.discard()


def test_no_frames_from_the_camera_is_an_error_not_a_crash():
    model = FakeModel([[]])
    job = CellDetection(lambda: None, lambda: None, cell_detection.Path("x.pt"), model_factory=lambda: model)
    job.start(minutes=0.2 / 60, every_seconds=0.01)
    status = wait_for(job, {"lista", "error"})
    assert status["estado"] == "error"
    assert "cuadro" in status["error"]


def test_invalid_durations_are_rejected():
    job, _ = detection(scene_frames())
    with pytest.raises(ValueError):
        job.start(minutes=0, every_seconds=5)
    with pytest.raises(ValueError):
        job.start(minutes=13 * 60, every_seconds=5)
