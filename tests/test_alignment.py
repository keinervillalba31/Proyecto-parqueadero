import cv2
import numpy as np
import pytest

from app.alignment import CellAligner


def scene(seed: int = 3, size=(720, 1280)) -> np.ndarray:
    """Imagen con textura en todo el cuadro, como el piso de un parqueadero."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (size[0] // 6, size[1] // 6), dtype=np.uint8)
    gray = cv2.resize(noise, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def warp(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)


def moved_by(dx: float, dy: float, tilt: float = 0.0) -> np.ndarray:
    """Movimiento de la cámara: desplazamiento y, con `tilt`, un cambio de perspectiva como el de un dron."""
    return np.array([[1.0, 0.01 * tilt, dx], [0.0, 1.0 + 0.02 * tilt, dy], [0.0, 2e-5 * tilt, 1.0]])


def mapped(aligner: CellAligner, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = aligner.transform(frame)
    assert transform is not None
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), transform).reshape(-1, 2)


GROUND_POINTS = np.array([[200, 400], [640, 500], [1100, 650], [300, 700]], float)


@pytest.mark.parametrize("dx, dy, tilt", [(12, -7, 0), (-20, 15, 0), (8, 5, 1), (-15, 10, 1)])
def test_points_of_the_ground_go_back_to_where_they_were_traced(dx, dy, tilt):
    reference = scene()
    aligner = CellAligner()
    aligner.set_reference(reference)
    camera_move = moved_by(dx, dy, tilt)
    now = cv2.perspectiveTransform(np.float32(GROUND_POINTS).reshape(-1, 1, 2), camera_move).reshape(-1, 2)
    assert mapped(aligner, warp(reference, camera_move), now) == pytest.approx(GROUND_POINTS, abs=1.5)


def test_without_reference_or_with_another_scene_it_does_not_move_anything():
    aligner = CellAligner()
    assert aligner.transform(scene()) is None
    aligner.set_reference(scene(seed=1))
    assert aligner.transform(scene(seed=2)) is None
    assert aligner.last_shift == (0.0, 0.0)


def test_the_reference_sent_by_the_editor_is_kept_on_disk(tmp_path):
    path = tmp_path / "referencia.png"
    CellAligner(path).set_reference(scene())
    reloaded = CellAligner(path)
    assert reloaded.source == "guardada"
    assert reloaded.transform(warp(scene(), moved_by(10, 5))) is not None
    assert reloaded.last_shift == pytest.approx((10, 5), abs=1.5)


def test_without_a_saved_reference_it_picks_the_frame_where_cells_fit_best():
    aligner = CellAligner(estimation_frames=3)
    base = scene()
    for fit, shift in [(10, 15), (20, 0), (12, -15)]:
        aligner.consider(warp(base, moved_by(shift, 0)), fit)
    assert aligner.source == "estimada"
    # La referencia elegida es la del mejor encaje (sin movimiento).
    assert aligner.transform(base) is not None
    assert aligner.last_shift == pytest.approx((0, 0), abs=1.0)


def test_new_cells_discard_an_estimated_reference_but_not_a_saved_one():
    aligner = CellAligner(estimation_frames=1)
    aligner.consider(scene(), 5)
    aligner.cells_changed()
    assert aligner.source == "ninguna"
    aligner.set_reference(scene())
    aligner.cells_changed()
    assert aligner.source == "guardada"


def test_the_editor_can_get_back_the_frame_the_cells_were_traced_on(tmp_path):
    aligner = CellAligner()
    assert aligner.reference_jpeg() is None

    reference = scene()
    aligner.set_reference(reference)
    image = cv2.imdecode(np.frombuffer(aligner.reference_jpeg(), np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == reference.shape
    assert np.abs(image.astype(int) - reference.astype(int)).mean() < 3

    path = tmp_path / "referencia.jpg"
    CellAligner(path).set_reference(reference)
    assert CellAligner(path).reference_jpeg() is not None


def test_a_discarded_estimated_reference_is_not_shown_anymore():
    aligner = CellAligner(estimation_frames=1)
    aligner.consider(scene(), fit=5)
    assert aligner.source == "estimada" and aligner.reference_jpeg() is not None
    aligner.cells_changed()
    assert aligner.reference_jpeg() is None


def test_health_reports_how_far_the_camera_moved_and_when_it_lost_the_reference():
    reference = scene()
    aligner = CellAligner()
    aligner.set_reference(reference)

    aligner.transform(warp(reference, moved_by(25.6, 0)))
    health = aligner.health()
    # 25,6 px de 1280 = 2 % del ancho.
    assert health["desplazamiento_pct"] == pytest.approx(2.0, abs=0.3)
    assert health["fallos_seguidos"] == 0

    other = scene(seed=99)
    aligner.transform(other)
    aligner.transform(other)
    assert aligner.health()["fallos_seguidos"] == 2

    aligner.transform(reference)
    assert aligner.health()["fallos_seguidos"] == 0


def test_without_reference_there_are_no_failures_to_report():
    aligner = CellAligner()
    aligner.transform(scene())
    assert aligner.health()["fallos_seguidos"] == 0
