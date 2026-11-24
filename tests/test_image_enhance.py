import cv2
import numpy as np

from app.image_enhance import enhance_for_tracing, enhance_jpeg


def hazy_floor() -> np.ndarray:
    """Piso gris lavado con rayas apenas más claras, como el video de una cámara de vigilancia."""
    image = np.full((360, 640, 3), 150, np.uint8)
    for x in range(40, 640, 80):
        cv2.line(image, (x, 0), (x, 359), (165, 165, 165), 3)
    noise = np.random.default_rng(1).normal(0, 3, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def line_contrast(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    return gray[:, 39:42].mean() - gray[:, 70:90].mean()


def test_the_floor_lines_stand_out_more_and_the_size_stays():
    original = hazy_floor()
    enhanced = enhance_for_tracing(original)
    assert enhanced.shape == original.shape and enhanced.dtype == np.uint8
    assert line_contrast(enhanced) > 1.5 * line_contrast(original)


def test_a_jpeg_that_cannot_be_read_comes_back_as_is():
    assert enhance_jpeg(b"no es una imagen") == b"no es una imagen"
    ok, data = cv2.imencode(".jpg", hazy_floor())
    assert cv2.imdecode(np.frombuffer(enhance_jpeg(data.tobytes()), np.uint8), cv2.IMREAD_COLOR).shape == (360, 640, 3)
