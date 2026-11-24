"""Mejora un cuadro de la cámara para trazar celdas sobre él.

Las cámaras de vigilancia graban con poco contraste y bastante compresión: las rayas del piso
y los bordes de los carros se ven lavados. Para el ojo de quien traza se sube el contraste
local (CLAHE sobre la luminosidad, sin cambiar los colores), se suaviza el ruido de la
compresión conservando los bordes y se enfoca con una máscara de desenfoque.

Solo es para mostrar: el detector siempre analiza el cuadro original.
"""

from __future__ import annotations

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# Cuánto se enfoca: 0 nada; más de ~1 empieza a marcar halos y ruido.
_SHARPEN = 0.6


def enhance_for_tracing(frame: np.ndarray) -> np.ndarray:
    """Copia del cuadro (BGR) con más contraste local y más nítida."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lab = cv2.merge((_CLAHE.apply(lightness), a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = cv2.bilateralFilter(image, 5, 25, 5)
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.addWeighted(image, 1 + _SHARPEN, blurred, -_SHARPEN, 0)


def enhance_jpeg(data: bytes, quality: int = 92) -> bytes:
    """La misma mejora sobre una imagen JPEG; si no se puede leer, la devuelve igual."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return data
    ok, encoded = cv2.imencode(".jpg", enhance_for_tracing(image), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if ok else data
