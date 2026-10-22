"""Compensa el movimiento de la cámara respecto del cuadro sobre el que se trazaron las celdas.

Si la cámara se mueve (viento, vibraciones o un dron como el del video de prueba), las
celdas quedan fijas y el carro de un puesto termina "tocando" el vecino: se cuenta dos
veces o se pierde. Aquí se compara cada cuadro con el de referencia y se calcula cómo se
movió el suelo. El suelo es plano, así que ese movimiento es una homografía: cubre el
desplazamiento, el giro y el cambio de perspectiva. Se mide con puntos característicos
(ORB) de la parte baja de la imagen, que es suelo, emparejados con RANSAC para ignorar
los carros que entran o salen.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Se compara a este ancho: alcanza para precisión de un píxel y es rápido.
_WORK_WIDTH = 640
# Fracción superior de la imagen que se ignora: árboles, cielo y edificios no están en el suelo.
_IGNORE_TOP = 0.35
_FEATURES = 3000
# Mínimo de puntos que encajan con el movimiento para fiarse de él.
_MIN_INLIERS = 40
# Si el centro de la imagen se mueve más que esta fracción del ancho no es un temblor:
# es otra escena o la cámara se movió de lugar.
_MAX_SHIFT_RATIO = 0.1
# Cuadros que se miran para estimar la referencia cuando no se guardó la del trazado.
_ESTIMATION_FRAMES = 40


class _Features:
    def __init__(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height, width = gray.shape[:2]
        self.scale = width / _WORK_WIDTH
        small = cv2.resize(gray, (_WORK_WIDTH, max(1, round(height / self.scale))), interpolation=cv2.INTER_AREA)
        mask = np.zeros(small.shape, np.uint8)
        mask[int(small.shape[0] * _IGNORE_TOP):, :] = 255
        self.size = small.shape
        self.points, self.descriptors = cv2.ORB_create(_FEATURES).detectAndCompute(small, mask)


class CellAligner:
    def __init__(self, reference_path: Path | None = None, estimation_frames: int = _ESTIMATION_FRAMES) -> None:
        self.reference_path = reference_path
        self.estimation_frames = estimation_frames
        self._lock = threading.Lock()
        self._reference: _Features | None = None
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # "guardada": la envió el editor al guardar las celdas; "estimada": la eligió el servicio.
        self.source = "ninguna"
        self._candidates = 0
        self._best: tuple[int, np.ndarray] | None = None
        # Último movimiento medido: cuadro actual -> referencia (None si no se compensa).
        self.last_transform: np.ndarray | None = None
        self.last_shift = (0.0, 0.0)
        if reference_path is not None and reference_path.exists():
            image = cv2.imread(str(reference_path))
            if image is not None:
                self._set(image, "guardada")
                log.info("Referencia de las celdas cargada de %s", reference_path)

    def _set(self, frame: np.ndarray, source: str) -> None:
        features = _Features(frame)
        with self._lock:
            self._reference = features
            self.source = source
            self._candidates = 0
            self._best = None

    def set_reference(self, frame: np.ndarray) -> None:
        """El cuadro sobre el que se trazaron las celdas, enviado por el editor al guardarlas."""
        self._set(frame, "guardada")
        if self.reference_path is not None:
            cv2.imwrite(str(self.reference_path), frame)

    def cells_changed(self) -> None:
        """Celdas nuevas: una referencia estimada ya no sirve (una guardada llega junto con ellas)."""
        with self._lock:
            if self.source == "estimada":
                self._reference = None
                self.source = "ninguna"
            self._candidates = 0
            self._best = None

    def consider(self, frame: np.ndarray, fit: int) -> None:
        """Sin referencia, se toma el cuadro donde más carros caen dentro de las celdas.

        Las celdas se trazan sobre los carros estacionados, así que el cuadro en que mejor
        coinciden es el más parecido al del trazado.
        """
        with self._lock:
            if self._reference is not None:
                return
            if self._best is None or fit > self._best[0]:
                self._best = (fit, frame.copy())
            self._candidates += 1
            done = self._candidates >= self.estimation_frames
            best = self._best
        if done and best is not None:
            self._set(best[1], "estimada")
            log.info("Referencia de las celdas estimada: %d celdas coinciden con vehículos", best[0])

    def _reset(self) -> None:
        self.last_transform = None
        self.last_shift = (0.0, 0.0)

    def transform(self, frame: np.ndarray) -> np.ndarray | None:
        """Homografía que lleva un punto del cuadro actual a donde estaba en la referencia; None si no se sabe."""
        with self._lock:
            reference = self._reference
        if reference is None:
            self._reset()
            return None
        current = _Features(frame)
        if (current.descriptors is None or reference.descriptors is None or current.size != reference.size):
            self._reset()
            return None
        matches = self._matcher.match(current.descriptors, reference.descriptors)
        if len(matches) < _MIN_INLIERS:
            self._reset()
            return None
        source = np.float32([current.points[m.queryIdx].pt for m in matches])
        target = np.float32([reference.points[m.trainIdx].pt for m in matches])
        small, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if small is None or inliers is None or int(inliers.sum()) < _MIN_INLIERS:
            self._reset()
            return None
        # Se calculó a tamaño reducido: se pasa a píxeles del cuadro.
        scale = np.diag([current.scale, current.scale, 1.0])
        full = scale @ small @ np.linalg.inv(scale)
        height, width = frame.shape[:2]
        center = cv2.perspectiveTransform(np.float32([[[width / 2, height / 2]]]), full)[0, 0]
        shift = (float(width / 2 - center[0]), float(height / 2 - center[1]))
        if not np.isfinite(full).all() or np.hypot(*shift) > _MAX_SHIFT_RATIO * width:
            self._reset()
            return None
        self.last_transform = full
        self.last_shift = shift
        return full
