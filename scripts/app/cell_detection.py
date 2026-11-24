"""Detección automática de puestos desde el servicio: observa la cámara un rato y propone celdas.

El editor de celdas de la plataforma la inicia ("Detectar puestos"), puede cerrarse mientras
tanto y, al volver, carga la propuesta para revisarla antes de guardar. Corre en un hilo propio
y analiza un cuadro cada pocos segundos, así que casi no le quita tiempo a la detección de
ocupación. Las celdas salen relativas (0..1) al cuadro de referencia de la observación, que se
entrega aparte para que el editor trace sobre la misma imagen.

Nada se guarda en disco: si el servicio se reinicia, la observación se pierde.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment import CellAligner
from .auto_cells import Observation, propose, stationary_vehicles

log = logging.getLogger(__name__)

VEHICLE_CLASSES = [2, 3, 5, 7]  # carro, moto, bus, camión (COCO)
MAX_MINUTES = 12 * 60
MIN_EVERY_SECONDS = 1.0


class CellDetection:
    def __init__(
        self,
        frame_source: Callable[[], np.ndarray | None],
        reference_source: Callable[[], np.ndarray | None],
        model_path: Path,
        image_size: int = 1600,
        confidence: float = 0.15,
        dwell_minutes: float = 3.0,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.reference_source = reference_source
        self.model_path = model_path
        self.image_size = image_size
        self.confidence = confidence
        self.dwell_minutes = dwell_minutes
        self._model_factory = model_factory
        self._model: Any = None
        self._lock = threading.Lock()
        # Cada observación tiene su número y su señal de parada: si se descarta y se empieza otra,
        # el hilo anterior no puede escribir en la nueva.
        self._generation = 0
        self._stop = threading.Event()
        self._reset()

    def _reset(self) -> None:
        self.state = "inactiva"
        self.error: str | None = None
        self.started_at = 0.0
        self.finished_at = 0.0
        self.minutes = 0.0
        self.every_seconds = 0.0
        self.frames: list[list[Observation]] = []
        self.reference: np.ndarray | None = None
        self.proposal: dict[str, Any] | None = None

    # --- control --------------------------------------------------------------------------------

    def start(self, minutes: float, every_seconds: float) -> None:
        if not 0 < minutes <= MAX_MINUTES:
            raise ValueError(f"La observación debe durar entre unos segundos y {MAX_MINUTES // 60} horas")
        if every_seconds < MIN_EVERY_SECONDS:
            raise ValueError("Se analiza como mucho un cuadro por segundo")
        with self._lock:
            if self.state in ("observando", "calculando"):
                raise RuntimeError("Ya hay una detección en curso")
            self._reset()
            self._generation += 1
            self._stop = threading.Event()
            self.state = "observando"
            self.started_at = time.time()
            self.minutes = minutes
            self.every_seconds = every_seconds
            job = (self._generation, self._stop)
        threading.Thread(target=self._run, args=job, name="deteccion-puestos", daemon=True).start()

    def finish(self) -> None:
        """Termina de observar ya y calcula la propuesta con lo visto."""
        with self._lock:
            if self.state != "observando":
                raise RuntimeError("No hay una observación en curso")
            self._stop.set()

    def discard(self) -> None:
        """Cancela la observación (si la hay) y olvida la propuesta. No espera al hilo: el que
        estaba corriendo se da cuenta en su próximo cuadro y termina sin escribir nada."""
        with self._lock:
            self._stop.set()
            self._generation += 1
            self._reset()

    # --- lectura --------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0.0
            return {
                "estado": self.state,
                "error": self.error,
                "minutos": self.minutes,
                "cada_segundos": self.every_seconds,
                "inicio": self.started_at or None,
                "transcurrido_s": round(elapsed),
                "restante_s": max(0, round(self.minutes * 60 - elapsed)) if self.state == "observando" else 0,
                "cuadros": len(self.frames),
                "propuesta": self.proposal,
            }

    def reference_frame(self) -> np.ndarray | None:
        """Copia del cuadro al que están referidas las celdas propuestas; None si todavía no hay."""
        with self._lock:
            return None if self.reference is None else self.reference.copy()

    def reference_jpeg(self) -> bytes | None:
        with self._lock:
            reference = self.reference
        if reference is None:
            return None
        ok, encoded = cv2.imencode(".jpg", reference, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return encoded.tobytes() if ok else None

    # --- hilo -----------------------------------------------------------------------------------

    def _model_instance(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                from ultralytics import YOLO

                path = self.model_path if self.model_path.exists() else Path(self.model_path.name)
                self._model = YOLO(str(path))
        return self._model

    def _current(self, generation: int) -> bool:
        return generation == self._generation

    def _run(self, generation: int, stop: threading.Event) -> None:
        try:
            self._observe(generation, stop)
            with self._lock:
                if not self._current(generation):
                    return  # se descartó
                self.state = "calculando"
            proposal = self._propose()
            with self._lock:
                if self._current(generation):
                    self.proposal = proposal
                    self.state = "lista"
                    self.finished_at = time.time()
        except Exception as error:  # noqa: BLE001 - se informa al editor en vez de tumbar el hilo
            log.exception("Falló la detección automática de puestos")
            with self._lock:
                if self._current(generation):
                    self.state = "error"
                    self.error = str(error)
                    self.finished_at = time.time()

    def _observe(self, generation: int, stop: threading.Event) -> None:
        deadline = self.started_at + self.minutes * 60
        aligner: CellAligner | None = None
        while not stop.is_set() and time.time() < deadline:
            began = time.monotonic()
            frame = self.frame_source()
            if frame is not None:
                if aligner is None:
                    # Las celdas actuales se trazaron sobre su referencia: si existe, la propuesta
                    # queda en esas mismas coordenadas; si no, la referencia es el primer cuadro.
                    reference = self.reference_source()
                    aligner = CellAligner()
                    if reference is not None and reference.shape == frame.shape:
                        aligner.set_reference(reference)
                        if aligner.matches(frame) is False:
                            # La referencia es de otra escena (se cambió el video o se movió la
                            # cámara): la propuesta se calcularía sobre una imagen que ya no es.
                            log.info("La referencia de las celdas no coincide con la cámara: se usa el cuadro actual")
                            reference = None
                    if reference is None or reference.shape != frame.shape:
                        reference = frame.copy()
                        aligner.set_reference(reference)
                    with self._lock:
                        if self._current(generation):
                            self.reference = reference
                items = self._segment(frame, aligner.transform(frame))
                with self._lock:
                    if not self._current(generation):
                        return
                    self.frames.append(items)
            stop.wait(max(0.0, self.every_seconds - (time.monotonic() - began)))

    def _segment(self, frame: np.ndarray, to_reference: np.ndarray | None) -> list[Observation]:
        result = self._model_instance()(
            frame, conf=self.confidence, imgsz=self.image_size, verbose=False, classes=VEHICLE_CLASSES, retina_masks=True
        )[0]
        items = []
        if result.masks is None:
            return items
        for polygon, class_id in zip(result.masks.xy, result.boxes.cls.cpu().numpy()):
            if len(polygon) < 3:
                continue
            points = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
            if to_reference is not None:
                points = cv2.perspectiveTransform(points, to_reference)
            items.append(Observation(int(class_id), points.reshape(-1, 2).astype(float)))
        return items

    def _propose(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self.frames)
            reference = self.reference
        if reference is None or not frames:
            raise RuntimeError("No llegó ningún cuadro de la cámara durante la observación")
        height, width = reference.shape[:2]
        # Un puesto cuenta si un vehículo estuvo quieto en él unos minutos, aunque el resto del
        # tiempo esté libre; en una observación corta, si estuvo la mitad del tiempo.
        dwell = math.ceil(self.dwell_minutes * 60 / self.every_seconds)
        sightings = max(3, min(dwell, math.ceil(0.5 * len(frames))))
        vehicles = stationary_vehicles(frames, (height, width), min_presence=0.0, min_sightings=sightings)
        result = propose(vehicles, (height, width))
        cells = [
            {
                "puntos": np.round(np.clip(cell.corners / [width, height], 0.0, 1.0), 5).tolist(),
                "tipo": cell.kind,
                "fila": cell.row,
                "observada": cell.observed,
            }
            for cell in result.cells
        ]
        observed = sum(cell["observada"] for cell in cells)
        return {
            "celdas": cells,
            "observadas": observed,
            "deducidas": len(cells) - observed,
            "vehiculos": len(vehicles),
            "filas": len(result.rows),
            "ancho": width,
            "alto": height,
        }
