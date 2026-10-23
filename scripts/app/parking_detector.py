from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from .alignment import CellAligner


VEHICLE_CLASSES = {2: "Carro", 3: "Moto", 5: "Bus", 7: "Camion"}


@dataclass
class Vehicle:
    class_id: int
    confidence: float
    box: tuple[int, int, int, int]
    # True si el vehículo está dentro de una celda (estacionado).
    in_cell: bool = False

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)

    @property
    def footpoint(self) -> Point:
        x1, _, x2, y2 = self.box
        return Point((x1 + x2) / 2, y2)


# Una caja que queda casi entera dentro de otra más grande es el mismo vehículo.
_CONTAINED_RATIO = 0.7


def _drop_partial_boxes(
    detections: list[tuple[int, float, tuple[int, int, int, int]]],
) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    """Quita las cajas parciales de un vehículo que ya tiene su caja completa.

    Cuando el borde de un recorte corta un carro, YOLO devuelve una caja con solo una parte;
    se superpone poco con la caja completa, así que el NMS la deja pasar y su punto de apoyo
    cae en el puesto vecino: el carro se cuenta dos veces.
    """
    def area(box: tuple[int, int, int, int]) -> int:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def contained(small: tuple[int, int, int, int], large: tuple[int, int, int, int]) -> float:
        width = min(small[2], large[2]) - max(small[0], large[0])
        height = min(small[3], large[3]) - max(small[1], large[1])
        return max(0, width) * max(0, height) / max(area(small), 1)

    kept: list[tuple[int, float, tuple[int, int, int, int]]] = []
    for detection in sorted(detections, key=lambda item: area(item[2]), reverse=True):
        if all(contained(detection[2], other[2]) <= _CONTAINED_RATIO for other in kept):
            kept.append(detection)
    return kept


def best_device() -> str:
    """"cuda" si hay una GPU NVIDIA usable; si no, "cpu"."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SharedModel:
    """Un solo YOLO en memoria para todas las cámaras de un nodo.

    YOLO no se puede llamar desde dos hilos a la vez, así que cada inferencia toma el
    candado: las cámaras se turnan solas, sin cargar una copia del modelo por cámara.
    """

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model = YOLO(str(model_path))
        self.device = device or best_device()
        self.lock = threading.Lock()
        # Para el estado del nodo: cuántas inferencias y cuánto tiempo de modelo en total.
        self.calls = 0
        self.busy_seconds = 0.0

    def __call__(self, source: np.ndarray, **options: Any) -> Any:
        with self.lock:
            started = time.perf_counter()
            try:
                return self.model(source, device=self.device, **options)
            finally:
                self.calls += 1
                self.busy_seconds += time.perf_counter() - started


class VehicleDetector:
    """Detecta vehículos con YOLO, analizando también recortes del cuadro."""

    def __init__(
        self,
        model_path: Path,
        confidence: float,
        image_size: int,
        tile_grid: int,
        minimum_box_area: int,
        minimum_motorcycle_area: int,
        model: SharedModel | None = None,
    ) -> None:
        # Con un modelo compartido (nodo con varias cámaras) no se carga otra copia.
        self.model = model if model is not None else YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.tile_grid = max(tile_grid, 1)
        self.minimum_box_area = minimum_box_area
        self.minimum_motorcycle_area = minimum_motorcycle_area

    def detect(self, frame: np.ndarray) -> list[Vehicle]:
        vehicles = []
        for class_id, confidence, box in self._detect_with_tiles(frame):
            vehicle = Vehicle(class_id, confidence, box)
            minimum_area = (
                self.minimum_motorcycle_area if class_id == 3 else self.minimum_box_area
            )
            if vehicle.area >= minimum_area:
                vehicles.append(vehicle)
        return vehicles

    def _detect_with_tiles(
        self, frame: np.ndarray
    ) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        height, width = frame.shape[:2]
        candidates: list[tuple[int, float, tuple[int, int, int, int]]] = []

        sources = [(frame, 0, 0)]
        if self.tile_grid > 1:
            tile_width = min(width, int(width / self.tile_grid * 1.35))
            tile_height = min(height, int(height / self.tile_grid * 1.35))
            x_starts = self._tile_starts(width, tile_width)
            y_starts = self._tile_starts(height, tile_height)
            sources.extend(
                (frame[y : y + tile_height, x : x + tile_width], x, y)
                for y in y_starts
                for x in x_starts
            )

        for source, offset_x, offset_y in sources:
            result = self.model(
                source,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False,
            )[0]
            for detection in result.boxes:
                class_id = int(detection.cls[0])
                if class_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                candidates.append(
                    (
                        class_id,
                        float(detection.conf[0]),
                        (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    )
                )

        if not candidates:
            return []
        boxes = [[x1, y1, x2 - x1, y2 - y1] for _, _, (x1, y1, x2, y2) in candidates]
        kept = cv2.dnn.NMSBoxes(
            boxes,
            [confidence for _, confidence, _ in candidates],
            self.confidence,
            0.45,
        )
        return _drop_partial_boxes([candidates[index] for index in np.array(kept).flatten()])

    def _tile_starts(self, length: int, tile_length: int) -> list[int]:
        """Reparte `tile_grid` recortes solapados a lo largo de un eje."""
        last = max(length - tile_length, 0)
        return sorted(
            {round(last * index / (self.tile_grid - 1)) for index in range(self.tile_grid)}
        )


class ParkingDetector:
    def __init__(
        self,
        cells_path: Path,
        model_path: Path,
        confidence: float,
        image_size: int,
        tile_grid: int,
        minimum_box_area: int,
        minimum_motorcycle_area: int,
        occupy_seconds: float = 0.0,
        free_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        aligner: CellAligner | None = None,
        model: SharedModel | None = None,
    ) -> None:
        # Sin archivo de celdas locales (cámara de un nodo): empieza sin celdas y las trae la plataforma.
        parking_cells = {}
        if cells_path.exists():
            with cells_path.open("r", encoding="utf-8") as file:
                parking_cells = json.load(file)

        self.cells = {
            name: {
                "points": np.array(points, dtype=np.int32),
                "polygon": Polygon(points),
            }
            for name, points in parking_cells.items()
        }
        self.vehicles = VehicleDetector(
            model_path=model_path,
            confidence=confidence,
            image_size=image_size,
            tile_grid=tile_grid,
            minimum_box_area=minimum_box_area,
            minimum_motorcycle_area=minimum_motorcycle_area,
            model=model,
        )
        # Vehículos del último cuadro, para recortarlos y leer sus placas.
        self.last_vehicles: list[Vehicle] = []
        # Un cambio de estado solo se acepta si se sostiene un tiempo: ocupar pide
        # `occupy_seconds` viendo el vehículo y liberar `free_seconds` sin verlo.
        # Liberar exige más porque YOLO pierde a ratos los carros lejanos o
        # tapados, y un carro estacionado no se va en un segundo. Va por tiempo,
        # no por cuadros, para que no dependa de lo rápido que analice el equipo.
        self.occupy_seconds = max(occupy_seconds, 0.0)
        self.free_seconds = max(free_seconds, 0.0)
        self._clock = clock
        # Si se da, las celdas siguen a la imagen cuando la cámara se corre.
        self.aligner = aligner
        self.stable_occupied = {name: False for name in self.cells}
        # Desde cuándo una celda muestra lo contrario de su estado, sin interrupción.
        self.pending_since: dict[str, float | None] = {name: None for name in self.cells}
        # Celdas traídas de la plataforma, en coordenadas relativas (0 a 1); se
        # convierten a píxeles según el tamaño real del video. nombre -> datos.
        self.cell_meta: dict[str, dict[str, Any]] = {}
        self._relative_cells: dict[str, dict[str, Any]] | None = None
        self._scaled_size: tuple[int, int] | None = None
        self._incoming_cells: dict[str, dict[str, Any]] | None = None
        self._incoming_lock = threading.Lock()

    def set_relative_cells(self, cells: dict[str, dict[str, Any]]) -> None:
        """Reemplaza las celdas por las de la plataforma.

        Se puede llamar desde otro hilo: el cambio se aplica al inicio del
        siguiente análisis, para no tocar las celdas mientras se usan.
        cells: nombre -> {"points": [[x, y] x4, valores 0..1], "puesto_id": int|None, "puesto": str|None}
        """
        with self._incoming_lock:
            self._incoming_cells = cells

    def _apply_cell_changes(self, width: int, height: int) -> None:
        with self._incoming_lock:
            incoming, self._incoming_cells = self._incoming_cells, None
        if incoming is not None:
            self._relative_cells = incoming
            self._scaled_size = None
            if self.aligner is not None:
                self.aligner.cells_changed()
        if self._relative_cells is None or self._scaled_size == (width, height):
            return

        cells = {}
        for name, data in self._relative_cells.items():
            points = [[round(x * width), round(y * height)] for x, y in data["points"]]
            cells[name] = {"points": np.array(points, dtype=np.int32), "polygon": Polygon(points)}
        # Las celdas que siguen existiendo conservan su estado ya suavizado.
        self.stable_occupied = {name: self.stable_occupied.get(name, False) for name in cells}
        self.pending_since = {name: self.pending_since.get(name) for name in cells}
        self.cell_meta = {
            name: {"puesto_id": data.get("puesto_id"), "puesto": data.get("puesto")}
            for name, data in self._relative_cells.items()
        }
        self.cells = cells
        self._scaled_size = (width, height)

    def _smooth(self, raw_occupied: dict[str, bool]) -> dict[str, bool]:
        now = self._clock()
        for name, is_occupied in raw_occupied.items():
            if is_occupied == self.stable_occupied[name]:
                # Basta un cuadro que confirme el estado para descartar el cambio pendiente.
                self.pending_since[name] = None
                continue
            started = self.pending_since[name]
            if started is None:
                started = self.pending_since[name] = now
            if now - started >= (self.occupy_seconds if is_occupied else self.free_seconds):
                self.stable_occupied[name] = is_occupied
                self.pending_since[name] = None
        return dict(self.stable_occupied)

    def detect(
        self, frame: np.ndarray, reserved: dict[str, str] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Analiza el cuadro y dibuja el resultado sobre él."""
        reserved = reserved or {}
        state, vehicles = self.analyze(frame, reserved)
        self.draw(frame, vehicles, state, reserved)
        return frame, state

    def draw(
        self,
        frame: np.ndarray,
        vehicles: list[Vehicle],
        state: dict[str, Any],
        reserved: dict[str, str] | None = None,
    ) -> None:
        """Dibuja vehículos y celdas sobre un cuadro.

        Está separado del análisis para poder pintar el último resultado sobre
        cada cuadro nuevo del video sin tener que volver a correr YOLO.
        """
        reserved = reserved or {}
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                VEHICLE_CLASSES[vehicle.class_id],
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        cells_state = state.get("celdas", {})
        # Las celdas se dibujan donde quedan en este cuadro, si la cámara se movió.
        to_reference = self.aligner.last_transform if self.aligner is not None else None
        to_frame = np.linalg.inv(to_reference) if to_reference is not None else None
        for name, cell in self.cells.items():
            if cells_state.get(name, {}).get("ocupado"):
                color, status = (0, 0, 255), "Ocupado"
            elif name in reserved:
                color, status = (0, 215, 255), f"Reservado {reserved[name]}"
            else:
                color, status = (0, 255, 0), "Libre"
            points = cell["points"]
            if to_frame is not None:
                points = cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), to_frame)
                points = np.round(points).astype(np.int32).reshape(-1, 2)
            cv2.polylines(frame, [points], True, color, 2)
            cv2.putText(
                frame,
                f"{name}: {status}",
                tuple(points[0] + np.array([0, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )

    def analyze(
        self, frame: np.ndarray, reserved: dict[str, str] | None = None
    ) -> tuple[dict[str, Any], list[Vehicle]]:
        """Corre YOLO y decide qué celdas están ocupadas, sin modificar el cuadro."""
        reserved = reserved or {}
        height, width = frame.shape[:2]
        self._apply_cell_changes(width, height)
        occupied = {name: False for name in self.cells}
        vehicles = self.vehicles.detect(frame)
        # Las celdas están donde se trazaron: el punto de apoyo de cada vehículo se lleva allí.
        to_reference = self.aligner.transform(frame) if self.aligner is not None else None

        for vehicle in vehicles:
            foot = vehicle.footpoint
            moved = foot
            if to_reference is not None:
                x, y = cv2.perspectiveTransform(np.float32([[[foot.x, foot.y]]]), to_reference)[0, 0]
                moved = Point(float(x), float(y))
            for name, cell in self.cells.items():
                if cell["polygon"].covers(moved):
                    occupied[name] = True
                    vehicle.in_cell = True

        if self.aligner is not None:
            self.aligner.consider(frame, sum(occupied.values()))
        self.last_vehicles = vehicles
        occupied = self._smooth(occupied)

        occupied_count = sum(occupied.values())
        state = {
            "timestamp": time.time(),
            "total_celdas": len(self.cells),
            "ocupadas": occupied_count,
            "libres": len(self.cells) - occupied_count,
            "vehiculos_detectados": len(vehicles),
            "alineacion": {
                "referencia": self.aligner.source if self.aligner is not None else "ninguna",
                "compensando": to_reference is not None,
                "dx": round(self.aligner.last_shift[0], 1) if self.aligner is not None else 0.0,
                "dy": round(self.aligner.last_shift[1], 1) if self.aligner is not None else 0.0,
            },
            "congestion_porcentaje": round(occupied_count / len(self.cells) * 100, 2)
            if self.cells
            else 0.0,
            "celdas": {
                name: {
                    "ocupado": is_occupied,
                    "estado": "ocupado"
                    if is_occupied
                    else "reservado"
                    if name in reserved
                    else "libre",
                    "puesto_id": self.cell_meta.get(name, {}).get("puesto_id"),
                    "puesto": self.cell_meta.get(name, {}).get("puesto"),
                }
                for name, is_occupied in occupied.items()
            },
        }
        return state, vehicles

    def first_free_space(self, state: dict[str, Any], reserved: set[str]) -> str | None:
        return next(
            (
                name
                for name, cell in state.get("celdas", {}).items()
                if not cell["ocupado"] and name not in reserved
            ),
            None,
        )

    @property
    def total_spaces(self) -> int:
        return len(self.cells)
