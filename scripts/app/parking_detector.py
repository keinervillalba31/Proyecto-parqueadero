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
# Para dos cajas del cuadro completo (ver _drop_partial_boxes).
_CONTAINED_RATIO_FULL_FRAME = 0.9


def _drop_partial_boxes(
    detections: list[tuple[int, float, tuple[int, int, int, int]]],
    from_crops: list[bool] | None = None,
) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    """Quita las cajas parciales de un vehículo que ya tiene su caja completa.

    Cuando el borde de un recorte corta un carro, YOLO devuelve una caja con solo una parte;
    se superpone poco con la caja completa, así que el NMS la deja pasar y su punto de apoyo
    cae en el puesto vecino: el carro se cuenta dos veces.

    `from_crops` dice qué cajas salieron de un recorte. A esas se les quita si quedan dentro de
    otra en más de un 70 %. A las del cuadro completo se les exige un 90 %: en una fila vista en
    perspectiva el carro de atrás queda en buena parte dentro de la caja del de adelante y es otro
    vehículo, pero YOLO también da a veces una segunda caja, más chica, de la parte de arriba del
    mismo carro. Medido en el video de Archivos, el 90 % quita esos duplicados (sin ellos, cero
    falsos positivos en cuadros sueltos) y pierde 7 puestos de 258 en vez de 11 con el 70 %.
    Sin `from_crops`, todas cuentan como de recorte.
    """
    crop_box = {id(d): True for d in detections} if from_crops is None else {
        id(d): crop for d, crop in zip(detections, from_crops)
    }
    def area(box: tuple[int, int, int, int]) -> int:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def contained(small: tuple[int, int, int, int], large: tuple[int, int, int, int]) -> float:
        width = min(small[2], large[2]) - max(small[0], large[0])
        height = min(small[3], large[3]) - max(small[1], large[1])
        return max(0, width) * max(0, height) / max(area(small), 1)

    kept: list[tuple[int, float, tuple[int, int, int, int]]] = []
    for detection in sorted(detections, key=lambda item: area(item[2]), reverse=True):
        limit = _CONTAINED_RATIO if crop_box[id(detection)] else _CONTAINED_RATIO_FULL_FRAME
        if all(contained(detection[2], other[2]) < limit for other in kept):
            kept.append(detection)
    return kept


def _touches_border(box: tuple[int, int, int, int], width: int, height: int, margin: int = 4) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def far_cells(polygons: list[np.ndarray], relative_area: float = 0.4) -> list[int]:
    """Índices de las celdas lejanas: las de menos de `relative_area` del área de la más grande."""
    if not polygons:
        return []
    areas = [abs(cv2.contourArea(p.astype(np.float32))) for p in polygons]
    largest = max(areas)
    return [i for i, a in enumerate(areas) if 0 < a < largest * relative_area]


def far_zone(
    polygons: list[np.ndarray], width: int, height: int, relative_area: float = 0.4, min_zoom: float = 1.4,
    image_size: int = 1280,
) -> tuple[int, int, int, int] | None:
    """Recuadro de las celdas lejanas, para analizarlo otra vez ampliado.

    Lejanas son las celdas de menos de `relative_area` del área de la más grande: en la imagen
    son tan chicas que una moto ocupa unos 15×20 px y el modelo no la ve. El recuadro sube por
    encima de las celdas (los vehículos sobresalen hacia arriba) y solo vale la pena si YOLO lo
    va a ampliar al menos `min_zoom` veces. None si no hay celdas lejanas (cámara cenital).
    """
    small = [polygons[i] for i in far_cells(polygons, relative_area)]
    if not small:
        return None
    points = np.concatenate(small).astype(np.float32)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    rise = (y1 - y0) * 0.6 + 20
    box = (int(max(0, x0 - 20)), int(max(0, y0 - rise)), int(min(width, x1 + 20)), int(min(height, y1 + 10)))
    if image_size / max(box[2] - box[0], box[3] - box[1], 1) < min_zoom:
        return None
    return box


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
        # Zonas (x0, y0, x1, y1) que se analizan otra vez ampliadas: las celdas lejanas (ver far_zone).
        self.zoom_zones: list[tuple[int, int, int, int]] = []

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
        from_crop: list[bool] = []

        # (imagen, desplazamiento x, y, es recorte, descartar cajas que tocan su borde)
        sources = [(frame, 0, 0, False, False)]
        if self.tile_grid > 1:
            tile_width = min(width, int(width / self.tile_grid * 1.35))
            tile_height = min(height, int(height / self.tile_grid * 1.35))
            x_starts = self._tile_starts(width, tile_width)
            y_starts = self._tile_starts(height, tile_height)
            sources.extend(
                (frame[y : y + tile_height, x : x + tile_width], x, y, True, False)
                for y in y_starts
                for x in x_starts
            )
        # Las zonas ampliadas solo aportan lo que el cuadro completo no ve bien (vehículos chicos):
        # un vehículo cortado por el borde de la zona ya lo detecta el cuadro completo.
        sources.extend((frame[y0:y1, x0:x1], x0, y0, True, True) for x0, y0, x1, y1 in self.zoom_zones)

        for source, offset_x, offset_y, is_crop, skip_cut in sources:
            source_height, source_width = source.shape[:2]
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
                if skip_cut and _touches_border((x1, y1, x2, y2), source_width, source_height):
                    continue
                candidates.append(
                    (
                        class_id,
                        float(detection.conf[0]),
                        (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    )
                )
                from_crop.append(is_crop)

        if not candidates:
            return []
        boxes = [[x1, y1, x2 - x1, y2 - y1] for _, _, (x1, y1, x2, y2) in candidates]
        kept = np.array(
            cv2.dnn.NMSBoxes(
                boxes,
                [confidence for _, confidence, _ in candidates],
                self.confidence,
                0.45,
            )
        ).flatten()
        return _drop_partial_boxes([candidates[i] for i in kept], [from_crop[i] for i in kept])

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
        zoom_far_cells: bool = False,
        still_seconds: float = 0.0,
        free_seconds_near: float | None = None,
        footpoint_lift: float = 0.0,
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
        # Analizar otra vez, ampliada, la zona de las celdas lejanas (motos del fondo).
        self.zoom_far_cells = zoom_far_cells
        self._zoom_key: tuple[int, int, int] | None = None
        # Contra falsos positivos:
        # - Un vehículo solo ocupa un puesto si lleva `still_seconds` quieto (su punto de apoyo no
        #   se mueve): el carro que pasa por el pasillo o el que maniobra no cuenta.
        # - Las celdas cercanas, donde el detector casi no pierde los carros, se liberan tras
        #   `free_seconds_near` (None: igual que las lejanas, `free_seconds`).
        # - El punto de apoyo sube `footpoint_lift` de la altura de la caja: la caja incluye la
        #   sombra y la defensa, y su borde de abajo cae en el puesto de adelante.
        self.still_seconds = max(still_seconds, 0.0)
        self.free_seconds_near = free_seconds_near
        self.footpoint_lift = max(footpoint_lift, 0.0)
        # Cuánto puede moverse el punto de apoyo (fracción del lado mayor de la caja) sin dejar
        # de contar como quieto, y cuánto se recuerda un vehículo que el detector dejó de ver.
        self.still_tolerance = 0.15
        self.still_memory_seconds = 10.0
        self._tracks: list[dict[str, Any]] = []
        self._far_cell_names: set[str] = set()
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

    def _update_zoom_zone(self, width: int, height: int) -> None:
        """Recalcula la zona ampliada cuando cambian las celdas o el tamaño del video."""
        key = (width, height, id(self.cells))
        if key == self._zoom_key:
            return
        self._zoom_key = key
        names = list(self.cells)
        self._far_cell_names = {
            names[i] for i in far_cells([cell["points"] for cell in self.cells.values()])
        }
        zone = (
            far_zone([cell["points"] for cell in self.cells.values()], width, height,
                     image_size=self.vehicles.image_size)
            if self.zoom_far_cells
            else None
        )
        self.vehicles.zoom_zones = [zone] if zone else []

    def _free_seconds_for(self, name: str) -> float:
        if self.free_seconds_near is None or name in self._far_cell_names:
            return self.free_seconds
        return max(self.free_seconds_near, 0.0)

    def footpoint_of(self, vehicle: Vehicle, to_reference: np.ndarray | None = None) -> Point:
        """Punto de apoyo del vehículo en las coordenadas de las celdas (subido `footpoint_lift`)."""
        x1, y1, x2, y2 = vehicle.box
        x, y = (x1 + x2) / 2, y2 - (y2 - y1) * self.footpoint_lift
        if to_reference is not None:
            x, y = cv2.perspectiveTransform(np.float32([[[x, y]]]), to_reference)[0, 0]
        return Point(float(x), float(y))

    def cell_of(self, point: Point) -> str | None:
        """La celda donde se apoya el punto; si cae en dos (bordes que se tocan), la de centro más cercano."""
        covering = [name for name, cell in self.cells.items() if cell["polygon"].covers(point)]
        if len(covering) <= 1:
            return covering[0] if covering else None
        return min(covering, key=lambda name: self.cells[name]["polygon"].centroid.distance(point))

    def _still(self, vehicles: list[Vehicle], points: list[Point], now: float) -> list[bool]:
        """Si cada vehículo lleva `still_seconds` quieto en el mismo lugar.

        Sigue cada punto de apoyo de un análisis al siguiente. Un vehículo cuenta como quieto si
        su punto se queda cerca de donde apareció (tolerancia según el tamaño de la caja, porque
        la caja tiembla). La pista sobrevive unos segundos sin verlo: YOLO pierde a ratos los
        vehículos lejanos y no por eso deben volver a empezar.
        """
        if self.still_seconds <= 0:
            return [True] * len(vehicles)
        memory = max(self.still_seconds, self.still_memory_seconds)
        self._tracks = [t for t in self._tracks if now - t["seen"] <= memory]
        free = set(range(len(self._tracks)))
        result = []
        for vehicle, point in zip(vehicles, points):
            x1, y1, x2, y2 = vehicle.box
            tolerance = 3.0 + self.still_tolerance * max(y2 - y1, x2 - x1)
            best, best_distance = None, tolerance
            for i in free:
                distance = point.distance(self._tracks[i]["anchor"])
                if distance <= best_distance:
                    best, best_distance = i, distance
            if best is None:
                self._tracks.append({"anchor": point, "since": now, "seen": now})
                result.append(False)
                continue
            free.discard(best)
            track = self._tracks[best]
            track["seen"] = now
            result.append(now - track["since"] >= self.still_seconds)
        return result

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
            if now - started >= (self.occupy_seconds if is_occupied else self._free_seconds_for(name)):
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
        self._update_zoom_zone(width, height)
        occupied = {name: False for name in self.cells}
        vehicles = self.vehicles.detect(frame)
        # Las celdas están donde se trazaron: el punto de apoyo de cada vehículo se lleva allí.
        to_reference = self.aligner.transform(frame) if self.aligner is not None else None

        occupied = self._decide(vehicles, to_reference)

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

    def _decide(self, vehicles: list[Vehicle], to_reference: np.ndarray | None) -> dict[str, bool]:
        """Qué celdas tienen apoyado un vehículo quieto en este análisis (antes del suavizado)."""
        occupied = {name: False for name in self.cells}
        points = [self.footpoint_of(vehicle, to_reference) for vehicle in vehicles]
        still = self._still(vehicles, points, self._clock())
        for vehicle, point, is_still in zip(vehicles, points, still):
            name = self.cell_of(point)
            if name is None:
                continue
            vehicle.in_cell = True
            if is_still:
                occupied[name] = True
        return occupied

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
