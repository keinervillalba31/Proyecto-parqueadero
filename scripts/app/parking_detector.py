from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


VEHICLE_CLASSES = {2: "Carro", 3: "Moto", 5: "Bus", 7: "Camion"}


class ParkingDetector:
    def __init__(
        self,
        cells_path: Path,
        model_path: Path,
        confidence: float,
        image_size: int,
        tile_grid: int,
        coverage_threshold: float,
        minimum_box_area: int,
        minimum_motorcycle_area: int,
    ) -> None:
        with cells_path.open("r", encoding="utf-8") as file:
            parking_cells = json.load(file)

        self.cells = {
            name: {
                "points": np.array(points, dtype=np.int32),
                "polygon": Polygon(points),
            }
            for name, points in parking_cells.items()
        }
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.tile_grid = max(tile_grid, 1)
        self.coverage_threshold = coverage_threshold
        self.minimum_box_area = minimum_box_area
        self.minimum_motorcycle_area = minimum_motorcycle_area

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        occupied = {name: False for name in self.cells}
        detected_vehicles = 0
        detections = self._detect_with_tiles(frame)

        for class_id, confidence, coordinates in detections:
            x1, y1, x2, y2 = coordinates
            minimum_area = (
                self.minimum_motorcycle_area
                if class_id == 3
                else self.minimum_box_area
            )
            if (x2 - x1) * (y2 - y1) < minimum_area:
                continue

            detected_vehicles += 1
            vehicle_footpoint = Point((x1 + x2) / 2, y2)
            label = VEHICLE_CLASSES[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            for name, cell in self.cells.items():
                if cell["polygon"].covers(vehicle_footpoint):
                    occupied[name] = True

        for name, cell in self.cells.items():
            is_occupied = occupied[name]
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            status = "Ocupado" if is_occupied else "Libre"
            points = cell["points"]
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

        occupied_count = sum(occupied.values())
        state = {
            "timestamp": time.time(),
            "total_celdas": len(self.cells),
            "ocupadas": occupied_count,
            "libres": len(self.cells) - occupied_count,
            "vehiculos_detectados": detected_vehicles,
            "congestion_porcentaje": round(occupied_count / len(self.cells) * 100, 2)
            if self.cells
            else 0.0,
            "celdas": {
                name: {
                    "ocupado": is_occupied,
                    "estado": "ocupado" if is_occupied else "libre",
                }
                for name, is_occupied in occupied.items()
            },
        }
        return frame, state

    def _detect_with_tiles(
        self, frame: np.ndarray
    ) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        height, width = frame.shape[:2]
        candidates: list[tuple[int, float, tuple[int, int, int, int]]] = []

        sources = [(frame, 0, 0)]
        if self.tile_grid > 1:
            tile_width = min(width, int(width / self.tile_grid * 1.35))
            tile_height = min(height, int(height / self.tile_grid * 1.35))
            x_starts = sorted({0, max(width - tile_width, 0)})
            y_starts = sorted({0, max(height - tile_height, 0)})
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

        boxes = [[x1, y1, x2 - x1, y2 - y1] for _, _, (x1, y1, x2, y2) in candidates]
        kept = cv2.dnn.NMSBoxes(
            boxes,
            [confidence for _, confidence, _ in candidates],
            self.confidence,
            0.45,
        )
        return [candidates[index] for index in np.array(kept).flatten()]

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
