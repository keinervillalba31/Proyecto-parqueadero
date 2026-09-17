from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .parking_detector import ParkingDetector
from .plate_reader import PlateReader
from .state import RuntimeState


class VideoService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parking = ParkingDetector(
            cells_path=settings.cells_path,
            model_path=settings.model_path,
            confidence=settings.yolo_confidence,
            image_size=settings.yolo_image_size,
            tile_grid=settings.yolo_tile_grid,
            coverage_threshold=settings.coverage_threshold,
            minimum_box_area=settings.minimum_box_area,
            minimum_motorcycle_area=settings.minimum_motorcycle_area,
        )
        self.plates = PlateReader(
            workspace=settings.roboflow_workspace,
            workflow=settings.roboflow_workflow,
            confirmation_reads=settings.plate_confirmation_reads,
            cooldown_seconds=settings.plate_cooldown_seconds,
        )
        self.state = RuntimeState(self.parking.total_spaces)
        self.registered_vehicles = self._load_registered_vehicles(
            settings.registered_vehicles_path
        )
        self.reserved_spaces: dict[str, str] = {}
        self.plate_thread: threading.Thread | None = None
        self.vehicle_was_detected = False
        self.frame_number = 0
        self.thread: threading.Thread | None = None
        self.running = False

    @staticmethod
    def _load_registered_vehicles(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, name="video-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def _open_source(self) -> cv2.VideoCapture:
        source: int | str = (
            int(self.settings.video_source)
            if self.settings.video_source.isdigit()
            else self.settings.video_source
        )
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
        return capture

    def _plate_attempt_allowed(self, vehicle_detected: bool) -> bool:
        if self.settings.plate_require_vehicle and not vehicle_detected:
            return False
        just_arrived = vehicle_detected and not self.vehicle_was_detected
        return (
            just_arrived or self.frame_number % self.settings.plate_interval_frames == 0
        )

    def _read_plate(self, frame: np.ndarray) -> None:
        try:
            self.state.update_plate(
                {
                    **self.state.snapshot().get("placa", {}),
                    "ultimo_intento": time.time(),
                    "error": None,
                    "estado": "leyendo",
                }
            )
            plate = self.plates.read(frame)
            if not plate:
                return

            confirmed = self.plates.confirm(plate)
            current = self.state.snapshot()
            plate_state = {
                **current.get("placa", {}),
                "lecturas": list(self.plates.reads),
                "ultimo_resultado": plate,
            }
            if confirmed is None:
                self.state.update_plate(plate_state)
                return

            confirmed_plate, read_count = confirmed
            registered = confirmed_plate in self.registered_vehicles
            assigned_space = self.reserved_spaces.get(confirmed_plate)
            if registered and assigned_space is None:
                assigned_space = self.parking.first_free_space(
                    current, set(self.reserved_spaces.values())
                )
                if assigned_space:
                    self.reserved_spaces[confirmed_plate] = assigned_space

            self.state.update_plate(
                {
                    "valor": confirmed_plate,
                    "lecturas": [confirmed_plate] * read_count,
                    "registrado": registered,
                    "estado": "autorizado" if registered else "no_registrado",
                    "celda_asignada": assigned_space,
                    "ultimo_intento": time.time(),
                    "error": None,
                }
            )
            self.plates.next_attempt_at = time.monotonic() + self.settings.plate_cooldown_seconds
        except Exception as error:
            current = self.state.snapshot().get("placa", {})
            self.state.update_plate(
                {**current, "estado": "error_lectura", "error": str(error)}
            )

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        while self.running:
            try:
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    capture = self._open_source()
                    self.state.worker_error = None

                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    capture = None
                    self.state.worker_error = "La fuente dejó de entregar cuadros"
                    time.sleep(self.settings.video_retry_seconds)
                    continue

                self.frame_number += 1
                annotated, parking_state = self.parking.detect(frame)
                vehicle_detected = parking_state["vehiculos_detectados"] > 0
                if (
                    self._plate_attempt_allowed(vehicle_detected)
                    and self.plates.next_attempt_at <= time.monotonic()
                    and (self.plate_thread is None or not self.plate_thread.is_alive())
                ):
                    self.plate_thread = threading.Thread(
                        target=self._read_plate,
                        args=(frame.copy(),),
                        name="plate-reader",
                        daemon=True,
                    )
                    self.plate_thread.start()
                self.vehicle_was_detected = vehicle_detected

                parking_state["placa"] = self.state.snapshot().get("placa", {})
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if encoded_ok:
                    self.state.set_frame(encoded.tobytes())
                self.state.update(parking_state)
            except Exception as error:
                self.state.worker_error = str(error)
                if capture is not None:
                    capture.release()
                    capture = None
                time.sleep(self.settings.video_retry_seconds)

        if capture is not None:
            capture.release()

    def health(self) -> dict[str, str | bool]:
        return {"ok": self.state.worker_error is None, "error": self.state.worker_error or ""}
