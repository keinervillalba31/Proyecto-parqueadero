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
from .plate_reader import PlateReader, vehicle_crops
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
            minimum_box_area=settings.minimum_box_area,
            minimum_motorcycle_area=settings.minimum_motorcycle_area,
            occupancy_frames=settings.occupancy_frames,
        )
        self.plates = PlateReader(
            workspace=settings.roboflow_workspace,
            workflow=settings.roboflow_workflow,
            confirmation_reads=settings.plate_confirmation_reads,
            cooldown_seconds=settings.plate_cooldown_seconds,
            plate_pattern=settings.plate_pattern,
        )
        self.state = RuntimeState(self.parking.total_spaces)
        self.registered_vehicles = self._load_registered_vehicles(
            settings.registered_vehicles_path
        )
        # placa -> {"celda", "desde", "estacionado"}; la usan el hilo de video
        # y el de placas, por eso tiene su propio candado.
        self.reservations: dict[str, dict[str, Any]] = {}
        self.reservations_lock = threading.Lock()
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

    def _source_is_file(self) -> bool:
        return Path(self.settings.video_source).is_file()

    def _plate_attempt_allowed(self, vehicle_detected: bool) -> bool:
        if self.settings.plate_require_vehicle and not vehicle_detected:
            return False
        just_arrived = vehicle_detected and not self.vehicle_was_detected
        return (
            just_arrived or self.frame_number % self.settings.plate_interval_frames == 0
        )

    def reserved_cells(self) -> dict[str, str]:
        """Devuelve celda -> placa de las reservas vigentes."""
        with self.reservations_lock:
            return {data["celda"]: plate for plate, data in self.reservations.items()}

    def _assign_space(self, plate: str, parking_state: dict[str, Any]) -> str | None:
        with self.reservations_lock:
            if plate in self.reservations:
                return self.reservations[plate]["celda"]
            taken = {data["celda"] for data in self.reservations.values()}
            space = self.parking.first_free_space(parking_state, taken)
            if space:
                self.reservations[plate] = {
                    "celda": space,
                    "desde": time.time(),
                    "estacionado": False,
                }
            return space

    def _update_reservations(self, parking_state: dict[str, Any]) -> None:
        """Marca las reservas ocupadas y libera las que ya no se usan.

        - Si la celda reservada se ocupa, el vehículo se considera estacionado.
        - Si estaba estacionado y la celda queda libre, el vehículo salió.
        - Si nunca llegó a ocuparla dentro del tiempo límite, se libera.
        """
        cells = parking_state.get("celdas", {})
        now = time.time()
        with self.reservations_lock:
            for plate, data in list(self.reservations.items()):
                occupied = cells.get(data["celda"], {}).get("ocupado", False)
                if occupied:
                    data["estacionado"] = True
                elif data["estacionado"]:
                    del self.reservations[plate]
                elif now - data["desde"] > self.settings.reservation_timeout_seconds:
                    del self.reservations[plate]
            parking_state["reservas"] = {
                plate: dict(data) for plate, data in self.reservations.items()
            }

    def _plate_images(self, clean_frame: np.ndarray) -> list[np.ndarray]:
        """Imágenes a enviar al OCR: un recorte por vehículo o el cuadro completo."""
        if not self.settings.plate_crop_vehicles:
            return [clean_frame]
        return vehicle_crops(
            clean_frame,
            self.parking.last_vehicles,
            max_vehicles=self.settings.plate_max_vehicles,
            minimum_width=self.settings.plate_min_crop_width,
            skip_parked=self.settings.plate_skip_parked,
        )

    def _publish_confirmed(self, plate: str, read_count: int, last_read: str) -> None:
        registered = plate in self.registered_vehicles
        assigned_space = (
            self._assign_space(plate, self.state.snapshot()) if registered else None
        )
        self.state.update_plate(
            {
                "valor": plate,
                "lecturas": [plate] * read_count,
                "ultimo_resultado": last_read,
                "registrado": registered,
                "estado": "autorizado" if registered else "no_registrado",
                "celda_asignada": assigned_space,
                "ultimo_intento": time.time(),
                "error": None,
            },
            replace=True,
        )
        self.plates.start_cooldown(plate)

    def _read_plate(self, images: list[np.ndarray]) -> None:
        try:
            self.state.update_plate(
                {"ultimo_intento": time.time(), "error": None, "estado": "leyendo"}
            )
            plates = [plate for image in images for plate in self.plates.read_all(image)]
            if not plates:
                self.state.update_plate({"estado": "sin_lectura"})
                return

            confirmed_any = False
            for plate in plates:
                confirmed = self.plates.confirm(plate)
                if confirmed:
                    self._publish_confirmed(*confirmed, last_read=plate)
                    confirmed_any = True

            if not confirmed_any:
                self.state.update_plate(
                    {
                        "lecturas": list(self.plates.reads),
                        "ultimo_resultado": plates[-1],
                        "estado": "confirmando",
                    }
                )
        except Exception as error:
            self.state.update_plate({"estado": "error_lectura", "error": str(error)})

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
                if not ok and self._source_is_file():
                    # Un video de prueba terminó: se repite desde el inicio.
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok:
                    capture.release()
                    capture = None
                    self.state.worker_error = "La fuente dejó de entregar cuadros"
                    time.sleep(self.settings.video_retry_seconds)
                    continue

                self.frame_number += 1
                # Se copia antes de detectar porque `detect` dibuja sobre el cuadro.
                clean_frame = frame.copy()
                annotated, parking_state = self.parking.detect(
                    frame, self.reserved_cells()
                )
                self._update_reservations(parking_state)
                vehicle_detected = parking_state["vehiculos_detectados"] > 0
                if self._plate_attempt_allowed(vehicle_detected) and (
                    self.plate_thread is None or not self.plate_thread.is_alive()
                ):
                    images = self._plate_images(clean_frame)
                    if images:
                        self.plate_thread = threading.Thread(
                            target=self._read_plate,
                            args=(images,),
                            name="plate-reader",
                            daemon=True,
                        )
                        self.plate_thread.start()
                self.vehicle_was_detected = vehicle_detected

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
