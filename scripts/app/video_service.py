from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .backend_client import BackendClient, BackendError, PlateEntryResult
from .config import Settings
from .parking_detector import ParkingDetector, VehicleDetector
from .plate_reader import PlateReader, vehicle_crops
from .state import RuntimeState

log = logging.getLogger("parqueadero")


class VideoService:
    # Lo que el operador enciende o apaga desde el panel. El video sigue
    # transmitiéndose aunque los dos estén apagados.
    spaces_enabled = True
    plates_enabled = True
    backend: BackendClient | None = None

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = BackendClient(settings) if settings.backend_sync_enabled else None
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
        self.spaces_enabled, self.plates_enabled = self._load_services(
            settings.services_state_path
        )
        self.state.update_services(self.services())
        if not self.plates_enabled:
            self.state.update_plate({"estado": "detenido"})
        self.registered_vehicles = self._load_registered_vehicles(
            settings.registered_vehicles_path
        )
        # placa -> {"celda", "desde", "estacionado"}; la usan el hilo de video
        # y el de placas, por eso tiene su propio candado.
        self.reservations: dict[str, dict[str, Any]] = {}
        self.reservations_lock = threading.Lock()
        self.plate_thread: threading.Thread | None = None
        # placa -> hasta cuándo no se vuelve a consultar al backend por su primera lectura.
        self.first_read_checks: dict[str, float] = {}
        self.vehicle_was_detected = False
        self.frame_number = 0
        self.thread: threading.Thread | None = None
        self.running = False
        # Cuadro más reciente del video y último resultado de YOLO; los
        # comparten el hilo de video y el de detección.
        self.frame_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_id = 0
        self.plate_frame_lock = threading.Lock()
        self.latest_plate_frame: bytes | None = None
        self.latest_plate_frame_id = 0
        self.overlay: tuple[Any, ...] | None = None
        self.detect_thread: threading.Thread | None = None
        self.plate_video_thread: threading.Thread | None = None
        self.plate_vehicle_detector: VehicleDetector | None = None
        self.cells_thread: threading.Thread | None = None
        self.cells_version: str | None = None
        self.occupancy_thread: threading.Thread | None = None
        self.services_thread: threading.Thread | None = None
        # Última ocupación que aceptó el backend (puesto_id -> ocupado) y cuándo.
        self.reported_occupancy: dict[int, bool] | None = None
        self.reported_at = 0.0
        self.detector_failed = False
        self.analysis_seconds = 0.0
        self.next_frame_at = 0.0
        self.video_frames = 0
        self.fps_window_start = time.monotonic()
        self.active_source = settings.video_source
        self.loop_count = 0
        self.using_fallback = False
        self.last_primary_attempt = 0.0
        self.probe_lock = threading.Lock()
        self.probe_thread: threading.Thread | None = None
        self.probed_capture: cv2.VideoCapture | None = None

    @staticmethod
    def _load_registered_vehicles(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_services(path: Path) -> tuple[bool, bool]:
        """Lo que quedó encendido la última vez; sin archivo, todo encendido."""
        try:
            with path.open("r", encoding="utf-8") as file:
                saved = json.load(file)
            return bool(saved.get("espacios", True)), bool(saved.get("placas", True))
        except (OSError, ValueError, AttributeError):
            return True, True

    def services(self) -> dict[str, bool]:
        return {"espacios": self.spaces_enabled, "placas": self.plates_enabled}

    def set_services(
        self, spaces: bool | None = None, plates: bool | None = None
    ) -> dict[str, bool]:
        """Cambio pedido desde el panel local (None = no cambiar).

        Con el backend activo manda lo que diga la plataforma, así que el cambio
        se guarda primero allá; si no, la siguiente consulta lo desharía.
        """
        if self.backend:
            try:
                self.backend.set_services(spaces, plates)
            except BackendError as error:
                log.warning("No se pudo guardar el cambio de servicios en la plataforma: %s", error)
        return self._apply_services(spaces, plates)

    def _apply_services(
        self, spaces: bool | None = None, plates: bool | None = None
    ) -> dict[str, bool]:
        """Enciende o apaga la detección de espacios y la lectura de placas (None = no cambiar).

        Se guarda en disco para que un reinicio del servicio respete lo que dejó el operador.
        """
        if (spaces is None or spaces == self.spaces_enabled) and (
            plates is None or plates == self.plates_enabled
        ):
            return self.services()
        if spaces is not None:
            self.spaces_enabled = spaces
        if plates is not None and plates != self.plates_enabled:
            self.plates_enabled = plates
            self.state.update_plate({"estado": "esperando" if plates else "detenido"})
        current = self.services()
        self.state.update_services(current)
        log.info(
            "Servicios: espacios %s, placas %s",
            "encendido" if current["espacios"] else "apagado",
            "encendido" if current["placas"] else "apagado",
        )
        try:
            self.settings.services_state_path.write_text(json.dumps(current), encoding="utf-8")
        except OSError as error:
            log.warning("No se pudo guardar el estado de los servicios: %s", error)
        return current

    def _services_loop(self) -> None:
        """Obedece los interruptores de la plataforma (Monitoreo > Servicios de detección)."""
        failing = False
        while self.running:
            try:
                wanted = self.backend.get_services()
            except BackendError as error:
                if not failing:
                    log.warning("No se pudieron consultar los servicios en la plataforma: %s", error)
                failing = True
            else:
                failing = False
                self._apply_services(wanted["espacios"], wanted["placas"])
            waited = 0.0
            while self.running and waited < self.settings.backend_services_poll_seconds:
                time.sleep(0.5)
                waited += 0.5

    def _detection_needed(self) -> bool:
        """YOLO corre para los espacios, o para las placas si comparten la cámara del parqueadero."""
        return self.spaces_enabled or (
            self.plates_enabled and not self.settings.plate_video_source
        )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        if self.backend:
            self.services_thread = threading.Thread(
                target=self._services_loop, name="services-sync", daemon=True
            )
            self.services_thread.start()
        if self.backend and self.settings.backend_camera_id is not None:
            self.cells_thread = threading.Thread(target=self._cells_loop, name="cells-sync", daemon=True)
            self.cells_thread.start()
            self.occupancy_thread = threading.Thread(
                target=self._occupancy_loop, name="occupancy-sync", daemon=True
            )
            self.occupancy_thread.start()
        self.thread = threading.Thread(target=self._run, name="video-worker", daemon=True)
        self.detect_thread = threading.Thread(
            target=self._detect_loop, name="detector-worker", daemon=True
        )
        self.thread.start()
        self.detect_thread.start()
        if self.settings.plate_video_source:
            self.plate_video_thread = threading.Thread(
                target=self._run_plate_video, name="plate-video-worker", daemon=True
            )
            self.plate_video_thread.start()

    def stop(self) -> None:
        self.running = False
        for worker in (
            self.thread,
            self.detect_thread,
            self.plate_video_thread,
            self.cells_thread,
            self.occupancy_thread,
            self.services_thread,
        ):
            if worker and worker is not threading.current_thread():
                worker.join(timeout=2)
        with self.probe_lock:
            if self.probed_capture is not None:
                self.probed_capture.release()
                self.probed_capture = None

    @staticmethod
    def _open_capture(source_text: str) -> cv2.VideoCapture:
        source: int | str = int(source_text) if source_text.isdigit() else source_text
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
        return capture

    def _use_fallback(self) -> bool:
        fallback = self.settings.video_fallback
        return bool(fallback) and fallback != self.settings.video_source

    def _open_source(self) -> cv2.VideoCapture:
        """Abre la cámara principal; si no responde, el video de respaldo en bucle."""
        try:
            capture = self._open_capture(self.settings.video_source)
        except RuntimeError:
            if not self._use_fallback():
                raise
            capture = self._open_capture(self.settings.video_fallback)
            self._set_source(self.settings.video_fallback, using_fallback=True)
            return capture
        self._set_source(self.settings.video_source, using_fallback=False)
        return capture

    def _set_source(self, origin: str, using_fallback: bool) -> None:
        if using_fallback:
            log.warning("Cámara principal no disponible; usando el video de respaldo en bucle: %s", origin)
        else:
            log.info("Usando la fuente principal: %s", origin)
        self.active_source = origin
        self.using_fallback = using_fallback
        self.last_primary_attempt = time.monotonic()
        self.state.update_source(
            {
                "modo": "respaldo" if using_fallback else "principal",
                "origen": origin,
                "respaldo": self.settings.video_fallback or None,
            }
        )

    def _source_is_file(self) -> bool:
        return Path(self.active_source).is_file()

    def _probe_primary(self) -> None:
        """Intenta abrir la cámara principal en un hilo aparte.

        Abrir una URL que no responde puede tardar decenas de segundos; hacerlo
        en el hilo del video congelaría la reproducción del respaldo.
        """
        try:
            capture = self._open_capture(self.settings.video_source)
        except RuntimeError:
            return
        with self.probe_lock:
            if self.probed_capture is not None:
                self.probed_capture.release()
            self.probed_capture = capture

    def _maybe_switch_to_primary(self, capture: cv2.VideoCapture) -> cv2.VideoCapture:
        """Si la cámara principal ya responde, deja el respaldo y pasa a ella."""
        if not self.using_fallback:
            return capture

        with self.probe_lock:
            ready, self.probed_capture = self.probed_capture, None
        if ready is not None:
            capture.release()
            self._set_source(self.settings.video_source, using_fallback=False)
            return ready

        due = time.monotonic() - self.last_primary_attempt >= self.settings.video_camera_retry_seconds
        if due and (self.probe_thread is None or not self.probe_thread.is_alive()):
            self.last_primary_attempt = time.monotonic()
            self.probe_thread = threading.Thread(
                target=self._probe_primary, name="camera-probe", daemon=True
            )
            self.probe_thread.start()
        return capture

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

    def _check_first_read(self, plate: str) -> threading.Thread | None:
        """Primera lectura de una placa: se le pregunta de una vez al backend si la conoce.

        Si es de un estudiante, eso ya la confirma y no se espera a repetir la
        lectura (que es lo que más demora). Corre en su propio hilo para no
        frenar la lectura; si el backend no la conoce, la confirmación por
        lecturas repetidas sigue su curso. Cada placa se consulta una sola vez
        por PLATE_COOLDOWN_SECONDS.
        """
        if not self.backend or self.plates.in_cooldown(plate):
            return None
        now = time.monotonic()
        self.first_read_checks = {
            read: until for read, until in self.first_read_checks.items() if until > now
        }
        if plate in self.first_read_checks:
            return None
        self.first_read_checks[plate] = now + self.settings.plate_cooldown_seconds
        worker = threading.Thread(
            target=self._sync_first_read, args=(plate,), name="backend-first-read", daemon=True
        )
        worker.start()
        return worker

    def _sync_first_read(self, plate: str) -> None:
        try:
            result = self.backend.sync_plate_detection(plate)
        except BackendError as error:
            log.warning("No se pudo consultar la placa %s en el backend: %s", plate, error)
            return
        if result.outcome == "NOT_REGISTERED":
            return
        log.info("Placa %s reconocida por el backend en la primera lectura (%s)", plate, result.outcome)
        self._publish_confirmed(plate, 1, last_read=plate, backend_result=result)

    def _publish_confirmed(
        self,
        plate: str,
        read_count: int,
        last_read: str,
        backend_result: PlateEntryResult | None = None,
    ) -> None:
        registered = plate in self.registered_vehicles
        # Con los espacios apagados la ocupación está desactualizada: no se reserva celda local.
        assigned_space = (
            self._assign_space(plate, self.state.snapshot())
            if registered and self.spaces_enabled
            else None
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
                "backend_estado": "pendiente" if self.backend else "deshabilitado",
            },
            replace=True,
        )
        self.plates.start_cooldown(plate)
        if backend_result is not None:
            # El backend ya respondió (primera lectura): no hace falta volver a enviarla.
            self._apply_backend_result(backend_result)
            self._recheck_soon_if_pending(plate, backend_result)
        elif self.backend:
            threading.Thread(
                target=self._sync_with_backend,
                args=(plate, last_read),
                name="backend-sync",
                daemon=True,
            ).start()

    def _sync_with_backend(self, plate: str, other_read: str | None = None) -> None:
        """Le avisa al backend de la placa confirmada.

        Corre en su propio hilo: una falla de red o un rechazo del backend
        nunca debe detener la detección de video ni el reconocimiento de placas.
        """
        try:
            result = self.backend.sync_plate_detection(plate)
            if result.outcome == "NOT_REGISTERED" and other_read and other_read != plate:
                # Dos lecturas casi iguales (ABC128 y ABC123) se confirmaron entre sí:
                # si el backend no conoce la elegida pero sí la otra, la otra es la real.
                other = self.backend.sync_plate_detection(other_read)
                if other.outcome != "NOT_REGISTERED":
                    self.state.update_plate({"valor": other_read})
                    result = other
        except BackendError as error:
            self.state.update_plate({"backend_estado": "error", "backend_error": str(error)})
            return
        self._apply_backend_result(result)
        self._recheck_soon_if_pending(result.plate, result)

    def _recheck_soon_if_pending(self, plate: str, result: PlateEntryResult) -> None:
        """Acorta la espera cuando el backend conoce la placa pero aún no hizo nada con ella.

        "Ya tiene puesto" o "no hay cupo" pueden cambiar en segundos (libera el
        puesto, se desocupa una celda); esperar PLATE_COOLDOWN_SECONDS completos
        dejaría al estudiante medio minuto sin su aviso. Cuando ya se le avisó o
        se le asignó, la espera larga sí aplica.
        """
        if result.outcome in ("PROMPTED", "ASSIGNED", "NOT_REGISTERED"):
            return
        wait = self.settings.plate_recheck_seconds
        self.plates.start_cooldown(plate, wait)
        self.first_read_checks[plate] = time.monotonic() + wait
    def _apply_backend_result(self, result: PlateEntryResult) -> None:
        if not result.assigned:
            # Se le avisó al estudiante, o no aplica (no registrada, ya tiene puesto...).
            self.state.update_plate(
                {"backend_estado": result.outcome.lower(), "backend_error": None}
            )
            return

        self.state.update_plate(
            {
                "backend_estado": "sincronizado",
                "backend_error": None,
                "backend_asignacion_id": result.assignment_id,
                "backend_puesto_id": result.parking_space_id,
                "backend_puesto": result.space_number,
                "backend_notificado": result.notified,
            }
        )

    def _read_plate(self, images: list[np.ndarray]) -> None:
        try:
            self.state.update_plate(
                {"ultimo_intento": time.time(), "error": None, "estado": "leyendo"}
            )
            started = time.perf_counter()
            plates = [plate for image in images for plate in self.plates.read_all(image)]
            log.info(
                "Lectura de placas: %d imagen(es) en %.1f s -> %s",
                len(images),
                time.perf_counter() - started,
                ", ".join(plates) or "nada",
            )
            if not self.plates_enabled:
                # Se apagó mientras se leía: el resultado se descarta.
                self.state.update_plate({"estado": "detenido"})
                return
            if not plates:
                self.state.update_plate({"estado": "sin_lectura"})
                return

            confirmed_any = False
            for plate in plates:
                confirmed = self.plates.confirm(plate)
                if confirmed:
                    self._publish_confirmed(*confirmed, last_read=plate)
                    confirmed_any = True
                else:
                    self._check_first_read(plate)

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

    def _refresh_cells(self) -> bool:
        """Trae las celdas de la plataforma y las aplica si cambiaron. True si hubo cambio."""
        try:
            data = self.backend.get_camera_cells(self.settings.backend_camera_id)
        except BackendError as error:
            log.warning("No se pudieron traer las celdas de la plataforma: %s", error)
            return False

        version = str(data.get("version", ""))
        if version == self.cells_version:
            return False
        self.cells_version = version

        cells = {
            cell["label"]: {
                "points": cell["points"],
                "puesto_id": cell.get("parkingSpaceId"),
                "puesto": cell.get("parkingSpaceNumber"),
            }
            for cell in data.get("cells", [])
        }
        if not cells:
            log.info("La plataforma no tiene celdas para esta cámara; se siguen usando las locales.")
            self.state.update_source({"celdas_origen": "local", "celdas_version": version})
            return False

        self.parking.set_relative_cells(cells)
        self.state.update_source({"celdas_origen": "plataforma", "celdas_version": version})
        log.info("Celdas actualizadas desde la plataforma: %d (versión %s)", len(cells), version)
        return True

    def _cells_loop(self) -> None:
        """Revisa cada cierto tiempo si el administrador cambió las celdas."""
        while self.running:
            self._refresh_cells()
            waited = 0.0
            while self.running and waited < self.settings.backend_cells_poll_seconds:
                time.sleep(0.5)
                waited += 0.5

    def _report_occupancy(self) -> bool:
        """Envía al backend la ocupación de las celdas vinculadas a un puesto. True si se envió.

        Solo se envía cuando cambia algo, o cada BACKEND_OCCUPANCY_RESEND_SECONDS
        para corregir el backend si se reinició o alguien cambió un puesto a mano.
        """
        if not self.spaces_enabled:
            return False
        cells = self.state.snapshot().get("celdas", {})
        occupancy = {
            cell["puesto_id"]: bool(cell.get("ocupado"))
            for cell in cells.values()
            if cell.get("puesto_id") is not None
        }
        if not occupancy:
            return False
        resend_due = time.monotonic() - self.reported_at >= self.settings.backend_occupancy_resend_seconds
        if occupancy == self.reported_occupancy and not resend_due:
            return False
        try:
            self.backend.report_occupancy(self.settings.backend_camera_id, occupancy)
        except BackendError as error:
            log.warning("No se pudo enviar la ocupación al backend: %s", error)
            return False
        if occupancy != self.reported_occupancy:
            log.info(
                "Ocupación enviada al backend: %d de %d puestos ocupados",
                sum(occupancy.values()),
                len(occupancy),
            )
        self.reported_occupancy = occupancy
        self.reported_at = time.monotonic()
        return True

    def _occupancy_loop(self) -> None:
        """Mantiene el estado de los puestos del backend al día con lo que ve la cámara."""
        while self.running:
            self._report_occupancy()
            waited = 0.0
            while self.running and waited < self.settings.backend_occupancy_report_seconds:
                time.sleep(0.5)
                waited += 0.5

    def _frame_interval(self, capture: cv2.VideoCapture) -> float:
        """Segundos entre cuadros para reproducir un archivo a su velocidad real.

        Una cámara entrega los cuadros a su propio ritmo; un archivo no, así que
        hay que frenarlo o se reproduciría tan rápido como lo lea el disco.
        """
        if not self._source_is_file():
            return 0.0
        fps = capture.get(cv2.CAP_PROP_FPS)
        return 1.0 / fps if fps and 1 <= fps <= 120 else 1.0 / 25

    def _pace(self, interval: float) -> None:
        if interval <= 0:
            self.next_frame_at = 0.0
            return
        now = time.monotonic()
        if self.next_frame_at == 0.0 or now - self.next_frame_at > 1.0:
            self.next_frame_at = now
        wait = self.next_frame_at - now
        if wait > 0:
            time.sleep(wait)
        self.next_frame_at += interval

    def _publish_frame(self, frame: np.ndarray) -> None:
        """Entrega el cuadro original al detector y al stream en vivo."""
        with self.frame_lock:
            self.latest_frame = frame
            self.latest_frame_id += 1
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if encoded_ok:
            self.state.set_frame(encoded.tobytes())
        self._count_video_frame()

    def _publish_plate_frame(self, frame: np.ndarray) -> None:
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded_ok:
            raise RuntimeError("No se pudo codificar un cuadro del video de placas")
        with self.plate_frame_lock:
            self.latest_plate_frame = encoded.tobytes()
            self.latest_plate_frame_id += 1

    def get_plate_frame(self) -> tuple[int, bytes | None]:
        with self.plate_frame_lock:
            return self.latest_plate_frame_id, self.latest_plate_frame

    def _plate_video_images(self, frame: np.ndarray) -> list[np.ndarray]:
        """Recorta los vehículos de la cámara de placas, como `probar_placas.py`.

        Tiene su propio YOLO porque el del parqueadero lo usa otro hilo. Si no
        hay ningún vehículo no se devuelve nada y así no se gasta una llamada a
        Roboflow.
        """
        if not self.settings.plate_crop_vehicles:
            return [frame]
        if self.plate_vehicle_detector is None:
            self.plate_vehicle_detector = VehicleDetector(
                model_path=self.settings.model_path,
                confidence=self.settings.yolo_confidence,
                image_size=self.settings.yolo_image_size,
                # Sin recortes: en la cámara de placas el vehículo está cerca y se ve
                # grande; los recortes son para los lejanos del parqueadero y aquí
                # solo multiplicarían por cinco el tiempo de cada intento.
                tile_grid=1,
                minimum_box_area=self.settings.minimum_box_area,
                minimum_motorcycle_area=self.settings.minimum_motorcycle_area,
            )
        return vehicle_crops(
            frame,
            self.plate_vehicle_detector.detect(frame),
            max_vehicles=self.settings.plate_max_vehicles,
            minimum_width=self.settings.plate_min_crop_width,
        )

    def _read_plate_video_frame(self, frame: np.ndarray) -> None:
        try:
            images = self._plate_video_images(frame)
        except Exception as error:
            self.state.update_plate({"estado": "error_lectura", "error": str(error)})
            return
        if not images:
            self.state.update_plate({"estado": "sin_vehiculo"})
            return
        self._read_plate(images)

    def _run_plate_video(self) -> None:
        """Captura un flujo independiente para transmitirlo y leer sus placas."""
        source = self.settings.plate_video_source
        capture: cv2.VideoCapture | None = None
        interval = 0.0
        frame_number = 0
        next_frame_at = 0.0
        self.state.update_source({"placas_origen": source, "placas_error": None})

        while self.running:
            try:
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    capture = self._open_capture(source)
                    if Path(source).is_file():
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        interval = 1.0 / fps if fps and 1 <= fps <= 120 else 1.0 / 25
                    else:
                        interval = 0.0
                    next_frame_at = 0.0
                    self.state.update_source({"placas_error": None})

                if interval > 0:
                    now = time.monotonic()
                    if next_frame_at == 0.0 or now - next_frame_at > 1.0:
                        next_frame_at = now
                    wait = next_frame_at - now
                    if wait > 0:
                        time.sleep(wait)
                    next_frame_at += interval

                ok, frame = capture.read()
                if not ok and Path(source).is_file():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("La fuente de placas dejó de entregar cuadros")

                self._publish_plate_frame(frame)
                frame_number += 1
                if (
                    self.plates_enabled
                    and frame_number % max(self.settings.plate_interval_frames, 1) == 0
                    and (self.plate_thread is None or not self.plate_thread.is_alive())
                ):
                    self.plate_thread = threading.Thread(
                        target=self._read_plate_video_frame,
                        args=(frame.copy(),),
                        name="plate-reader",
                        daemon=True,
                    )
                    self.plate_thread.start()
            except Exception as error:
                self.state.update_source({"placas_error": str(error)})
                log.exception("Falló el flujo de video de placas")
                if capture is not None:
                    capture.release()
                    capture = None
                time.sleep(self.settings.video_retry_seconds)

        if capture is not None:
            capture.release()

    def _count_video_frame(self) -> None:
        self.video_frames += 1
        now = time.monotonic()
        elapsed = now - self.fps_window_start
        if elapsed >= 2.0:
            self.state.update_performance({"fps_video": round(self.video_frames / elapsed, 1)})
            self.video_frames = 0
            self.fps_window_start = now

    def _run(self) -> None:
        """Hilo de video: lee cuadros y los publica sin esperar a YOLO."""
        capture: cv2.VideoCapture | None = None
        interval = 0.0
        while self.running:
            try:
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    capture = self._open_source()
                    interval = self._frame_interval(capture)
                    self.state.worker_error = None

                switched = self._maybe_switch_to_primary(capture)
                if switched is not capture:
                    capture = switched
                    interval = self._frame_interval(capture)

                self._pace(interval)
                ok, frame = capture.read()
                if not ok and self._source_is_file():
                    # Un video de prueba terminó: se repite desde el inicio.
                    self.loop_count += 1
                    log.info("El video terminó; se repite desde el inicio (vuelta %d)", self.loop_count)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok:
                    capture.release()
                    capture = None
                    self.state.worker_error = "La fuente dejó de entregar cuadros"
                    time.sleep(self.settings.video_retry_seconds)
                    continue

                self._publish_frame(frame)
            except Exception as error:
                self.state.worker_error = str(error)
                if capture is not None:
                    capture.release()
                    capture = None
                time.sleep(self.settings.video_retry_seconds)

        if capture is not None:
            capture.release()

    def _detect_loop(self) -> None:
        """Hilo de detección: analiza siempre el cuadro más reciente.

        Si YOLO es más lento que el video, los cuadros intermedios se descartan
        en vez de acumularse; así el análisis nunca se atrasa respecto a la realidad.
        """
        last_id = 0
        while self.running:
            if not self._detection_needed():
                time.sleep(0.2)
                continue
            with self.frame_lock:
                frame, frame_id = self.latest_frame, self.latest_frame_id
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                continue
            last_id = frame_id
            try:
                self._process_detection(frame)
            except Exception as error:
                self.state.worker_error = str(error)
                self.detector_failed = True
                time.sleep(self.settings.video_retry_seconds)

    def _process_detection(self, frame: np.ndarray) -> None:
        started = time.perf_counter()
        self.frame_number += 1
        reserved = self.reserved_cells()
        parking_state, vehicles = self.parking.analyze(frame, reserved)
        self._update_reservations(parking_state)
        with self.frame_lock:
            self.overlay = (vehicles, parking_state, reserved)

        vehicle_detected = parking_state["vehiculos_detectados"] > 0
        if self.plates_enabled and not self.settings.plate_video_source and self._plate_attempt_allowed(vehicle_detected) and (
            self.plate_thread is None or not self.plate_thread.is_alive()
        ):
            images = self._plate_images(frame)
            if images:
                self.plate_thread = threading.Thread(
                    target=self._read_plate, args=(images,), name="plate-reader", daemon=True
                )
                self.plate_thread.start()
        self.vehicle_was_detected = vehicle_detected

        if self.spaces_enabled:
            self.state.update(parking_state)
        if self.detector_failed:
            self.detector_failed = False
            self.state.worker_error = None
        seconds = time.perf_counter() - started
        self.analysis_seconds = seconds if not self.analysis_seconds else 0.8 * self.analysis_seconds + 0.2 * seconds
        self.state.update_performance({"fps_analisis": round(1 / self.analysis_seconds, 1)})

    def snapshot_jpeg(self) -> bytes | None:
        """El cuadro más reciente tal como lo entrega la fuente, sin cajas ni celdas dibujadas."""
        with self.frame_lock:
            frame = self.latest_frame
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return encoded.tobytes() if ok else None

    def health(self) -> dict[str, str | bool]:
        return {
            "ok": self.state.worker_error is None,
            "error": self.state.worker_error or "",
            "fuente": "respaldo" if self.using_fallback else "principal",
        }
