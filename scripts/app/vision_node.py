"""Nodo de visión: un solo proceso que atiende varias cámaras con un solo modelo.

Con NODE_CODE, el nodo le pregunta a la plataforma qué cámaras le tocan (y cada cuánto vuelve
a preguntar), arranca un VideoService por cámara, las para o las rearma cuando cambian, aplica
los interruptores de la plataforma y le reporta el estado de cada una. Con NODE_CAMERAS_FILE
hace lo mismo con un archivo, sin plataforma (para simular varias cámaras con videos).

Sin ninguno de los dos, es una sola cámara configurada en el .env, como siempre.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .backend_client import BackendClient, BackendError
from .config import Settings, resolve_source
from .parking_detector import SharedModel, best_device
from .video_service import ROLE_PLATES, ROLE_SPACES, VideoService

log = logging.getLogger("parqueadero")

# Cada cuánto se analiza un cuadro si la cámara no dice otra cosa: la ocupación cambia
# despacio (la histéresis ya pide segundos) y una placa pasa rápido frente a la cámara.
DEFAULT_ANALYSIS_SECONDS = {"OCCUPANCY": 3.0, "ENTRY": 0.5, "MIXED": 1.0}
# El video en vivo de cada cámara del nodo no necesita más que esto para verse fluido.
DEFAULT_STREAM_FPS = 10.0


@dataclass(frozen=True)
class CameraConfig:
    """Una cámara tal como la describe la plataforma (o el archivo de simulación)."""

    id: int
    name: str
    role: str
    source: str
    parking_id: int | None = None
    enabled: bool = True
    analysis_seconds: float | None = None
    # Recortes por lado al analizar (2 = 4 recortes además del cuadro); None = según el equipo.
    tile_grid: int | None = None

    @staticmethod
    def parse(raw: dict[str, Any]) -> "CameraConfig | None":
        source = str(raw.get("sourceUrl") or raw.get("source") or "").strip()
        try:
            camera_id = int(raw["id"])
        except (KeyError, TypeError, ValueError):
            return None
        if not source:
            return None
        role = str(raw.get("type") or "OCCUPANCY").upper()
        seconds = raw.get("analysisSeconds")
        tiles = raw.get("tileGrid")
        return CameraConfig(
            id=camera_id,
            name=str(raw.get("name") or f"Cámara {camera_id}"),
            role=role if role in ROLE_SPACES | ROLE_PLATES else "OCCUPANCY",
            source=resolve_source(source),
            parking_id=int(raw["parkingId"]) if raw.get("parkingId") is not None else None,
            enabled=bool(raw.get("detectionEnabled", True)),
            analysis_seconds=float(seconds) if isinstance(seconds, (int, float)) and seconds >= 0 else None,
            tile_grid=int(tiles) if isinstance(tiles, int) and 1 <= tiles <= 4 else None,
        )

    def restart_key(self) -> tuple[Any, ...]:
        """Lo que, si cambia, obliga a rearmar la cámara (lo demás se aplica en caliente)."""
        return (self.role, self.source, self.parking_id, self.analysis_seconds, self.tile_grid)


class VisionNode:
    def __init__(
        self,
        settings: Settings,
        *,
        backend: BackendClient | None = None,
        model_factory: Callable[[Path], SharedModel] = SharedModel,
        worker_factory: Callable[..., VideoService] = VideoService,
    ) -> None:
        self.settings = settings
        self.mode = "nodo" if settings.node_code or settings.node_cameras_file else "unica"
        if backend is not None:
            self.backend = backend
        else:
            self.backend = BackendClient(settings) if settings.backend_sync_enabled else None
        self._model_factory = model_factory
        self._worker_factory = worker_factory
        self.model: SharedModel | None = None
        self.workers: dict[int, VideoService] = {}
        self.cameras: dict[int, CameraConfig] = {}
        # Interruptores generales de la plataforma; cada cámara además puede estar apagada.
        self.services = {"espacios": True, "placas": True}
        self._lock = threading.RLock()
        self.running = False
        self.last_sync_error: str | None = None
        self.last_sync_at: float | None = None
        self._threads: list[threading.Thread] = []

    # -- ciclo de vida ------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        if self.mode == "unica":
            # Una sola cámara, configurada en el .env: el servicio de siempre.
            worker = self._worker_factory(self.settings)
            self.workers[self.settings.backend_camera_id or 0] = worker
            worker.start()
            return
        log.info("Nodo de visión %s: buscando sus cámaras", self.settings.node_code or "(simulación)")
        self.sync()
        self._spawn(self._config_loop, "node-config")
        if self.backend and self.settings.node_code:
            self._spawn(self._heartbeat_loop, "node-heartbeat")

    def stop(self) -> None:
        self.running = False
        with self._lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            worker.stop()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2)

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _sleep(self, seconds: float) -> None:
        waited = 0.0
        while self.running and waited < seconds:
            time.sleep(0.5)
            waited += 0.5

    # -- configuración ------------------------------------------------------

    def _desired(self) -> tuple[list[CameraConfig], dict[str, bool]]:
        """Las cámaras que le tocan al nodo y los interruptores generales."""
        if self.settings.node_cameras_file is not None:
            raw = json.loads(Path(self.settings.node_cameras_file).read_text(encoding="utf-8"))
            cameras = raw.get("cameras", raw) if isinstance(raw, dict) else raw
            services = raw.get("services", {}) if isinstance(raw, dict) else {}
        else:
            if self.backend is None:
                raise BackendError("NODE_CODE necesita BACKEND_SYNC_ENABLED=true para hablar con la plataforma")
            config = self.backend.get_node_config(self.settings.node_code)
            cameras = config.get("cameras", [])
            services = config.get("services", {})
        parsed = [camera for camera in (CameraConfig.parse(item) for item in cameras) if camera is not None]
        return parsed, {
            "espacios": bool(services.get("spacesEnabled", True)),
            "placas": bool(services.get("platesEnabled", True)),
        }

    def sync(self) -> bool:
        """Pone el nodo como dice la configuración. True si se pudo consultar."""
        try:
            desired, services = self._desired()
        except (BackendError, OSError, ValueError) as error:
            if self.last_sync_error != str(error):
                log.warning("No se pudo leer la configuración del nodo: %s", error)
            self.last_sync_error = str(error)
            return False
        self.last_sync_error = None
        self.last_sync_at = time.time()
        with self._lock:
            self.services = services
            wanted = {camera.id: camera for camera in desired}
            for camera_id in list(self.workers):
                current = self.cameras.get(camera_id)
                camera = wanted.get(camera_id)
                if camera is None or current is None or camera.restart_key() != current.restart_key():
                    self._remove(camera_id, "se quitó del nodo" if camera is None else "cambió su configuración")
            for camera in desired:
                if camera.id not in self.workers:
                    self._add(camera)
                self.cameras[camera.id] = camera
            for camera_id in list(self.cameras):
                if camera_id not in wanted:
                    del self.cameras[camera_id]
            self._apply_services()
        return True

    def _camera_settings(self, camera: CameraConfig) -> Settings:
        folder = self.settings.node_data_dir / str(camera.id)
        folder.mkdir(parents=True, exist_ok=True)
        return replace(
            self.settings,
            video_source=camera.source,
            video_fallback="",
            plate_video_source="",
            backend_camera_id=camera.id,
            backend_parking_id=camera.parking_id,
            cells_path=folder / "celdas.json",
            cells_reference_path=folder / "referencia.jpg",
            services_state_path=folder / "servicios.json",
            yolo_tile_grid=self._tile_grid(camera),
            analysis_interval_seconds=camera.analysis_seconds
            if camera.analysis_seconds is not None
            else DEFAULT_ANALYSIS_SECONDS[camera.role],
            stream_max_fps=self.settings.stream_max_fps or DEFAULT_STREAM_FPS,
        )

    def _tile_grid(self, camera: CameraConfig) -> int:
        """Recortes al analizar. Cada recorte es otra pasada del modelo: con 2 por lado, cinco.

        Medido con 6 cámaras en un PC sin GPU: con recortes el modelo no alcanza (un análisis
        cada 6,4 s en vez de 3 y la cuenta oscila); sin ellos, cada 3 s con el modelo al 45 %
        y la misma ocupación. Por eso sin GPU no se recorta, salvo que la cámara lo pida.
        """
        if camera.tile_grid is not None:
            return camera.tile_grid
        if camera.role == "ENTRY":
            # En una entrada el vehículo está cerca y se ve grande.
            return 1
        device = self.model.device if self.model is not None else best_device()
        return self.settings.yolo_tile_grid if device == "cuda" else 1

    def _add(self, camera: CameraConfig) -> None:
        if self.model is None:
            self.model = self._model_factory(self.settings.model_path)
            log.info("Modelo cargado una vez para todas las cámaras (dispositivo: %s)", self.model.device)
        worker = self._worker_factory(
            self._camera_settings(camera),
            shared_model=self.model,
            backend=self.backend,
            role=camera.role,
            manage_services=False,
        )
        self.workers[camera.id] = worker
        if self.running:
            worker.start()
        log.info("Cámara %d (%s, %s) atendida por el nodo: %s", camera.id, camera.name, camera.role, camera.source)

    def _remove(self, camera_id: int, reason: str) -> None:
        worker = self.workers.pop(camera_id, None)
        if worker is not None:
            worker.stop()
            log.info("Cámara %d detenida: %s", camera_id, reason)

    def _apply_services(self) -> None:
        for camera_id, worker in self.workers.items():
            camera = self.cameras.get(camera_id)
            enabled = camera.enabled if camera else True
            worker._apply_services(
                spaces=self.services["espacios"] and enabled,
                plates=self.services["placas"] and enabled,
            )

    def _config_loop(self) -> None:
        while self.running:
            self._sleep(self.settings.node_config_poll_seconds)
            if self.running:
                self.sync()

    # -- estado -------------------------------------------------------------

    def public_url(self) -> str:
        if self.settings.node_public_url:
            return self.settings.node_public_url
        host = self.settings.service_host
        host = "localhost" if host in ("0.0.0.0", "::", "") else host
        return f"http://{host}:{self.settings.service_port}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            workers = dict(self.workers)
            cameras = dict(self.cameras)
        model = self.model
        return {
            "modo": self.mode,
            "codigo": self.settings.node_code or None,
            "url": self.public_url(),
            "dispositivo": model.device if model else best_device(),
            "modelo": {
                "inferencias": model.calls if model else 0,
                "segundos_ocupado": round(model.busy_seconds, 1) if model else 0.0,
            },
            "servicios": dict(self.services),
            "ultima_sincronizacion": self.last_sync_at,
            "error_sincronizacion": self.last_sync_error,
            "camaras": [
                {**worker.summary(), "nombre": cameras[camera_id].name if camera_id in cameras else None}
                for camera_id, worker in workers.items()
            ],
        }

    def _heartbeat_loop(self) -> None:
        while self.running:
            status = self.status()
            try:
                self.backend.node_heartbeat(
                    self.settings.node_code,
                    {
                        "baseUrl": status["url"],
                        "device": status["dispositivo"],
                        "cameras": [
                            {
                                "id": camera["id"],
                                "online": bool(camera["en_linea"]),
                                "error": camera["error"],
                                "fpsVideo": camera["fps_video"],
                                "fpsAnalysis": camera["fps_analisis"],
                            }
                            for camera in status["camaras"]
                            if camera["id"] is not None
                        ],
                    },
                )
            except BackendError as error:
                log.warning("No se pudo reportar el estado del nodo: %s", error)
            self._sleep(self.settings.node_heartbeat_seconds)

    # -- acceso a cada cámara -----------------------------------------------

    def worker(self, camera_id: int) -> VideoService | None:
        with self._lock:
            return self.workers.get(camera_id)

    def default_worker(self) -> VideoService | None:
        """La cámara de las rutas sin número (/video, /snapshot...): la única, o la primera de ocupación."""
        with self._lock:
            if not self.workers:
                return None
            for worker in self.workers.values():
                if worker.role in ROLE_SPACES:
                    return worker
            return next(iter(self.workers.values()))
