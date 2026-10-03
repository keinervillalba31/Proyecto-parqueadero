from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from .plate_reader import DEFAULT_PLATE_PATTERN


SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / ".env")


def project_path(variable: str, default: Path) -> Path:
    """Lee una ruta del entorno; si es relativa, la toma desde la raíz del proyecto."""
    value = os.getenv(variable)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def resolve_source(value: str) -> str:
    """Índice de cámara, URL o ruta de archivo (relativa a la raíz del proyecto)."""
    if not value or value.isdigit() or "://" in value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_DIR / path)


def video_source_from_env(default: str) -> str:
    return resolve_source(os.getenv("VIDEO_SOURCE", default))


def video_fallback_from_env() -> str:
    return resolve_source(os.getenv("VIDEO_FALLBACK", ""))


@dataclass(frozen=True)
class Settings:
    model_path: Path = project_path("YOLO_MODEL", SCRIPT_DIR / "yolo11n.pt")
    cells_path: Path = project_path("PARKING_CELLS", SCRIPT_DIR / "estacionamientos.json")
    registered_vehicles_path: Path = project_path(
        "REGISTERED_VEHICLES", SCRIPT_DIR / "vehiculos_registrados.json"
    )
    video_source: str = video_source_from_env("0")
    yolo_confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.15"))
    yolo_image_size: int = int(os.getenv("YOLO_IMAGE_SIZE", "960"))
    yolo_tile_grid: int = int(os.getenv("YOLO_TILE_GRID", "2"))
    occupancy_frames: int = int(os.getenv("OCCUPANCY_FRAMES", "3"))
    reservation_timeout_seconds: float = float(
        os.getenv("RESERVATION_TIMEOUT_SECONDS", "600")
    )
    minimum_box_area: int = int(os.getenv("MIN_BOX_AREA", "500"))
    minimum_motorcycle_area: int = int(os.getenv("MIN_MOTORCYCLE_AREA", "100"))
    plate_interval_frames: int = int(os.getenv("PLATE_INTERVAL_FRAMES", "15"))
    plate_confirmation_reads: int = int(os.getenv("PLATE_CONFIRMATION_READS", "2"))
    plate_cooldown_seconds: float = float(os.getenv("PLATE_COOLDOWN_SECONDS", "30"))
    plate_require_vehicle: bool = os.getenv("PLATE_REQUIRE_VEHICLE", "true").lower() == "true"
    # Vacío = acepta cualquier texto. Por defecto, placas colombianas.
    plate_pattern: str = os.getenv("PLATE_PATTERN", DEFAULT_PLATE_PATTERN)
    plate_crop_vehicles: bool = os.getenv("PLATE_CROP_VEHICLES", "true").lower() == "true"
    plate_max_vehicles: int = int(os.getenv("PLATE_MAX_VEHICLES", "3"))
    plate_min_crop_width: int = int(os.getenv("PLATE_MIN_CROP_WIDTH", "120"))
    plate_skip_parked: bool = os.getenv("PLATE_SKIP_PARKED", "true").lower() == "true"
    video_retry_seconds: float = float(os.getenv("VIDEO_RETRY_SECONDS", "2"))
    # Video que se repite en bucle mientras la cámara principal no responde.
    video_fallback: str = video_fallback_from_env()
    # Cada cuántos segundos se reintenta la cámara principal mientras se usa el respaldo.
    video_camera_retry_seconds: float = float(os.getenv("VIDEO_CAMERA_RETRY_SECONDS", "10"))
    service_host: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    service_port: int = int(os.getenv("SERVICE_PORT", "8000"))
    roboflow_workspace: str = os.getenv("ROBOFLOW_WORKSPACE", "")
    roboflow_workflow: str = os.getenv("ROBOFLOW_WORKFLOW", "")

    # --- Sincronización con el backend (educore-backend) ---
    # Cuando una placa se confirma y está registrada, además de reservar la
    # celda localmente se le pide al backend que asigne y guarde el espacio.
    backend_sync_enabled: bool = os.getenv("BACKEND_SYNC_ENABLED", "false").lower() == "true"
    backend_base_url: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")
    # Cuenta de servicio: un usuario con rol que tenga ASSIGNMENTS_MANAGE y
    # PLATES_VIEW (el rol Operador ya los tiene).
    backend_service_user_code: str = os.getenv("BACKEND_SERVICE_USER_CODE", "")
    backend_service_identity_document: str = os.getenv("BACKEND_SERVICE_IDENTITY_DOCUMENT", "")
    backend_service_password: str = os.getenv("BACKEND_SERVICE_PASSWORD", "")
    # Id del parqueadero de esta cámara en el backend (opcional; vacío = cualquiera).
    backend_parking_id: int | None = (
        int(os.getenv("BACKEND_PARKING_ID")) if os.getenv("BACKEND_PARKING_ID") else None
    )
    backend_request_timeout_seconds: float = float(os.getenv("BACKEND_REQUEST_TIMEOUT_SECONDS", "5"))
    backend_students_cache_seconds: float = float(os.getenv("BACKEND_STUDENTS_CACHE_SECONDS", "300"))


settings = Settings()
