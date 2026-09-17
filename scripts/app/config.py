from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("YOLO_MODEL", SCRIPT_DIR / "yolo11n.pt"))
    cells_path: Path = Path(os.getenv("PARKING_CELLS", SCRIPT_DIR / "estacionamientos.json"))
    registered_vehicles_path: Path = Path(
        os.getenv("REGISTERED_VEHICLES", SCRIPT_DIR / "vehiculos_registrados.json")
    )
    video_source: str = os.getenv("VIDEO_SOURCE", "0")
    yolo_confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.15"))
    yolo_image_size: int = int(os.getenv("YOLO_IMAGE_SIZE", "960"))
    yolo_tile_grid: int = int(os.getenv("YOLO_TILE_GRID", "2"))
    coverage_threshold: float = float(os.getenv("COVERAGE_THRESHOLD", "0.20"))
    minimum_box_area: int = int(os.getenv("MIN_BOX_AREA", "500"))
    minimum_motorcycle_area: int = int(os.getenv("MIN_MOTORCYCLE_AREA", "100"))
    plate_interval_frames: int = int(os.getenv("PLATE_INTERVAL_FRAMES", "15"))
    plate_confirmation_reads: int = int(os.getenv("PLATE_CONFIRMATION_READS", "2"))
    plate_cooldown_seconds: float = float(os.getenv("PLATE_COOLDOWN_SECONDS", "30"))
    plate_require_vehicle: bool = os.getenv("PLATE_REQUIRE_VEHICLE", "true").lower() == "true"
    video_retry_seconds: float = float(os.getenv("VIDEO_RETRY_SECONDS", "2"))
    roboflow_workspace: str = os.getenv("ROBOFLOW_WORKSPACE", "")
    roboflow_workflow: str = os.getenv("ROBOFLOW_WORKFLOW", "")


settings = Settings()
