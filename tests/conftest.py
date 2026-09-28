import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# Las pruebas no cargan el modelo real: si ultralytics no está instalado se
# reemplaza por un módulo mínimo, y en cualquier caso YOLO se simula.
if "ultralytics" not in sys.modules:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        fake = types.ModuleType("ultralytics")
        fake.YOLO = lambda *args, **kwargs: None
        sys.modules["ultralytics"] = fake


# Dos celdas cuadradas de 100x100, una al lado de la otra.
CELLS = {
    "Celda_1": [[0, 0], [100, 0], [100, 100], [0, 100]],
    "Celda_2": [[100, 0], [200, 0], [200, 100], [100, 100]],
}


@pytest.fixture
def cells_path(tmp_path: Path) -> Path:
    path = tmp_path / "estacionamientos.json"
    path.write_text(json.dumps(CELLS), encoding="utf-8")
    return path


@pytest.fixture
def make_detector(cells_path, monkeypatch):
    from app import parking_detector

    monkeypatch.setattr(parking_detector, "YOLO", lambda *args, **kwargs: None)

    def factory(occupancy_frames: int = 1, tile_grid: int = 1):
        return parking_detector.ParkingDetector(
            cells_path=cells_path,
            model_path=Path("modelo_falso.pt"),
            confidence=0.15,
            image_size=640,
            tile_grid=tile_grid,
            minimum_box_area=500,
            minimum_motorcycle_area=100,
            occupancy_frames=occupancy_frames,
        )

    return factory
