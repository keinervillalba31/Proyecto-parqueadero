"""Propone las celdas de una cámara observando dónde se estacionan los vehículos.

Ejemplos, desde `Proyecto-parqueadero`:

    python scripts/admin/detectar_puestos.py videos/parking_simulacion.mp4
    python scripts/admin/detectar_puestos.py grabacion.mp4 --cada 30 --salida salida_puestos
    python scripts/admin/detectar_puestos.py videos/parking_simulacion.mp4 --imagen-referencia referencia_celdas.jpg --referencia celdas_camara_4.json
    python scripts/admin/detectar_puestos.py rtsp://usuario:clave@camara/stream --minutos 120 --cada-segundos 10

Se segmentan los vehículos (yolo11n-seg) en uno de cada N cuadros, se llevan al cuadro de
referencia para que un temblor de la cámara no los mueva, y con los que se quedan quietos se
arman las filas y los puestos (ver app/auto_cells.py). Sirve un timelapse: una foto cada
5-10 s durante unas horas con movimiento alcanza.

Salida, en `--salida`:
- celdas_propuestas.json: las celdas en el formato de la plataforma (puntos relativos 0..1),
  con `observada` (se vio un vehículo en ella) o deducida (hueco entre carros, fila de enfrente).
- celdas_propuestas.jpg: el cuadro de referencia con las celdas (verde observadas, magenta deducidas).
- observaciones.json (con --guardar-observaciones): las siluetas, para repetir sin el modelo
  con --desde-observaciones.

Con `--referencia` (celdas trazadas a mano en ese mismo cuadro) se mide qué tan bien quedó:
puestos encontrados, IoU y si la ocupación sale igual que con las celdas manuales.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.alignment import CellAligner  # noqa: E402
from app.auto_cells import Observation, evaluate, occupancy_agreement, propose, stationary_vehicles  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = SCRIPT_DIR / "yolo11n-seg.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]  # carro, moto, bus, camión (COCO)


def observe(args: argparse.Namespace) -> tuple[np.ndarray, list[list[Observation]]]:
    """Recorre la fuente, segmenta los vehículos y los lleva al cuadro de referencia."""
    from ultralytics import YOLO

    model = YOLO(str(args.modelo) if Path(args.modelo).exists() else Path(args.modelo).name)
    capture = cv2.VideoCapture(args.fuente)
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {args.fuente}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    every = args.cada or max(1, round(args.cada_segundos * fps))
    deadline = time.monotonic() + args.minutos * 60 if args.minutos else None

    reference = cv2.imread(args.imagen_referencia) if args.imagen_referencia else None
    aligner: CellAligner | None = None
    frames: list[list[Observation]] = []
    index, aligned = 0, 0
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok or (deadline is not None and time.monotonic() > deadline):
            break
        if index % every == 0:
            if reference is None:
                reference = frame.copy()
            if aligner is None:
                aligner = CellAligner()
                aligner.set_reference(reference)
            to_reference = aligner.transform(frame)
            aligned += to_reference is not None
            result = model(frame, conf=args.confianza, imgsz=args.tamano, verbose=False, classes=VEHICLE_CLASSES, retina_masks=True)[0]
            items = []
            if result.masks is not None:
                for polygon, class_id in zip(result.masks.xy, result.boxes.cls.numpy()):
                    if len(polygon) < 3:
                        continue
                    points = polygon.astype(np.float32).reshape(-1, 1, 2)
                    if to_reference is not None:
                        points = cv2.perspectiveTransform(points, to_reference)
                    items.append(Observation(int(class_id), points.reshape(-1, 2).astype(float)))
            frames.append(items)
            if len(frames) % 20 == 0:
                print(f"  {len(frames)} cuadros analizados...", flush=True)
            if args.max_cuadros and len(frames) >= args.max_cuadros:
                break
        index += 1
    capture.release()
    if reference is None:
        raise SystemExit("La fuente no entregó ningún cuadro")
    print(
        f"Observación: {len(frames)} cuadros en {time.perf_counter() - started:.0f} s "
        f"({aligned} alineados con la referencia)"
    )
    return reference, frames


def save_observations(path: Path, reference_path: Path, frames: list[list[Observation]]) -> None:
    data = {
        "referencia": str(reference_path),
        "cuadros": [[{"clase": o.class_id, "silueta": np.round(o.polygon, 1).tolist()} for o in frame] for frame in frames],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def load_observations(path: Path) -> tuple[np.ndarray, list[list[Observation]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    reference = cv2.imread(data["referencia"])
    if reference is None:
        raise SystemExit(f"No se encontró el cuadro de referencia {data['referencia']}")
    frames = [[Observation(o["clase"], np.array(o["silueta"], float)) for o in frame] for frame in data["cuadros"]]
    return reference, frames


def load_reference_cells(path: Path, width: int, height: int) -> list[np.ndarray]:
    """Celdas manuales en el formato de la plataforma: lista (o diccionario) con puntos relativos."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.values() if isinstance(data, dict) else data
    cells = []
    for item in items:
        points = np.array(item["points"] if isinstance(item, dict) else item, float)
        if points.max() <= 1.5:
            points = points * [width, height]
        cells.append(points)
    return cells


def draw(reference: np.ndarray, cells, vehicles, real: list[np.ndarray] | None) -> np.ndarray:
    image = reference.copy()
    for points in real or []:
        cv2.polylines(image, [np.round(points).astype(np.int32)], True, (255, 255, 255), 1)
    for vehicle in vehicles:
        contours, _ = cv2.findContours(vehicle.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 220, 255), 1)
    for number, cell in enumerate(cells, start=1):
        color = (0, 200, 0) if cell.observed else (200, 0, 200)
        corners = np.round(cell.corners).astype(np.int32)
        cv2.polylines(image, [corners], True, color, 2)
        cv2.putText(image, str(number), tuple(corners.mean(axis=0).astype(int) - [6, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fuente", nargs="?", help="Video, timelapse o cámara (rtsp://...)")
    parser.add_argument("--cada", type=int, help="Analizar uno de cada N cuadros (por defecto, uno cada --cada-segundos)")
    parser.add_argument("--cada-segundos", type=float, default=0.2, help="Segundos entre cuadros analizados (por defecto 0.2)")
    parser.add_argument("--minutos", type=float, help="Dejar de observar después de estos minutos (para cámaras en vivo)")
    parser.add_argument("--max-cuadros", type=int, help="Máximo de cuadros a analizar")
    parser.add_argument("--imagen-referencia", help="Cuadro al que se llevan las celdas (por defecto, el primero)")
    parser.add_argument("--referencia", type=Path, help="Celdas trazadas a mano (JSON de la plataforma) para medir el resultado")
    parser.add_argument("--salida", type=Path, default=Path("salida_puestos"), help="Carpeta de salida")
    parser.add_argument("--modelo", default=str(DEFAULT_MODEL), help="Modelo de segmentación de YOLO")
    parser.add_argument("--tamano", type=int, default=1600, help="Tamaño de imagen para el modelo (más grande ve carros más lejanos)")
    parser.add_argument("--confianza", type=float, default=0.15)
    parser.add_argument("--presencia", type=float, default=0.5, help="Fracción de cuadros en que debe verse quieto un vehículo")
    parser.add_argument("--guardar-observaciones", action="store_true", help="Guardar las siluetas para repetir sin el modelo")
    parser.add_argument("--desde-observaciones", type=Path, help="Usar siluetas guardadas en vez de analizar la fuente")
    args = parser.parse_args()
    if not args.fuente and not args.desde_observaciones:
        parser.error("Falta la fuente (o --desde-observaciones)")

    args.salida.mkdir(parents=True, exist_ok=True)
    if args.desde_observaciones:
        reference, frames = load_observations(args.desde_observaciones)
    else:
        reference, frames = observe(args)
        reference_path = args.salida / "referencia.jpg"
        cv2.imwrite(str(reference_path), reference)
        if args.guardar_observaciones:
            save_observations(args.salida / "observaciones.json", reference_path, frames)

    height, width = reference.shape[:2]
    vehicles = stationary_vehicles(frames, (height, width), min_presence=args.presencia)
    proposal = propose(vehicles, (height, width))
    cells = proposal.cells
    observed = sum(cell.observed for cell in cells)
    print(
        f"Vehículos quietos: {len(vehicles)} · filas: {len(proposal.rows)} · celdas propuestas: {len(cells)} "
        f"({observed} observadas, {len(cells) - observed} deducidas)"
    )

    output = [
        {
            "label": f"AUTO-{number:02d}",
            "points": np.round(cell.corners / [width, height], 5).tolist(),
            "type": cell.kind,
            "row": cell.row,
            "observada": cell.observed,
        }
        for number, cell in enumerate(cells, start=1)
    ]
    (args.salida / "celdas_propuestas.json").write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")

    real = load_reference_cells(args.referencia, width, height) if args.referencia else None
    cv2.imwrite(str(args.salida / "celdas_propuestas.jpg"), draw(reference, cells, vehicles, real))
    print(f"Guardado en {args.salida}/ (celdas_propuestas.json y .jpg)")

    if real is not None:
        result = evaluate(cells, real)
        agreement = occupancy_agreement(cells, real, frames, result["pairs"])
        print(
            f"\nContra {args.referencia.name}: {result['found']} de {result['reference']} puestos encontrados "
            f"({result['found'] / max(result['reference'], 1):.0%}), precisión {result['found'] / max(len(cells), 1):.0%}, "
            f"IoU medio {result['mean_iou']:.2f}"
        )
        print(
            f"  observadas correctas: {result['found_observed']}/{result['proposed_observed']} · "
            f"deducidas correctas: {result['found_deduced']}/{result['proposed_deduced']}"
        )
        print(
            f"  ocupación por cuadro: manual {agreement['occupied_real']:.1f} · propuesta {agreement['occupied_proposed']:.1f} "
            f"(error medio {agreement['count_error']:.1f}) · mismo estado por celda: {agreement['cell_agreement']:.0%}"
        )


if __name__ == "__main__":
    main()
