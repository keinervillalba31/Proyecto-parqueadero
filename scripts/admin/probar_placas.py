"""Prueba el lector de placas de Roboflow con un video o una imagen.

Ejemplos, desde `Proyecto-parqueadero`:

    python scripts/admin/probar_placas.py videos/video-placa-prueba.mp4 --cada 5 --max-lecturas 40 --guardar salida_placas
    python scripts/admin/probar_placas.py imagenes/imagen1.jpg
    python scripts/admin/probar_placas.py videos/video-placa-prueba.mp4 --cuadro-completo
    python scripts/admin/probar_placas.py videos/video-placa-prueba.mp4 --patron ""

Por defecto se recortan los vehículos con YOLO y se envía cada recorte a
Roboflow. Cada recorte enviado es una llamada a la API, por eso se limita con
`--max-lecturas`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.plate_reader import PlateReader, vehicle_crops  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_vehicle_detector():
    from app.parking_detector import VehicleDetector

    return VehicleDetector(
        model_path=settings.model_path,
        confidence=settings.yolo_confidence,
        image_size=settings.yolo_image_size,
        tile_grid=settings.yolo_tile_grid,
        minimum_box_area=settings.minimum_box_area,
        minimum_motorcycle_area=settings.minimum_motorcycle_area,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fuente", help="Ruta a un video o a una imagen")
    parser.add_argument("--cada", type=int, default=15, help="Leer uno de cada N cuadros")
    parser.add_argument("--max-lecturas", type=int, default=20, help="Máximo de llamadas a la API")
    parser.add_argument("--guardar", type=Path, help="Carpeta donde guardar las imágenes enviadas")
    parser.add_argument(
        "--cuadro-completo",
        action="store_true",
        help="Enviar el cuadro completo en vez de recortar los vehículos",
    )
    parser.add_argument(
        "--patron",
        default=settings.plate_pattern,
        help='Expresión regular del formato de placa ("" = aceptar todo). '
        "Por defecto PLATE_PATTERN o el formato colombiano",
    )
    args = parser.parse_args()

    reader = PlateReader(
        workspace=settings.roboflow_workspace,
        workflow=settings.roboflow_workflow,
        confirmation_reads=settings.plate_confirmation_reads,
        cooldown_seconds=settings.plate_cooldown_seconds,
        plate_pattern=args.patron,
    )
    detector = None if args.cuadro_completo else build_vehicle_detector()
    if args.guardar:
        args.guardar.mkdir(parents=True, exist_ok=True)

    def images_for(frame: np.ndarray) -> list[np.ndarray]:
        if detector is None:
            return [frame]
        return vehicle_crops(
            frame,
            detector.detect(frame),
            max_vehicles=settings.plate_max_vehicles,
            minimum_width=settings.plate_min_crop_width,
        )

    source = Path(args.fuente)
    if source.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(source))
        if frame is None:
            sys.exit(f"No se pudo abrir la imagen: {source}")
        frames = iter([(1, frame)])
        capture = None
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            sys.exit(f"No se pudo abrir el video: {source}")

        def video_frames():
            number = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                number += 1
                if number % args.cada == 0:
                    yield number, frame

        frames = video_frames()

    attempts = 0
    confirmed_plates: list[str] = []
    for frame_number, frame in frames:
        images = images_for(frame)
        if not images:
            print(f"Cuadro {frame_number}: ningún vehículo suficientemente grande")
            continue

        for index, image in enumerate(images, start=1):
            if attempts >= args.max_lecturas:
                break
            attempts += 1
            label = f"Cuadro {frame_number} vehículo {index}"
            try:
                texts = reader.read_texts(image)
            except Exception as error:
                print(f"{label}: error -> {error}")
                continue

            plates = reader.valid_plates(texts)
            rejected = [text for text in texts if reader.validate(text) is None]
            line = f"{label}: {', '.join(plates) or '(sin placa válida)'}"
            if rejected:
                line += f"  | descartado: {', '.join(rejected)}"
            for plate in plates:
                if reader.in_cooldown(plate):
                    continue
                confirmed = reader.confirm(plate)
                if confirmed:
                    line += f"  -> CONFIRMADA {confirmed[0]} ({confirmed[1]} lecturas)"
                    confirmed_plates.append(confirmed[0])
                    reader.start_cooldown(confirmed[0])
            print(line)

            if args.guardar:
                name = f"cuadro_{frame_number:06d}_v{index}_{plates[0] if plates else 'sin_placa'}.jpg"
                cv2.imwrite(str(args.guardar / name), image)

        if attempts >= args.max_lecturas:
            break

    if capture is not None:
        capture.release()
    print(f"\nLlamadas a la API: {attempts}")
    print(f"Placas confirmadas: {confirmed_plates or 'ninguna'}")


if __name__ == "__main__":
    main()
