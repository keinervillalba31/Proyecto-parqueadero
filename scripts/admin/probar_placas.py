"""Prueba el lector de placas (local, Roboflow o híbrido) con un video o una imagen.

Ejemplos, desde `Proyecto-parqueadero`:

    python scripts/admin/probar_placas.py videos/video-placa.mp4 --placa-real YZA91G
    python scripts/admin/probar_placas.py videos/video-placa.mp4 --motor roboflow --max-lecturas 10
    python scripts/admin/probar_placas.py videos/video-placa-prueba.mp4 --cada 5 --guardar salida_placas
    python scripts/admin/probar_placas.py imagenes/imagen1.jpg
    python scripts/admin/probar_placas.py videos/video-placa-prueba.mp4 --cuadro-completo
    python scripts/admin/probar_placas.py videos/video-placa-prueba.mp4 --patron ""

Por defecto se recortan los vehículos con YOLO y se lee cada recorte con el motor de
PLATE_ENGINE (local, sin créditos, si no se dice otra cosa). Con Roboflow cada recorte es una
llamada a la API, por eso se limita con `--max-lecturas`.

Con `--placa-real` se mide la exactitud contra la placa que de verdad aparece en el video:
lecturas exactas, placas confirmadas correctas o erróneas y tiempo por lectura. Sirve para
evaluar el lector con grabaciones propias.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.plate_reader import ENGINES, PlateReader, vehicle_crops_with_kinds  # noqa: E402


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
    parser.add_argument(
        "--max-lecturas", type=int,
        help="Máximo de lecturas (por defecto 20 con Roboflow, para cuidar los créditos; sin límite con el motor local)",
    )
    parser.add_argument("--motor", choices=ENGINES, default=settings.plate_engine, help="Motor de lectura (PLATE_ENGINE)")
    parser.add_argument("--placa-real", help="Placa que de verdad aparece, para medir la exactitud")
    parser.add_argument(
        "--sin-tipo", action="store_true",
        help="No usar el tipo de vehículo (carro o moto) para elegir el formato de la placa",
    )
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
        engine=args.motor,
        local_detector=settings.plate_local_detector,
        local_ocr=settings.plate_local_ocr,
    )
    max_reads = args.max_lecturas if args.max_lecturas is not None else (20 if args.motor == "roboflow" else 0)
    truth = PlateReader.normalize(args.placa_real) if args.placa_real else None
    detector = None if args.cuadro_completo else build_vehicle_detector()
    if args.guardar:
        args.guardar.mkdir(parents=True, exist_ok=True)

    def images_for(frame: np.ndarray) -> list[tuple[np.ndarray, str | None]]:
        if detector is None:
            return [(frame, None)]
        return vehicle_crops_with_kinds(
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
    exact = with_plate = 0
    times: list[float] = []
    confirmed_plates: list[str] = []
    for frame_number, frame in frames:
        images = images_for(frame)
        if not images:
            print(f"Cuadro {frame_number}: ningún vehículo suficientemente grande")
            continue

        for index, (image, kind) in enumerate(images, start=1):
            if max_reads and attempts >= max_reads:
                break
            attempts += 1
            kind = None if args.sin_tipo else kind
            label = f"Cuadro {frame_number} vehículo {index}" + (f" ({kind})" if kind else "")
            started = time.perf_counter()
            try:
                texts, source = reader.read_candidates(image, kind)
            except Exception as error:
                print(f"{label}: error -> {error}")
                continue
            times.append((time.perf_counter() - started) * 1000)

            plates = reader.valid_plates(texts, kind)
            rejected = [text for text in texts if reader.validate(text, kind) is None]
            if truth and plates:
                with_plate += 1
                exact += truth in plates
            line = f"{label} [{source}]: {', '.join(plates) or '(sin placa válida)'}"
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

        if max_reads and attempts >= max_reads:
            break

    if capture is not None:
        capture.release()
    print(f"\nMotor: {args.motor} · lecturas: {attempts} · locales: {reader.stats['local']} · "
          f"consultas a Roboflow: {reader.stats['roboflow']}")
    if times:
        print(f"Tiempo por lectura: mediana {statistics.median(times):.0f} ms, máximo {max(times):.0f} ms")
    print(f"Placas confirmadas: {confirmed_plates or 'ninguna'}")
    if truth:
        wrong = [plate for plate in confirmed_plates if plate != truth]
        print(f"Exactitud contra {truth}: {exact} de {with_plate} lecturas con placa válida fueron exactas"
              + (f" ({exact / with_plate:.0%})" if with_plate else ""))
        print(f"Confirmación: {'correcta' if truth in confirmed_plates else 'no se confirmó'}"
              + (f"; confirmaciones erróneas: {', '.join(wrong)}" if wrong else "; sin confirmaciones erróneas"))


if __name__ == "__main__":
    main()
