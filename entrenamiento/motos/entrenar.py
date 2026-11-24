"""Ajusta yolo11n (el que corre en el servicio) con las imágenes de la cámara de Archivos.

Parte de los pesos de COCO y congela la columna (las primeras capas): así aprende las motos chicas y
los carros apilados de esta cámara sin olvidar lo general, y el entrenamiento cabe en CPU. El modelo
sigue siendo el nano, así que en vivo es igual de rápido. Las clases conservan los números de COCO.

Uso: python entrenamiento/motos/entrenar.py [--epocas 30] [--imgsz 960]   (si se corta: --reanudar)
El resultado queda en entrenamiento/motos/runs/<nombre>/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datos", type=Path, default=AQUI / "datos" / "datos.yaml")
    p.add_argument("--base", type=Path, default=RAIZ / "scripts" / "yolo11n.pt")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--congelar", type=int, default=10, help="Capas iniciales que no se entrenan")
    p.add_argument("--nombre", default="archivos")
    p.add_argument("--fraccion", type=float, default=1.0, help="Parte de los datos (para medir el tiempo)")
    p.add_argument("--reanudar", action="store_true", help="Sigue un entrenamiento interrumpido desde last.pt")
    args = p.parse_args()

    from ultralytics import YOLO

    if args.reanudar:
        YOLO(str(AQUI / "runs" / args.nombre / "weights" / "last.pt")).train(resume=True)
        return

    YOLO(str(args.base)).train(
        data=str(args.datos),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.lote,
        device="cpu",
        workers=4,
        freeze=args.congelar,
        fraction=args.fraccion,
        patience=10,
        project=str(AQUI / "runs"),
        name=args.nombre,
        exist_ok=True,
        seed=7,
        plots=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
