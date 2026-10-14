"""Genera los datos del prototipo 3D suelto (prototipos/datos_3d.js).

El cálculo del plano de planta está en scripts/app/parking_layout.py, que es el
mismo que usa el servicio para enviarle el plano a la plataforma. Este script
solo agrega lo que necesita el prototipo: la foto de la cámara enderezada para
usarla de piso.

Uso:  python prototipos/generar_parqueadero_3d.py prototipos/celdas_camara_4.json
donde el JSON es una lista de {label, points (0..1), number, type, status, zone}: las
celdas de la cámara (tabla camera_cells) con el número y estado de su puesto.
Después se abre prototipos/parqueadero-3d.html en el navegador.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from app.parking_layout import rectify  # noqa: E402

VIDEO = PROJECT_DIR / "videos" / "parking_simulacion.mp4"
OUTPUT = Path(__file__).resolve().parent / "datos_3d.js"
MARGIN_M = 9.0
PIXELS_PER_M = 22


def main() -> None:
    cells = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(VIDEO))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"No se pudo leer {VIDEO}")
    height, width = frame.shape[:2]

    quads = np.array([[[x * width, y * height] for x, y in cell["points"]] for cell in cells])
    h, rect = rectify(quads)

    x_min, x_max = rect[:, 0].min() - MARGIN_M, rect[:, 1].max() + MARGIN_M
    y_min, y_max = rect[:, 2].min() - MARGIN_M, rect[:, 3].max() + MARGIN_M
    to_pixels = np.array([[PIXELS_PER_M, 0, -x_min * PIXELS_PER_M], [0, PIXELS_PER_M, -y_min * PIXELS_PER_M], [0, 0, 1]])
    size = (int((x_max - x_min) * PIXELS_PER_M), int((y_max - y_min) * PIXELS_PER_M))
    # Lo que la cámara no alcanzó a ver se rellena con el verde del terreno de la escena.
    top_down = cv2.warpPerspective(frame, to_pixels @ h, size, flags=cv2.INTER_LINEAR, borderValue=(119, 163, 134))
    ok, encoded = cv2.imencode(".jpg", top_down, [cv2.IMWRITE_JPEG_QUALITY, 82])

    data = {
        "bounds": {"xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max},
        "texture": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode(),
        "cells": [
            {
                "number": cell["number"] or cell["label"],
                "type": cell["type"],
                "zone": cell["zone"],
                "occupied": cell["status"] == "OCCUPIED",
                "x0": round(float(r[0]), 2), "x1": round(float(r[1]), 2),
                "y0": round(float(r[2]), 2), "y1": round(float(r[3]), 2),
            }
            for cell, r in zip(cells, rect)
        ],
    }
    OUTPUT.write_text("window.PARQUEADERO = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Planta: {x_max - x_min:.0f} x {y_max - y_min:.0f} m, textura {size[0]}x{size[1]} px, {len(encoded) // 1024} KB")


if __name__ == "__main__":
    main()
