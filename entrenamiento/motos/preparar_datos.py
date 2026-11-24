"""Arma el conjunto de entrenamiento para ajustar YOLO a la cámara de Archivos (motos del fondo).

1. Toma cuadros del video SOLO del período de entrenamiento (por defecto la mañana, hasta el cuadro
   `--corte`) y lejos (±`--margen` cuadros) de las muestras de la evaluación, para que la evaluación
   siga siendo independiente.
2. De cada cuadro guarda dos imágenes, las mismas dos vistas que analiza el servicio: el cuadro
   completo y la zona de las celdas lejanas ampliada al doble (`far_zone`).
3. Las etiqueta un modelo "profesor" grande (`yolo11x`), mucho mejor con objetos chicos pero muy lento
   para correr en vivo. Se guardan en formato YOLO con los números de clase de COCO (0 persona,
   1 bicicleta, 2 carro, 3 moto, 5 bus, 7 camión), así el modelo ajustado sigue usando los mismos.

Uso: python entrenamiento/motos/preparar_datos.py --profesor modelos/descargas/yolo11x.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))

from app.parking_detector import far_zone  # noqa: E402

AQUI = Path(__file__).resolve().parent
CLASES = {0: 0.30, 1: 0.25, 2: 0.30, 3: 0.25, 5: 0.35, 7: 0.35}  # clase COCO -> confianza mínima del profesor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=RAIZ / "videos" / "parqueadero-18nov-dia-completo.mp4")
    p.add_argument("--celdas", type=Path, default=RAIZ / "evaluacion" / "ocupacion" / "celdas.json")
    p.add_argument("--evaluacion", type=Path, default=RAIZ / "evaluacion" / "ocupacion" / "predicciones.json")
    p.add_argument("--profesor", type=Path, required=True)
    p.add_argument("--cuadros", type=int, default=300)
    p.add_argument("--corte", type=int, default=278_000, help="Último cuadro del período de entrenamiento")
    p.add_argument("--margen", type=int, default=9_000, help="Cuadros (±) que se evitan alrededor de cada muestra")
    p.add_argument("--salida", type=Path, default=AQUI / "datos")
    p.add_argument("--validacion", type=float, default=0.15)
    args = p.parse_args()

    from ultralytics import YOLO

    video = cv2.VideoCapture(str(args.video))
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    muestras = [m["cuadro"] for m in json.loads(args.evaluacion.read_text(encoding="utf-8")).values()]
    permitidos = [c for c in range(0, min(args.corte, total), 150)
                  if all(abs(c - m) > args.margen for m in muestras)]
    elegidos = [permitidos[i] for i in np.linspace(0, len(permitidos) - 1, args.cuadros).astype(int)]

    celdas = json.loads(args.celdas.read_text(encoding="utf-8"))
    ancho, alto = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zona = far_zone([(np.array(c["points"]) * [ancho, alto]).astype(np.int32) for c in celdas], ancho, alto)
    print("zona lejana:", zona, "| cuadros:", len(elegidos), "de", len(permitidos), "permitidos", flush=True)

    profesor = YOLO(str(args.profesor))
    random.seed(7)
    validacion = set(random.sample(elegidos, int(len(elegidos) * args.validacion)))
    cuenta = {c: 0 for c in CLASES}
    for n, cuadro in enumerate(elegidos):
        video.set(cv2.CAP_PROP_POS_FRAMES, cuadro)
        ok, frame = video.read()
        if not ok:
            continue
        parte = "val" if cuadro in validacion else "train"
        vistas = [("c", frame)]
        if zona:
            x0, y0, x1, y1 = zona
            vistas.append(("z", cv2.resize(frame[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)))
        for sufijo, imagen in vistas:
            nombre = f"f{cuadro:06d}{sufijo}"
            (args.salida / "images" / parte).mkdir(parents=True, exist_ok=True)
            (args.salida / "labels" / parte).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.salida / "images" / parte / f"{nombre}.jpg"), imagen, [cv2.IMWRITE_JPEG_QUALITY, 95])
            resultado = profesor(imagen, imgsz=1280 if sufijo == "c" else 1600, conf=0.2,
                                 classes=list(CLASES), verbose=False)[0]
            h, w = imagen.shape[:2]
            lineas = []
            for caja in resultado.boxes:
                clase, conf = int(caja.cls[0]), float(caja.conf[0])
                if conf < CLASES[clase]:
                    continue
                x1, y1, x2, y2 = caja.xyxy[0].tolist()
                lineas.append(f"{clase} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                cuenta[clase] += 1
            (args.salida / "labels" / parte / f"{nombre}.txt").write_text("\n".join(lineas), encoding="utf-8")
        print(f"{n + 1}/{len(elegidos)} cuadro {cuadro} | etiquetas por clase: {cuenta}", flush=True)

    nombres = {i: n for i, n in enumerate(YOLO(str(RAIZ / "scripts" / "yolo11n.pt")).names.values())}
    yaml = ["path: " + str(args.salida.resolve()).replace("\\", "/"), "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in nombres.items()]
    (args.salida / "datos.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print("listo:", args.salida / "datos.yaml")


if __name__ == "__main__":
    main()
