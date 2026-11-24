"""Evaluación de la ocupación por puesto (celda) sobre una grabación, con matriz de confusión.

Pasos:
1. `python evaluacion/evaluar_ocupacion.py correr` toma cuadros repartidos a lo largo del video y, para cada uno,
   corre el detector igual que el servicio (mismo modelo, mismos ajustes del .env, alineación con la referencia y
   suavizado en el tiempo) sobre los segundos anteriores, para que el suavizado actúe como en vivo. Guarda la
   predicción de cada celda (suavizada y la del cuadro solo) y las imágenes para etiquetar, con las celdas en un
   color neutro y SIN la predicción, para no sesgar a quien etiqueta.
2. Se etiqueta a mano `ocupacion/etiquetas.json`: por cuadro, la lista de celdas ocupadas; las dudosas van en
   "dudosas" y no cuentan.
3. `python evaluacion/evaluar_ocupacion.py medir` compara y escribe la matriz de confusión y las métricas, en total,
   por zona (fondo / medio / frente) y por celda.

Ocupado = hay un vehículo estacionado dentro del puesto en ese cuadro (uno que pasa por el pasillo no cuenta).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

from app.alignment import CellAligner  # noqa: E402
from app.config import settings  # noqa: E402
from app.parking_detector import ParkingDetector  # noqa: E402

SALIDA = RAIZ / "evaluacion" / "ocupacion"


def _celdas(path: Path) -> dict[str, dict]:
    """Celdas exportadas de la plataforma: [{label, points (0..1), row}]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["label"]: {"points": c["points"], "puesto": c["label"], "puesto_id": None} for c in data}


def _detector(referencia: Path | None, ajustes: dict | None = None) -> ParkingDetector:
    ajustes = ajustes or {}
    return ParkingDetector(
        cells_path=RAIZ / "no-existe.json",
        model_path=settings.model_path,
        confidence=settings.yolo_confidence,
        image_size=settings.yolo_image_size,
        tile_grid=settings.yolo_tile_grid,
        minimum_box_area=settings.minimum_box_area,
        minimum_motorcycle_area=settings.minimum_motorcycle_area,
        occupy_seconds=settings.occupancy_occupy_seconds,
        free_seconds=settings.occupancy_free_seconds,
        aligner=CellAligner(referencia) if referencia else None,
        zoom_far_cells=settings.zoom_far_cells,
        still_seconds=ajustes.get("quieto", settings.occupancy_still_seconds),
        free_seconds_near=ajustes.get("liberar_cerca", settings.occupancy_free_seconds_near),
        footpoint_lift=ajustes.get("subir", settings.footpoint_lift),
    )


def correr(args: argparse.Namespace) -> None:
    celdas = _celdas(args.celdas)
    video = cv2.VideoCapture(str(args.video))
    fps = video.get(cv2.CAP_PROP_FPS)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ventana = int(args.ventana * fps)
    paso = max(1, int(round(fps * args.cada)))
    inicio = ventana + 1
    muestras = np.linspace(inicio, total - 2, args.muestras).astype(int)
    SALIDA.mkdir(parents=True, exist_ok=True)
    resultados = {}
    detecciones = {}
    for n, objetivo in enumerate(muestras):
        vistos = []
        detector = _detector(args.referencia)
        detector.set_relative_cells(celdas)
        reloj = {"t": 0.0}
        detector._clock = lambda: reloj["t"]  # el tiempo del video, no el del equipo
        analizar = list(range(objetivo - ventana, objetivo, paso)) + [objetivo]
        posicion = analizar[0]
        video.set(cv2.CAP_PROP_POS_FRAMES, posicion)
        for idx in analizar:
            while posicion < idx:
                video.grab()
                posicion += 1
            ok, frame = video.read()
            posicion += 1
            if not ok:
                raise RuntimeError(f"No se pudo leer el cuadro {idx}")
            reloj["t"] = idx / fps
            estado, vehiculos = detector.analyze(frame)
            transformada = detector.aligner.last_transform if detector.aligner is not None else None
            vistos.append({
                "t": reloj["t"],
                "transformada": None if transformada is None else np.asarray(transformada).tolist(),
                "vehiculos": [list(map(int, v.box)) + [int(v.class_id), round(float(v.confidence), 3)]
                              for v in vehiculos],
            })
        # La decisión del último cuadro solo, sin suavizado: qué celdas tienen el apoyo de un vehículo.
        crudo = _crudo(detector, vehiculos)
        nombre = f"m{n:02d}"
        cv2.imwrite(str(SALIDA / f"{nombre}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        _etiquetar(frame, detector, SALIDA / f"{nombre}_etiquetar.jpg")
        resultados[nombre] = {
            "cuadro": int(objetivo),
            "segundo": round(objetivo / fps, 1),
            "suavizado": sorted(k for k, v in estado["celdas"].items() if v["ocupado"]),
            "cuadro_solo": sorted(k for k, v in crudo.items() if v),
            "vehiculos": [list(map(int, v.box)) + [int(v.class_id)] for v in detector.last_vehicles],
        }
        detecciones[nombre] = {"tamano": [frame.shape[1], frame.shape[0]], "analisis": vistos}
        print(f"{nombre} cuadro {objetivo}: {len(resultados[nombre]['suavizado'])} ocupadas", flush=True)
        (SALIDA / "predicciones.json").write_text(json.dumps(resultados, indent=1), encoding="utf-8")
        (SALIDA / "detecciones.json").write_text(json.dumps(detecciones), encoding="utf-8")


def repasar(args: argparse.Namespace) -> None:
    """Vuelve a decidir la ocupación con las detecciones guardadas por `correr` y otros ajustes.

    No corre YOLO: sirve para probar en segundos las reglas de ocupación (quieto, liberar, punto de
    apoyo). Reescribe predicciones.json y mide.
    """
    from app.parking_detector import Vehicle

    celdas = _celdas(args.celdas)
    detecciones = json.loads((SALIDA / "detecciones.json").read_text(encoding="utf-8"))
    predicciones = json.loads((SALIDA / "predicciones.json").read_text(encoding="utf-8"))
    ajustes = {k: v for k, v in (("quieto", args.quieto), ("liberar_cerca", args.liberar_cerca),
                                 ("subir", args.subir)) if v is not None}
    for nombre, muestra in detecciones.items():
        detector = _detector(None, ajustes)
        if args.liberar is not None:
            detector.free_seconds = args.liberar
        detector.set_relative_cells(celdas)
        reloj = {"t": 0.0}
        detector._clock = lambda: reloj["t"]
        ancho, alto = muestra["tamano"]
        detector._apply_cell_changes(ancho, alto)
        detector._update_zoom_zone(ancho, alto)
        for paso in muestra["analisis"]:
            reloj["t"] = paso["t"]
            transformada = np.array(paso["transformada"]) if paso["transformada"] is not None else None
            vehiculos = [Vehicle(v[4], v[5], tuple(v[:4])) for v in paso["vehiculos"]]
            crudo = detector._decide(vehiculos, transformada)
            estado = detector._smooth(crudo)
        solo = {name: False for name in detector.cells}
        for v in vehiculos:
            celda = detector.cell_of(detector.footpoint_of(v, transformada))
            if celda:
                solo[celda] = True
        predicciones[nombre]["suavizado"] = sorted(k for k, v in estado.items() if v)
        predicciones[nombre]["cuadro_solo"] = sorted(k for k, v in solo.items() if v)
    (SALIDA / "predicciones.json").write_text(json.dumps(predicciones, indent=1), encoding="utf-8")
    print("ajustes:", ajustes | ({"liberar": args.liberar} if args.liberar is not None else {}))
    medir(args)


def _crudo(detector: ParkingDetector, vehiculos) -> dict[str, bool]:
    to_reference = detector.aligner.last_transform if detector.aligner is not None else None
    ocupado = {name: False for name in detector.cells}
    for vehiculo in vehiculos:
        celda = detector.cell_of(detector.footpoint_of(vehiculo, to_reference))
        if celda:
            ocupado[celda] = True
    return ocupado


def _etiquetar(frame: np.ndarray, detector: ParkingDetector, destino: Path) -> None:
    """El cuadro con las celdas en un color neutro y su nombre, sin decir qué predijo el detector."""
    imagen = frame.copy()
    to_reference = detector.aligner.last_transform if detector.aligner is not None else None
    to_frame = np.linalg.inv(to_reference) if to_reference is not None else None
    for name, cell in detector.cells.items():
        points = cell["points"].astype(np.float32)
        if to_frame is not None:
            points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), to_frame).reshape(-1, 2)
        points = np.round(points).astype(np.int32)
        cv2.polylines(imagen, [points], True, (255, 255, 0), 1)
        centro = points.mean(axis=0).astype(int)
        cv2.putText(imagen, name.replace("AR-", ""), tuple(centro - [8, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3)
        cv2.putText(imagen, name.replace("AR-", ""), tuple(centro - [8, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite(str(destino), imagen, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _zona(fila: int | None, y: float) -> str:
    return "fondo" if y < 0.6 else "medio" if y < 0.72 else "frente"


def medir(args: argparse.Namespace) -> None:
    predicciones = json.loads((SALIDA / "predicciones.json").read_text(encoding="utf-8"))
    etiquetas = json.loads((SALIDA / "etiquetas.json").read_text(encoding="utf-8"))
    celdas = {c["label"]: c for c in json.loads(args.celdas.read_text(encoding="utf-8"))}
    zona = {name: _zona(c.get("row"), float(np.mean([p[1] for p in c["points"]]))) for name, c in celdas.items()}
    # Celdas de motos: las PM-* y los puestos de carro donde en la práctica se estacionan motos (--motos).
    motos = {n for n in celdas if n.startswith("PM-")} | {m for m in args.motos.split(",") if m}
    tipo = {name: "motos" if name in motos else "carros" for name in celdas}

    informe = {}
    for modo in ("suavizado", "cuadro_solo"):
        total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        por_zona = {z: dict(total) for z in ("fondo", "medio", "frente")}
        por_tipo = {t: dict(total) for t in ("carros", "motos")}
        por_celda = {name: dict(total) for name in celdas}
        errores = []
        for muestra, verdad in etiquetas.items():
            if muestra not in predicciones:
                continue
            pred = set(predicciones[muestra][modo])
            real = set(verdad["ocupadas"])
            dudosas = set(verdad.get("dudosas", []))
            for name in celdas:
                if name in dudosas:
                    continue
                clave = ("TP" if name in real else "FP") if name in pred else ("FN" if name in real else "TN")
                total[clave] += 1
                por_zona[zona[name]][clave] += 1
                por_tipo[tipo[name]][clave] += 1
                por_celda[name][clave] += 1
                if clave in ("FP", "FN"):
                    errores.append(f"{muestra} {name} {clave}")
        informe[modo] = {
            "total": total | _metricas(total),
            "por_zona": {z: m | _metricas(m) for z, m in por_zona.items()},
            "por_tipo": {t: m | _metricas(m) for t, m in por_tipo.items()},
            "por_celda": {n: m for n, m in por_celda.items() if m["FP"] or m["FN"]},
            "errores": errores,
        }
    (SALIDA / "resultado.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    for modo, datos in informe.items():
        t = datos["total"]
        print(f"\n== {modo}: TP {t['TP']} FP {t['FP']} FN {t['FN']} TN {t['TN']} | "
              f"precisión {t['precision']:.3f} recall {t['recall']:.3f} F1 {t['f1']:.3f} exactitud {t['exactitud']:.3f}")
        for grupo in ("por_zona", "por_tipo"):
            for z, m in datos[grupo].items():
                print(f"   {z:6s} TP {m['TP']:3d} FP {m['FP']:3d} FN {m['FN']:3d} TN {m['TN']:3d} "
                      f"precisión {m['precision']:.3f} recall {m['recall']:.3f}")
    _graficar(informe)


def _graficar(informe: dict) -> None:
    """Matrices de confusión (real en filas, predicho en columnas) del total, de carros y de motos."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datos = informe["suavizado"]
    paneles = [("Todas las celdas", datos["total"]), ("Puestos de carro", datos["por_tipo"]["carros"]),
               ("Puestos de moto", datos["por_tipo"]["motos"])]
    figura, ejes = plt.subplots(1, 3, figsize=(12, 4))
    for eje, (titulo, m) in zip(ejes, paneles):
        matriz = np.array([[m["TP"], m["FN"]], [m["FP"], m["TN"]]])
        eje.imshow(matriz / max(1, matriz.sum()), cmap="Blues", vmin=0, vmax=0.6)
        for (i, j), valor in np.ndenumerate(matriz):
            fila = matriz[i].sum()
            eje.text(j, i, f"{valor}\n({valor / max(1, fila):.0%})", ha="center", va="center", fontsize=12)
        eje.set_xticks([0, 1], ["Ocupado", "Libre"])
        eje.set_yticks([0, 1], ["Ocupado", "Libre"])
        eje.set_xlabel("Predicho por el sistema")
        eje.set_ylabel("Real (etiquetado a mano)")
        eje.set_title(f"{titulo}\nprecisión {m['precision']:.2f} · recall {m['recall']:.2f}")
    figura.suptitle("Ocupación por puesto · parqueadero Archivos · video del 18-nov (con suavizado)")
    figura.tight_layout()
    figura.savefig(SALIDA / "matriz_confusion.png", dpi=150)


def _metricas(m: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = m["TP"], m["FP"], m["FN"], m["TN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exactitud": (tp + tn) / max(1, tp + fp + fn + tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("accion", choices=("correr", "medir", "repasar"))
    parser.add_argument("--quieto", type=float, help="repasar: segundos quieto para ocupar")
    parser.add_argument("--liberar", type=float, help="repasar: segundos para liberar (celdas lejanas)")
    parser.add_argument("--liberar-cerca", type=float, help="repasar: segundos para liberar las celdas cercanas")
    parser.add_argument("--subir", type=float, help="repasar: fracción de la caja que se sube el punto de apoyo")
    parser.add_argument("--video", type=Path, default=RAIZ / "videos" / "parqueadero-18nov-dia-completo.mp4")
    parser.add_argument("--celdas", type=Path, default=SALIDA / "celdas.json",
                        help="Celdas de la cámara exportadas de la plataforma")
    parser.add_argument("--referencia", type=Path, default=RAIZ / "referencia_celdas.jpg")
    parser.add_argument("--muestras", type=int, default=16)
    parser.add_argument("--ventana", type=float, default=45.0, help="Segundos de video analizados antes de cada muestra")
    parser.add_argument("--cada", type=float, default=1.0, help="Segundos entre análisis dentro de la ventana")
    parser.add_argument("--motos", default="AR-38,AR-39",
                        help="Puestos de carro donde en la práctica se estacionan motos (cuentan como de moto)")
    args = parser.parse_args()
    {"correr": correr, "medir": medir, "repasar": repasar}[args.accion](args)


if __name__ == "__main__":
    main()
