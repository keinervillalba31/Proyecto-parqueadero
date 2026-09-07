from pathlib import Path
import json
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, box as ShapelyBox

# Rutas del proyecto
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent

imagen_path = project_dir / "imagenes" / "imagen2.jpg"
json_path = script_dir / "estacionamientos.json"

imagen = cv2.imread(str(imagen_path))
if imagen is None or not json_path.exists():
    raise FileNotFoundError("Revisa las rutas de la imagen o del archivo 'estacionamientos.json'.")

with open(json_path, "r", encoding="utf-8") as f:
    parqueaderos = json.load(f)

# Inicializar estructura de las celdas
estado_parqueaderos = {}
for nombre, pts in parqueaderos.items():
    pts_array = np.array(pts, np.int32)
    poly_shapely = Polygon(pts)
    estado_parqueaderos[nombre] = {
        "ocupado": False,
        "puntos_cv": pts_array,
        "poly_shapely": poly_shapely,
        "area_total": poly_shapely.area,
    }

# Cargar el modelo YOLO11 (Reemplaza la ruta si utilizas tu propio modelo fine-tuned: 'runs/.../best.pt')
model = YOLO("yolo11n.pt")
CLASES_VEHICULOS = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
NOMBRES_VEHICULOS = {
    2: "Carro",
    3: "Moto",
    5: "Bus",
    7: "Camion",
}

# Inferencia
results = model(imagen, conf=0.15, verbose=False)[0]

# Parámetros geométricos
UMBRAL_COBERTURA = 0.20  # La caja debe cubrir al menos el 20% de la celda
AREA_MINIMA_CAJA = 2000  # Evita falsos positivos pequeños

for b in results.boxes:
    cls_id = int(b.cls[0])
    if cls_id in CLASES_VEHICULOS:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        ancho = x2 - x1
        alto = y2 - y1
        area_box = ancho * alto

        if area_box < AREA_MINIMA_CAJA:
            continue

        box_poly = ShapelyBox(x1, y1, x2, y2)

        # Dibujar elementos de depuración
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (255, 255, 0), 1)
        centroide_x = int((x1 + x2) / 2)
        centroide_y = int((y1 + y2) / 2)
        cv2.circle(imagen, (centroide_x, centroide_y), 4, (255, 0, 255), -1)
        cv2.putText(
            imagen,
            NOMBRES_VEHICULOS[cls_id],
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Evaluación espacial contra cada celda
        for nombre, datos in estado_parqueaderos.items():
            cell_poly = datos["poly_shapely"]

            # Criterio 1: El centroide de la detección cae dentro de la celda
            if cell_poly.contains(box_poly.centroid):
                datos["ocupado"] = True
                continue

            # Criterio 2: Porcentaje de intersección sobre la celda
            if cell_poly.intersects(box_poly):
                area_interseccion = cell_poly.intersection(box_poly).area
                porcentaje_cobertura = area_interseccion / datos["area_total"]

                if porcentaje_cobertura >= UMBRAL_COBERTURA:
                    datos["ocupado"] = True

# Dibujar representación visual final
for nombre, datos in estado_parqueaderos.items():
    pts = datos["puntos_cv"]
    es_ocupado = datos["ocupado"]

    color = (0, 0, 255) if es_ocupado else (0, 255, 0)  # BGR: Rojo / Verde
    texto = "Ocupado" if es_ocupado else "Libre"

    cv2.polylines(imagen, [pts], isClosed=True, color=color, thickness=2)

    x_txt, y_txt = pts[0][0], pts[0][1] - 8
    cv2.putText(
        imagen,
        f"{nombre}: {texto}",
        (x_txt, y_txt),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )

cv2.imshow("Sistema de Monitoreo - YOLO11", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()