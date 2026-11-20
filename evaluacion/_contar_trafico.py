"""Cuenta vehículos simultáneos (como los ve el lector de placas) en trafico_dia1.mp4."""
import cv2, sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.app.config import Settings
from scripts.app.parking_detector import VehicleDetector
s = Settings()
det = VehicleDetector(s.model_path, s.yolo_confidence, s.yolo_image_size, 1, s.minimum_box_area, s.minimum_motorcycle_area)
cap = cv2.VideoCapture("C:/Users/WPOSS/Downloads/trafico_dia1.mp4")
n = int(cap.get(7)); rows = []
for f in range(0, n, 150):          # un cuadro cada 5 s
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    if not ok: continue
    vs = [v for v in det.detect(im) if v.box[2] - v.box[0] >= s.plate_min_crop_width]
    rows.append(dict(frame=f, total=len(vs), motos=sum(v.class_id == 3 for v in vs), anchos=sorted((v.box[2]-v.box[0] for v in vs), reverse=True)[:8]))
json.dump(rows, open(Path(__file__).parent / "resultados" / "trafico_simultaneos.json", "w"))
c = collections.Counter(r["total"] for r in rows)
print("cuadros", len(rows), "distribución de vehículos simultáneos:", dict(sorted(c.items())))
print(">3:", sum(r["total"] > 3 for r in rows), " >=5:", sum(r["total"] >= 5 for r in rows), " motos máx:", max(r["motos"] for r in rows))
print("top:", sorted(rows, key=lambda r: -r["total"])[:5])
