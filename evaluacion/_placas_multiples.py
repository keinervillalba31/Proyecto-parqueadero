"""En los cuadros con >=4 vehículos: placas con formato válido que salen con el tope de 3 vs sin tope."""
import cv2, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.app.config import Settings
from scripts.app.parking_detector import VehicleDetector
from scripts.app.plate_reader import LocalPlateEngine, PlateReader, vehicle_crops_with_kinds
s = Settings(); R = Path(__file__).parent
det = VehicleDetector(s.model_path, s.yolo_confidence, s.yolo_image_size, 1, s.minimum_box_area, s.minimum_motorcycle_area)
eng = LocalPlateEngine(); pr = PlateReader("x", "y", 2, 30, local_engine=eng)
rows = json.load(open(R / "resultados" / "trafico_simultaneos.json"))
cap = cv2.VideoCapture("C:/Users/WPOSS/Downloads/trafico_dia1.mp4")
out = []
def placas(crops):
    ok = []
    for img, kind in crops:
        r = eng.read(img)
        ok += [p for p in (pr.validate(t, kind) for t in r.texts) if p]
    return ok
for r in [r for r in rows if r["total"] >= 4]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"]); _, im = cap.read()
    vs = det.detect(im)
    a = placas(vehicle_crops_with_kinds(im, vs, 3, s.plate_min_crop_width))
    b = placas(vehicle_crops_with_kinds(im, vs, 20, s.plate_min_crop_width))
    out.append(dict(frame=r["frame"], vehiculos=r["total"], motos=r["motos"], placas_tope3=len(a), placas_sin_tope=len(b)))
    print(out[-1], flush=True)
json.dump(out, open(R / "resultados" / "placas_multiples.json", "w"))
print("total tope3:", sum(o["placas_tope3"] for o in out), "sin tope:", sum(o["placas_sin_tope"] for o in out), "vehículos:", sum(o["vehiculos"] for o in out))
