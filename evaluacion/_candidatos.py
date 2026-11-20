"""Dibuja las detecciones numeradas para etiquetarlas a mano."""
import cv2, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.app.config import Settings
from scripts.app.parking_detector import VehicleDetector
s = Settings(); R = Path(__file__).parent
det = VehicleDetector(s.model_path, s.yolo_confidence, s.yolo_image_size, s.yolo_tile_grid, s.minimum_box_area, s.minimum_motorcycle_area)
res = {}
for p in sorted((R / "conjunto_prueba").glob("*.jpg")):
    im = cv2.imread(str(p)); vs = det.detect(im)
    res[p.name] = [[v.class_id, round(v.confidence, 3), list(v.box)] for v in vs]
    o = im.copy()
    for i, v in enumerate(vs):
        x1, y1, x2, y2 = v.box; cv2.rectangle(o, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(o, str(i), (x1 + 2, y1 + 16), 0, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(R / "_cand_" + "") if False else str(R / f"_cand_{p.stem}.png"), o)
json.dump(res, open(R / "_candidatos.json", "w"))
print({k: len(v) for k, v in res.items()})
