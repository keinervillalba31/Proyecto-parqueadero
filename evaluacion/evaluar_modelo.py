"""EP14-HU099: evalúa la precisión del detector de vehículos (YOLO) con imágenes etiquetadas.

Uso:  python evaluacion/evaluar_modelo.py
Salida: evaluacion/resultados/{metricas.json, fallos/*.jpg}

Método:
- etiquetas.json: cajas de vehículos revisadas a mano en `conjunto_prueba/` (las zonas con carros tan amontonados
  que no se pueden etiquetar con seguridad van como "ignorar": no cuentan ni a favor ni en contra) y
  parqueaderos vacíos (verdad: 0 vehículos).
- Un acierto (TP) es una detección con IoU >= 0.5 con una caja etiquetada de la misma familia (moto / no moto).
- Condiciones de iluminación: se simulan sobre las mismas imágenes (las cajas verdad no cambian).
"""
import cv2, json, sys
import numpy as np
from pathlib import Path

R = Path(__file__).parent
sys.path.insert(0, str(R.parent))
from scripts.app.config import Settings
from scripts.app.parking_detector import VehicleDetector

IOU_MIN = 0.5


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def contained(small, large):
    w = min(small[2], large[2]) - max(small[0], large[0]); h = min(small[3], large[3]) - max(small[1], large[1])
    return max(0, w) * max(0, h) / max((small[2]-small[0])*(small[3]-small[1]), 1)


def gamma(im, g):
    t = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(im, t)


def noise(im, sigma, seed=0):
    r = np.random.default_rng(seed)
    return np.clip(im.astype(np.float32) + r.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


CONDICIONES = {
    "original": lambda im: im,
    "sobreexpuesto (sol fuerte)": lambda im: np.clip(gamma(im, 0.6).astype(np.float32) * 1.15, 0, 255).astype(np.uint8),
    "atardecer (-50% brillo)": lambda im: (im.astype(np.float32) * 0.5).astype(np.uint8),
    "poca luz (-75% + ruido)": lambda im: noise((im.astype(np.float32) * 0.25).astype(np.uint8), 6),
    "noche (-88% + ruido)": lambda im: noise((im.astype(np.float32) * 0.12).astype(np.uint8), 5),
    "niebla (bajo contraste)": lambda im: cv2.addWeighted(im, 0.55, np.full_like(im, 200), 0.45, 0),
    "compresión/ruido": lambda im: cv2.imdecode(cv2.imencode(".jpg", noise(im, 10), [cv2.IMWRITE_JPEG_QUALITY, 25])[1], 1),
}


def es_moto(c): return c == 3


def evaluar_imagen(det, im, et):
    vs = det.detect(im)
    gt = [(g[:4], g[4]) for g in et["vehiculos"]]
    ign = [tuple(b) for b in et["ignorar"]]
    usados, tp, fp, ignoradas, fallos_fp = set(), 0, 0, 0, []
    for v in sorted(vs, key=lambda v: -v.confidence):
        best, bi = 0.0, None
        for i, (b, c) in enumerate(gt):
            if i in usados or es_moto(c) != es_moto(v.class_id): continue
            o = iou(v.box, b)
            if o > best: best, bi = o, i
        if bi is not None and best >= IOU_MIN:
            usados.add(bi); tp += 1
        elif any(iou(v.box, b) > 0.3 or contained(v.box, b) > 0.7 for b in ign):
            ignoradas += 1
        else:
            fp += 1; fallos_fp.append(v.box)
    fn_cajas = [gt[i][0] for i in range(len(gt)) if i not in usados]
    return tp, fp, len(fn_cajas), ignoradas, fallos_fp, fn_cajas


def main():
    s = Settings()
    det = VehicleDetector(s.model_path, s.yolo_confidence, s.yolo_image_size, s.yolo_tile_grid, s.minimum_box_area, s.minimum_motorcycle_area)
    E = json.load(open(R / "etiquetas.json"))
    out = R / "resultados"; (out / "fallos").mkdir(parents=True, exist_ok=True)
    res = {"configuracion": dict(modelo=s.model_path.name, confianza=s.yolo_confidence, imgsz=s.yolo_image_size, tiles=s.yolo_tile_grid, iou_minimo=IOU_MIN), "condiciones": {}}
    for cname, fn in CONDICIONES.items():
        TP = FP = FN = IGN = 0; fp_vacios = 0; por_imagen = {}
        for name, et in E["imagenes"].items():
            im = fn(cv2.imread(str(R / "conjunto_prueba" / name)))
            tp, fp, fn_, ign, fps, fns = evaluar_imagen(det, im, et)
            TP += tp; FP += fp; FN += fn_; IGN += ign
            por_imagen[name] = dict(tp=tp, fp=fp, fn=fn_)
            if fp or fn_:
                o = im.copy()
                for b in et["vehiculos"]: cv2.rectangle(o, tuple(b[:2]), tuple(b[2:4]), (0, 255, 0), 2)
                for b in fns: cv2.rectangle(o, tuple(b[:2]), tuple(b[2:4]), (0, 0, 255), 4)      # rojo: no detectado
                for b in fps: cv2.rectangle(o, tuple(b[:2]), tuple(b[2:4]), (0, 165, 255), 4)    # naranja: falso positivo
                cv2.imwrite(str(out / "fallos" / f"{cname.split(' ')[0]}_{name}"), o)
        vac = {}
        for name in E["vacios"]:
            im = fn(cv2.imread(str(R / "conjunto_prueba" / name)))
            n = len(det.detect(im)); vac[name] = n; fp_vacios += n
            if n:
                o = im.copy()
                for v in det.detect(im): cv2.rectangle(o, tuple(v.box[:2]), tuple(v.box[2:]), (0, 165, 255), 4)
                cv2.imwrite(str(out / "fallos" / f"{cname.split(' ')[0]}_{name}"), o)
        P = TP / max(TP + FP + fp_vacios, 1); Rc = TP / max(TP + FN, 1)
        res["condiciones"][cname] = dict(tp=TP, fp_en_etiquetadas=FP, fp_en_vacios=fp_vacios, fn=FN, ignoradas=IGN,
            precision=round(P, 3), recall=round(Rc, 3), f1=round(2*P*Rc/max(P+Rc, 1e-9), 3), por_imagen=por_imagen, vacios=vac)
        print(f"{cname:30s} TP={TP:3d} FP={FP+fp_vacios:3d} FN={FN:3d}  P={P:.3f} R={Rc:.3f}", flush=True)
    json.dump(res, open(out / "metricas.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
