"""Extrae fotogramas de prueba (espaciados) de los videos y copia las imágenes sueltas."""
import cv2, shutil
from pathlib import Path
R = Path(__file__).parent; out = R / "conjunto_prueba"
def frames(video, idxs, prefix):
    c = cv2.VideoCapture(str(R.parent / "videos" / video))
    for i in idxs:
        c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = c.read()
        if ok: cv2.imwrite(str(out / f"{prefix}_{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
frames("parking_simulacion.mp4", [0, 60, 120, 180, 235], "sim")
frames("video-largo-parqueadero.mp4", [100, 900, 1700, 2500, 3300], "largo")
for p in (R.parent / "imagenes").iterdir():
    if p.suffix.lower() in (".jpg", ".png"):
        im = cv2.imread(str(p)); cv2.imwrite(str(out / f"img_{p.name.split('.')[0]}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
