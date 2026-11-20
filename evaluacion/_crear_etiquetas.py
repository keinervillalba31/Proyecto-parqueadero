"""Congela las etiquetas revisadas a mano (ids de _candidatos.json) en etiquetas.json."""
import json
from pathlib import Path
R = Path(__file__).parent
cand = json.load(open(R / "_candidatos.json"))
# tp: detecciones confirmadas como vehículo; ignorar: zonas amontonadas donde no se puede etiquetar con seguridad;
# el resto de ids de la imagen es falso positivo. vacios: parqueadero sin vehículos (verdad: 0).
rev = {
    "sim_00000.jpg": dict(tp=[0,1,2,3,4,6,9,13,18,22,25,26,27,28], ignorar=[5,7,8,10,11,12,14,16,17,19,20,21,23,24]),
    "largo_00900.jpg": dict(tp=[0,1,2,3,4,6,7,8,9,12,13], ignorar=[5,10,11,14,15,16,17,18,19,22]),
    "img_imagen2.jpg": dict(tp=[0,1,2,4,5,6,7,8,9,10], ignorar=[3]),
    "img_imagen3.jpg": dict(tp=[0,3,4,5,6,9,13,18,22,23], ignorar=[1,2,7,8,10,11,12,14,15,16,17,19,20,21]),
}
out = {"imagenes": {}, "vacios": ["img_imagen5.jpg","img_imagen6.jpg","img_imagen7.jpg","img_imagen8.jpg","img_imagen9.jpg","img_imagen10.jpg"]}
for name, r in rev.items():
    d = cand[name]
    out["imagenes"][name] = {
        "vehiculos": [d[i][2] + [d[i][0]] for i in r["tp"]],
        "ignorar": [d[i][2] for i in r["ignorar"]],
        "extras_no_detectados": [],
    }
json.dump(out, open(R / "etiquetas.json", "w"), indent=1)
print({k: len(v["vehiculos"]) for k, v in out["imagenes"].items()})
