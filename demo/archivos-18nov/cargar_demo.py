"""Carga en el backend la cámara de la demo del parqueadero Archivos (video del 18-nov) con sus 42 celdas.

Uso (con el backend encendido y el plano 3D de Archivos ya cargado, es decir, con los puestos AR-01…AR-41 y ARD-01):

    python demo/archivos-18nov/cargar_demo.py

Pide el código, documento y contraseña de un administrador. Si la cámara ya existe (mismo nombre) solo le vuelve a
cargar las celdas. Al final imprime el id de la cámara: ese número va en BACKEND_CAMERA_ID del .env.
Variables opcionales: BACKEND_BASE_URL (por defecto http://localhost:8080) y DEMO_STREAM_URL (la dirección del video
del servicio de IA que abre el navegador, por defecto http://localhost:8000/video/parqueadero).
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import sys
from pathlib import Path

import requests

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
BASE = os.environ.get("BACKEND_BASE_URL", "http://localhost:8080").rstrip("/") + "/api"
STREAM_URL = os.environ.get("DEMO_STREAM_URL", "http://localhost:8000/video/parqueadero")


class Api:
    def __init__(self) -> None:
        self.sesion = requests.Session()

    def _csrf(self) -> str:
        return self.sesion.get(BASE + "/auth/csrf", timeout=10).json()["data"]

    def llamar(self, metodo: str, ruta: str, cuerpo=None):
        respuesta = self.sesion.request(metodo, BASE + ruta, json=cuerpo, headers={"X-XSRF-TOKEN": self._csrf()},
                                        timeout=30)
        try:
            datos = respuesta.json()
        except ValueError:
            datos = {"message": respuesta.text[:300]}
        if not respuesta.ok:
            sys.exit(f"Error {respuesta.status_code} en {metodo} {ruta}: {datos.get('message')}")
        return datos.get("data")


def main() -> None:
    demo = json.loads((AQUI / "celdas_archivos.json").read_text(encoding="utf-8"))
    camara = demo["camara"]

    api = Api()
    print(f"Backend: {BASE}")
    codigo = input("Código del administrador: ").strip()
    documento = input("Documento del administrador: ").strip()
    clave = getpass.getpass("Contraseña: ")
    api.llamar("POST", "/auth/login", {"userCode": codigo, "identityDocument": documento, "password": clave})

    parqueadero = next((p for p in api.llamar("GET", "/parking") if p["name"] == camara["parqueadero"]), None)
    if parqueadero is None:
        sys.exit(f"No existe el parqueadero «{camara['parqueadero']}» en la plataforma.")

    puestos = {p["number"]: p["id"] for p in api.llamar("GET", "/parking-spaces")}
    faltan = [c["puesto"] for c in demo["celdas"] if c["puesto"] and c["puesto"] not in puestos]
    if faltan:
        sys.exit("Faltan estos puestos en la plataforma (carga el plano 3D de Archivos con «crear puestos»): "
                 + ", ".join(faltan))

    existente = next((c for c in api.llamar("GET", "/monitoring/cameras") if c["name"] == camara["nombre"]), None)
    if existente:
        id_camara = existente["id"]
        print(f"La cámara ya existe (id {id_camara}): se reemplazan sus celdas.")
    else:
        id_camara = api.llamar("POST", "/monitoring/cameras", {
            "name": camara["nombre"], "location": camara["ubicacion"], "streamUrl": STREAM_URL,
            "type": "OCCUPANCY", "parkingId": parqueadero["id"], "detectionEnabled": True,
        })["id"]
        print(f"Cámara creada (id {id_camara}).")

    celdas = [{"label": c["label"], "parkingSpaceId": puestos.get(c["puesto"]), "points": c["points"],
               "row": c["row"]} for c in demo["celdas"]]
    api.llamar("PUT", f"/monitoring/cameras/{id_camara}/cells", {"cells": celdas, "rows": demo.get("filas")})
    print(f"{len(celdas)} celdas cargadas.")

    destino = RAIZ / "referencia_celdas.jpg"
    if not destino.exists():
        shutil.copy(AQUI / "referencia_celdas.jpg", destino)
        print(f"Imagen de referencia copiada a {destino.name}.")

    print(f"\nListo. En el .env pon BACKEND_CAMERA_ID={id_camara} y reinicia el servicio de IA.")


if __name__ == "__main__":
    main()
