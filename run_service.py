"""Arranca el servicio de monitoreo con un solo comando: `python run_service.py`.

Al encender ya queda todo funcionando: lee la cámara (o el video de respaldo en
bucle), detecta celdas y placas, y sincroniza con el backend si está activado.
No hace falta ejecutar ningún otro script.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import uvicorn  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


def main() -> None:
    print(f"Fuente de video: {settings.video_source}")
    if settings.video_fallback:
        print(f"Respaldo en bucle: {settings.video_fallback}")
    print(f"Servicio en http://{settings.service_host}:{settings.service_port}")
    # /video y /ws/estado nunca terminan solos: sin este límite, Ctrl+C se queda esperando
    # mientras haya una pestaña con el video abierta.
    uvicorn.run(app, host=settings.service_host, port=settings.service_port, timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
