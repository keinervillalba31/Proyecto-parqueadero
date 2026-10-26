"""Arranca el servicio de monitoreo con un solo comando: `python run_service.py`.

Al encender ya queda todo funcionando: lee la cámara (o el video de respaldo en
bucle), detecta celdas y placas, y sincroniza con el backend si está activado.
No hace falta ejecutar ningún otro script.

Con `--panel` además abre el panel del operador en el navegador; si el servicio
ya estaba encendido, solo abre el panel. Es lo que usa `Abrir panel.bat`.
"""

import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Sin consola (pythonw: doble clic o tarea programada) no hay dónde escribir;
# los mensajes y errores van a un archivo para poder revisarlos.
if sys.stderr is None:
    sys.stdout = sys.stderr = open(PROJECT_DIR / "servicio.log", "a", encoding="utf-8", buffering=1)

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import uvicorn  # noqa: E402

from app.config import settings  # noqa: E402


def service_is_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", settings.service_port), timeout=1):
            return True
    except OSError:
        return False


def open_panel_when_ready(url: str) -> None:
    for _ in range(120):
        if service_is_up():
            webbrowser.open(url)
            return
        time.sleep(0.5)


def main() -> None:
    panel_url = f"http://localhost:{settings.service_port}/"
    open_panel = "--panel" in sys.argv[1:]
    if open_panel and service_is_up():
        webbrowser.open(panel_url)
        return

    # Se importa aquí porque carga YOLO: no hace falta si solo se iba a abrir el panel.
    from app.main import app
    from app.camera_discovery import mask_credentials

    print(f"Fuente de video: {mask_credentials(settings.video_source)}")
    if settings.video_fallback:
        print(f"Respaldo en bucle: {settings.video_fallback}")
    print(f"Servicio en http://{settings.service_host}:{settings.service_port}")
    print(f"Panel del operador: {panel_url}")
    if open_panel:
        threading.Thread(target=open_panel_when_ready, args=(panel_url,), daemon=True).start()
    # /video y /ws/estado nunca terminan solos: sin este límite, Ctrl+C se queda esperando
    # mientras haya una pestaña con el video abierta.
    uvicorn.run(app, host=settings.service_host, port=settings.service_port, timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
