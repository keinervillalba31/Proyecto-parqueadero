from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .calibration import Calibration, Row, Stall, calibrate, top_down
from .config import settings
from .video_service import VideoService


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PANEL_PATH = Path(__file__).resolve().parent / "panel.html"


class ServicesChange(BaseModel):
    """Lo que se quiere encender (true) o apagar (false); lo que no se envía no cambia."""

    espacios: bool | None = None
    placas: bool | None = None


class StallIn(BaseModel):
    etiqueta: str
    tipo: str = "CAR"


class RowIn(BaseModel):
    """Una fila marcada en la imagen: 0 -> 1 es el frente (pasillo) y 3 el fondo detrás de 0."""

    nombre: str
    esquinas: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    puestos: list[StallIn] = Field(min_length=1)
    angulo: float = 90.0


class CalibrationIn(BaseModel):
    filas: list[RowIn] = Field(min_length=1)
    # Ancho y largo en metros por tipo de puesto, si no son los de siempre.
    medidas: dict[str, tuple[float, float]] | None = None


def _calibrate(request: CalibrationIn) -> Calibration:
    rows = [
        Row(
            row.nombre,
            np.array(row.esquinas, dtype=float),
            tuple(Stall(stall.etiqueta, stall.tipo) for stall in row.puestos),
            row.angulo,
        )
        for row in request.filas
    ]
    try:
        return calibrate(rows, request.medidas)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _points(points: np.ndarray, decimals: int) -> list[list[float]]:
    return np.round(points, decimals).tolist()


service = VideoService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.start()
    yield
    service.stop()


app = FastAPI(title="Monitoreo de parqueadero", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def panel() -> FileResponse:
    """Panel del operador: video, estado e interruptores de espacios y placas."""
    return FileResponse(PANEL_PATH, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/servicios")
def services() -> dict[str, bool]:
    return service.services()


@app.post("/servicios")
def change_services(change: ServicesChange) -> dict[str, bool]:
    return service.set_services(spaces=change.espacios, plates=change.placas)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return service.health()


@app.get("/estado")
def state() -> dict[str, Any]:
    return service.state.snapshot()


@app.get("/espacios/estado")
def spaces_state() -> dict[str, Any]:
    current = service.state.snapshot()
    return {
        key: current.get(key)
        for key in (
            "timestamp",
            "total_celdas",
            "ocupadas",
            "libres",
            "vehiculos_detectados",
            "congestion_porcentaje",
            "celdas",
            "reservas",
        )
    }


@app.get("/placas/estado")
def plates_state() -> dict[str, Any]:
    return service.state.snapshot().get("placa", {})


@app.get("/reservas")
def reservations() -> dict[str, Any]:
    return service.state.snapshot().get("reservas", {})


@app.post("/vision/license-plates")
def read_license_plates(image: UploadFile = File(...)) -> dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen")

    content = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera los 10 MB")

    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=415, detail="No se pudo leer la imagen")

    try:
        plate = service.plates.read(frame)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"plate": plate, "registered": plate in service.registered_vehicles if plate else False}


@app.get("/snapshot")
def snapshot() -> Response:
    """Cuadro actual sin dibujos (para trazar celdas desde la plataforma)."""
    image = service.snapshot_jpeg()
    if image is None:
        raise HTTPException(status_code=503, detail="Aún no hay imagen de la cámara")
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/calibracion")
def calibration(request: CalibrationIn) -> dict[str, Any]:
    """Calibra la cámara con las filas marcadas y devuelve las celdas de cada puesto.

    Las coordenadas de imagen son relativas (0..1); las del plano, en metros.
    No guarda nada: la plataforma decide qué celdas guardar.
    """
    result = _calibrate(request)
    return {
        "homografia": result.homography.tolist(),
        "verificada": result.verified,
        "error_cm": round(result.error_m * 100, 1),
        "filas": [
            {"nombre": row.label, "error_cm": round(row.error_m * 100, 1), "planta": _points(row.ground, 3)}
            for row in result.rows
        ],
        "celdas": [
            {
                "etiqueta": cell.label,
                "tipo": cell.type,
                "fila": cell.row,
                "puntos": _points(cell.image, 5),
                "planta": _points(cell.ground, 3),
            }
            for cell in result.cells
        ],
    }


@app.post("/calibracion/vista-superior")
def calibration_top_view(request: CalibrationIn) -> dict[str, Any]:
    """La imagen actual de la cámara vista desde arriba, para editar las filas sobre ella.

    `metros_a_pixeles` lleva un punto del plano (en metros) a un píxel de esta imagen.
    """
    frame = service.clean_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Aún no hay imagen de la cámara")
    image, meters_to_pixels = top_down(frame, _calibrate(request))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo generar la vista superior")
    return {
        "imagen": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode(),
        "ancho": image.shape[1],
        "alto": image.shape[0],
        "metros_a_pixeles": meters_to_pixels.tolist(),
    }


def _mjpeg_stream(
    get_frame: Callable[[], tuple[int, bytes | None]],
) -> StreamingResponse:
    def frames():
        last_id = -1
        while True:
            frame_id, frame = get_frame()
            if frame is not None and frame_id != last_id:
                last_id = frame_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.01)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video")
@app.get("/video/parqueadero")
def parking_video_stream() -> StreamingResponse:
    return _mjpeg_stream(service.state.get_frame)


@app.get("/video/placas")
def plate_video_stream() -> StreamingResponse:
    if not service.settings.plate_video_source:
        raise HTTPException(
            status_code=503,
            detail="No se configuró PLATE_VIDEO_SOURCE para el video de placas",
        )
    if service.get_plate_frame()[1] is None:
        raise HTTPException(status_code=503, detail="Aún no hay imagen de la cámara de placas")
    return _mjpeg_stream(service.get_plate_frame)


@app.websocket("/ws/estado")
async def state_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(service.state.snapshot())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return
