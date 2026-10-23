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
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .calibration import Calibration, Row, Stall, calibrate, suggested_corners, top_down
from .config import settings
from .video_service import VideoService
from .vision_node import VisionNode


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


def _rows(request: CalibrationIn) -> list[Row]:
    return [
        Row(
            row.nombre,
            np.array(row.esquinas, dtype=float),
            tuple(Stall(stall.etiqueta, stall.tipo) for stall in row.puestos),
            row.angulo,
        )
        for row in request.filas
    ]


def _calibrate(request: CalibrationIn) -> Calibration:
    try:
        return calibrate(_rows(request), request.medidas)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _points(points: np.ndarray, decimals: int) -> list[list[float]]:
    return np.round(points, decimals).tolist()


node = VisionNode(settings)


def _camera(camera_id: int | None) -> VideoService:
    """La cámara de la ruta: /camaras/{id}/... o, sin número, la única (o la primera de ocupación)."""
    worker = node.default_worker() if camera_id is None else node.worker(camera_id)
    if worker is None:
        raise HTTPException(
            status_code=404,
            detail="Este nodo no atiende esa cámara" if camera_id is not None else "El nodo aún no tiene cámaras",
        )
    return worker


@asynccontextmanager
async def lifespan(_: FastAPI):
    node.start()
    yield
    node.stop()


app = FastAPI(title="Monitoreo de parqueadero", lifespan=lifespan)
# El editor de celdas de la plataforma calibra y pide la vista superior desde el navegador.
app.add_middleware(
    CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


@app.get("/", include_in_schema=False)
def panel() -> FileResponse:
    """Panel del operador: video, estado e interruptores de espacios y placas."""
    return FileResponse(PANEL_PATH, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/nodo")
def node_status() -> dict[str, Any]:
    """Qué cámaras atiende este proceso, con qué dispositivo y cómo va cada una."""
    return node.status()


@app.get("/servicios")
@app.get("/camaras/{camera_id}/servicios")
def services(camera_id: int | None = None) -> dict[str, bool]:
    return _camera(camera_id).services()


@app.post("/servicios")
@app.post("/camaras/{camera_id}/servicios")
def change_services(change: ServicesChange, camera_id: int | None = None) -> dict[str, bool]:
    return _camera(camera_id).set_services(spaces=change.espacios, plates=change.placas)


@app.get("/health")
@app.get("/camaras/{camera_id}/health")
def health(camera_id: int | None = None) -> dict[str, str | bool]:
    if camera_id is None and node.default_worker() is None:
        return {"ok": node.mode == "nodo", "error": node.last_sync_error or "", "fuente": "nodo"}
    return _camera(camera_id).health()


@app.get("/estado")
@app.get("/camaras/{camera_id}/estado")
def state(camera_id: int | None = None) -> dict[str, Any]:
    return _camera(camera_id).state.snapshot()


@app.get("/espacios/estado")
@app.get("/camaras/{camera_id}/espacios/estado")
def spaces_state(camera_id: int | None = None) -> dict[str, Any]:
    current = _camera(camera_id).state.snapshot()
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
@app.get("/camaras/{camera_id}/placas/estado")
def plates_state(camera_id: int | None = None) -> dict[str, Any]:
    return _camera(camera_id).state.snapshot().get("placa", {})


@app.get("/reservas")
@app.get("/camaras/{camera_id}/reservas")
def reservations(camera_id: int | None = None) -> dict[str, Any]:
    return _camera(camera_id).state.snapshot().get("reservas", {})


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

    camera = _camera(None)
    try:
        plate = camera.plates.read(frame)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"plate": plate, "registered": plate in camera.registered_vehicles if plate else False}


def _aligner(camera_id: int | None):
    aligner = _camera(camera_id).parking.aligner
    if aligner is None:
        raise HTTPException(status_code=409, detail="Esta cámara no detecta espacios: no tiene celdas")
    return aligner


@app.post("/celdas/referencia")
@app.post("/camaras/{camera_id}/celdas/referencia")
async def cells_reference(request: Request, camera_id: int | None = None) -> dict[str, Any]:
    """El cuadro sobre el que se trazaron las celdas (JPEG), que el editor envía al guardarlas.

    Con él se compensa el movimiento de la cámara: las celdas siguen a la imagen.
    """
    aligner = _aligner(camera_id)
    data = await request.body()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        raise HTTPException(status_code=400, detail="La imagen de referencia no es válida")
    aligner.set_reference(image)
    return {"referencia": "guardada", "ancho": image.shape[1], "alto": image.shape[0]}


@app.get("/celdas/referencia")
@app.get("/camaras/{camera_id}/celdas/referencia")
def cells_reference_image(camera_id: int | None = None) -> Response:
    """El cuadro sobre el que se trazaron las celdas, para que el editor las muestre donde se trazaron.

    Con la cámara moviéndose, un cuadro actual queda corrido respecto de las celdas aunque
    la detección lo compense. La cabecera X-Referencia dice si la guardó el editor o se estimó.
    """
    aligner = _aligner(camera_id)
    image = aligner.reference_jpeg()
    if image is None:
        raise HTTPException(status_code=404, detail="Todavía no hay imagen de referencia")
    return Response(
        content=image,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Referencia": aligner.source,
            "Access-Control-Expose-Headers": "X-Referencia",
        },
    )


@app.get("/snapshot")
@app.get("/camaras/{camera_id}/snapshot")
def snapshot(camera_id: int | None = None) -> Response:
    """Cuadro actual sin dibujos (para trazar celdas desde la plataforma)."""
    image = _camera(camera_id).snapshot_jpeg()
    if image is None:
        raise HTTPException(status_code=503, detail="Aún no hay imagen de la cámara")
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/calibracion")
@app.post("/camaras/{camera_id}/calibracion")
def calibration(request: CalibrationIn, camera_id: int | None = None) -> dict[str, Any]:
    """Calibra la cámara con las filas marcadas y devuelve las celdas de cada puesto.

    Las coordenadas de imagen son relativas (0..1); las del plano, en metros.
    No guarda nada: la plataforma decide qué celdas guardar. No depende de la cámara: la
    ruta con número existe para que el editor use una sola dirección base por cámara.
    """
    result = _calibrate(request)
    suggestions = suggested_corners(_rows(request), request.medidas)
    return {
        "homografia": result.homography.tolist(),
        "verificada": result.verified,
        "error_cm": round(result.error_m * 100, 1),
        "filas": [
            {
                "nombre": row.label,
                "error_cm": round(row.error_m * 100, 1),
                "planta": _points(row.ground, 3),
                # Dónde deberían ir sus esquinas para encajar con las demás filas (null con una sola fila).
                "ajustada": _points(suggestions[row.label].corners, 5) if row.label in suggestions else None,
                "desvio_cm": round(suggestions[row.label].deviation_m * 100, 1) if row.label in suggestions else None,
            }
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
@app.post("/camaras/{camera_id}/calibracion/vista-superior")
def calibration_top_view(request: CalibrationIn, imagen: str = "actual", camera_id: int | None = None) -> dict[str, Any]:
    """La imagen de la cámara vista desde arriba, para editar las filas sobre ella.

    Con `imagen=referencia` se usa el cuadro sobre el que se trazaron las celdas (si lo hay),
    para que coincida con lo que muestra el editor. `metros_a_pixeles` lleva un punto del
    plano (en metros) a un píxel de esta imagen.
    """
    camera = _camera(camera_id)
    aligner = camera.parking.aligner
    frame = aligner.reference_frame() if imagen == "referencia" and aligner is not None else None
    if frame is None:
        frame = camera.clean_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Aún no hay imagen de la cámara")
    result = _calibrate(request)
    image, meters_to_pixels = top_down(frame, result)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo generar la vista superior")
    return {
        "imagen": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode(),
        "ancho": image.shape[1],
        "alto": image.shape[0],
        "metros_a_pixeles": meters_to_pixels.tolist(),
        # La de esta imagen: si las filas cambian después, la imagen sigue correspondiendo a esta.
        "homografia": result.homography.tolist(),
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
@app.get("/camaras/{camera_id}/video")
def parking_video_stream(camera_id: int | None = None) -> StreamingResponse:
    return _mjpeg_stream(_camera(camera_id).state.get_frame)


@app.get("/video/placas")
def plate_video_stream() -> StreamingResponse:
    service = _camera(None)
    if not service.settings.plate_video_source:
        raise HTTPException(
            status_code=503,
            detail="No se configuró PLATE_VIDEO_SOURCE para el video de placas",
        )
    if service.get_plate_frame()[1] is None:
        raise HTTPException(status_code=503, detail="Aún no hay imagen de la cámara de placas")
    return _mjpeg_stream(service.get_plate_frame)


@app.websocket("/ws/estado")
@app.websocket("/camaras/{camera_id}/ws/estado")
async def state_websocket(websocket: WebSocket, camera_id: int | None = None) -> None:
    await websocket.accept()
    try:
        while True:
            worker = node.default_worker() if camera_id is None else node.worker(camera_id)
            if worker is not None:
                await websocket.send_json(worker.state.snapshot())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return
