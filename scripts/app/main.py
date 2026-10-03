from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .config import settings
from .video_service import VideoService


MAX_UPLOAD_BYTES = 10 * 1024 * 1024

service = VideoService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.start()
    yield
    service.stop()


app = FastAPI(title="Monitoreo de parqueadero", lifespan=lifespan)


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


@app.get("/video")
def video_stream() -> StreamingResponse:
    def frames():
        last_id = -1
        while True:
            frame_id, frame = service.state.get_frame()
            # Solo se envía el cuadro cuando el hilo de video produjo uno nuevo.
            if frame is not None and frame_id != last_id:
                last_id = frame_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.01)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/estado")
async def state_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(service.state.snapshot())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return
