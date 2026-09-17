from __future__ import annotations

import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .config import settings
from .video_service import VideoService


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
        key: current[key]
        for key in (
            "timestamp",
            "total_celdas",
            "ocupadas",
            "libres",
            "vehiculos_detectados",
            "congestion_porcentaje",
            "celdas",
        )
    }


@app.get("/placas/estado")
def plates_state() -> dict[str, Any]:
    return service.state.snapshot().get("placa", {})


@app.post("/vision/license-plates")
def read_license_plates(image: UploadFile = File(...)) -> dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen")

    temporary_path: str | None = None
    try:
        suffix = Path(image.filename or "").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            shutil.copyfileobj(image.file, temporary)
            temporary_path = temporary.name
        import cv2

        frame = cv2.imread(temporary_path)
        if frame is None:
            raise HTTPException(status_code=415, detail="No se pudo leer la imagen")
        plate = service.plates.read(frame)
        return {"plate": plate, "registered": plate in service.registered_vehicles if plate else False}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


@app.get("/video")
def video_stream() -> StreamingResponse:
    def frames():
        while True:
            frame = service.state.get_frame()
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await_seconds = 0.04
            import time

            time.sleep(await_seconds)

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
