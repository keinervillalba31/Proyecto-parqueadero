from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from shapely.geometry import Polygon, box as shapely_box
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_PATH = Path(os.getenv("YOLO_MODEL", SCRIPT_DIR / "yolo11n.pt"))
CELLS_PATH = Path(os.getenv("PARKING_CELLS", SCRIPT_DIR / "estacionamientos.json"))
SOURCE = os.getenv("VIDEO_SOURCE", "0")
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.15"))
COVERAGE_THRESHOLD = 0.20
MIN_BOX_AREA = 2000
VEHICLE_CLASSES = {2: "Carro", 3: "Moto", 5: "Bus", 7: "Camion"}

with CELLS_PATH.open("r", encoding="utf-8") as file:
    parking_cells = json.load(file)

cells = {
    name: {
        "points": np.array(points, dtype=np.int32),
        "polygon": Polygon(points),
    }
    for name, points in parking_cells.items()
}
model = YOLO(str(MODEL_PATH))

lock = threading.Lock()
latest_frame: bytes | None = None
latest_state: dict[str, Any] = {
    "timestamp": None,
    "total_celdas": len(cells),
    "ocupadas": 0,
    "libres": len(cells),
    "vehiculos_detectados": 0,
    "congestion_porcentaje": 0.0,
    "celdas": {},
}
worker_error: str | None = None


def open_source() -> cv2.VideoCapture:
    source: int | str = int(SOURCE) if SOURCE.isdigit() else SOURCE
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
    return capture


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    occupied = {name: False for name in cells}
    detected_vehicles = 0
    result = model(frame, conf=CONFIDENCE, verbose=False)[0]

    for detection in result.boxes:
        class_id = int(detection.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        detected_vehicles += 1
        vehicle_box = shapely_box(x1, y1, x2, y2)
        label = VEHICLE_CLASSES[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        for name, cell in cells.items():
            if cell["polygon"].contains(vehicle_box.centroid):
                occupied[name] = True
                continue
            if cell["polygon"].intersects(vehicle_box):
                intersection = cell["polygon"].intersection(vehicle_box).area
                if intersection / cell["polygon"].area >= COVERAGE_THRESHOLD:
                    occupied[name] = True

    for name, cell in cells.items():
        is_occupied = occupied[name]
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        status = "Ocupado" if is_occupied else "Libre"
        points = cell["points"]
        cv2.polylines(frame, [points], True, color, 2)
        cv2.putText(
            frame,
            f"{name}: {status}",
            tuple(points[0] + np.array([0, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )

    occupied_count = sum(occupied.values())
    state = {
        "timestamp": time.time(),
        "total_celdas": len(cells),
        "ocupadas": occupied_count,
        "libres": len(cells) - occupied_count,
        "vehiculos_detectados": detected_vehicles,
        "congestion_porcentaje": round(occupied_count / len(cells) * 100, 2)
        if cells
        else 0.0,
        "celdas": {
            name: {"ocupado": is_occupied, "estado": "ocupado" if is_occupied else "libre"}
            for name, is_occupied in occupied.items()
        },
    }
    return frame, state


def video_worker() -> None:
    global latest_frame, latest_state, worker_error
    try:
        capture = open_source()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated_frame, state = process_frame(frame)
            ok, encoded = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            with lock:
                latest_frame = encoded.tobytes()
                latest_state = state
    except Exception as error:
        worker_error = str(error)
    finally:
        if "capture" in locals():
            capture.release()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=video_worker, name="video-worker", daemon=True).start()
    yield


app = FastAPI(title="Monitoreo vehicular en tiempo real", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"ok": worker_error is None, "error": worker_error or ""}


@app.get("/estado")
def get_state() -> dict[str, Any]:
    with lock:
        return dict(latest_state)


@app.get("/video")
def video_stream() -> StreamingResponse:
    def frames():
        while True:
            with lock:
                frame = latest_frame
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/estado")
async def state_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            with lock:
                state = dict(latest_state)
            await websocket.send_json(state)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_video:app", host="0.0.0.0", port=8000)
