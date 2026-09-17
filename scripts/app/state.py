from __future__ import annotations

from threading import Lock
from typing import Any


class RuntimeState:
    def __init__(self, total_spaces: int) -> None:
        self.lock = Lock()
        self.latest_frame: bytes | None = None
        self.worker_error: str | None = None
        self.data: dict[str, Any] = {
            "timestamp": None,
            "total_celdas": total_spaces,
            "ocupadas": 0,
            "libres": total_spaces,
            "vehiculos_detectados": 0,
            "congestion_porcentaje": 0.0,
            "celdas": {},
            "placa": {
                "valor": None,
                "lecturas": [],
                "estado": "esperando",
                "ultimo_intento": None,
                "error": None,
            },
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.data)

    def update(self, data: dict[str, Any]) -> None:
        with self.lock:
            self.data = data

    def update_plate(self, data: dict[str, Any]) -> None:
        with self.lock:
            self.data["placa"] = data

    def set_frame(self, frame: bytes) -> None:
        with self.lock:
            self.latest_frame = frame

    def get_frame(self) -> bytes | None:
        with self.lock:
            return self.latest_frame
