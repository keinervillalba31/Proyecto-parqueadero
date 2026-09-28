from __future__ import annotations

import copy
from threading import Lock
from typing import Any


class RuntimeState:
    def __init__(self, total_spaces: int) -> None:
        self.lock = Lock()
        self.latest_frame: bytes | None = None
        self.frame_id = 0
        self._worker_error: str | None = None
        self.data: dict[str, Any] = {
            "timestamp": None,
            "total_celdas": total_spaces,
            "ocupadas": 0,
            "libres": total_spaces,
            "vehiculos_detectados": 0,
            "congestion_porcentaje": 0.0,
            "celdas": {},
            "reservas": {},
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
            return copy.deepcopy(self.data)

    def update(self, data: dict[str, Any]) -> None:
        """Actualiza la ocupación sin tocar la placa, que la escribe otro hilo."""
        with self.lock:
            placa = self.data["placa"]
            self.data = {**data, "placa": placa}

    def update_plate(self, changes: dict[str, Any], replace: bool = False) -> None:
        """Mezcla los cambios con la placa actual de forma atómica."""
        with self.lock:
            base = {} if replace else self.data["placa"]
            self.data["placa"] = {**base, **changes}

    def set_frame(self, frame: bytes) -> None:
        with self.lock:
            self.latest_frame = frame
            self.frame_id += 1

    def get_frame(self) -> tuple[int, bytes | None]:
        with self.lock:
            return self.frame_id, self.latest_frame

    @property
    def worker_error(self) -> str | None:
        with self.lock:
            return self._worker_error

    @worker_error.setter
    def worker_error(self, value: str | None) -> None:
        with self.lock:
            self._worker_error = value
