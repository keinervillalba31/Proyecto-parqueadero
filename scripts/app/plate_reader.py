from __future__ import annotations

import itertools
import os
import re
import tempfile
import time
from collections import Counter, deque
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


# Colombia: carros ABC123, motos ABC12D.
DEFAULT_PLATE_PATTERN = r"^([A-Z]{3}[0-9]{3}|[A-Z]{3}[0-9]{2}[A-Z])$"

# Caracteres que el OCR suele confundir entre letra y número.
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                   "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
                   "7": "T", "4": "A"}
MAX_AMBIGUOUS_CHARACTERS = 10


def edit_distance(first: str, second: str) -> int:
    """Número mínimo de letras a cambiar, agregar o quitar (Levenshtein)."""
    previous = list(range(len(second) + 1))
    for i, char_a in enumerate(first, start=1):
        current = [i]
        for j, char_b in enumerate(second, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def similar(first: str, second: str) -> bool:
    return edit_distance(first, second) <= 1


def vehicle_crops(
    frame: np.ndarray,
    vehicles: Iterable[Any],
    max_vehicles: int,
    minimum_width: int,
    skip_parked: bool = False,
    margin: float = 0.1,
) -> list[np.ndarray]:
    """Recorta los vehículos más grandes (los más cercanos) del cuadro.

    Enviar el recorte en vez del cuadro completo hace que la placa llegue más
    grande al OCR y sin letreros ni otras placas alrededor.
    """
    height, width = frame.shape[:2]
    selected = [
        vehicle
        for vehicle in vehicles
        if vehicle.box[2] - vehicle.box[0] >= minimum_width
        and not (skip_parked and vehicle.in_cell)
    ]
    selected.sort(key=lambda vehicle: vehicle.area, reverse=True)

    crops = []
    for vehicle in selected[:max_vehicles]:
        x1, y1, x2, y2 = vehicle.box
        pad_x = int((x2 - x1) * margin)
        pad_y = int((y2 - y1) * margin)
        crops.append(
            frame[
                max(y1 - pad_y, 0) : min(y2 + pad_y, height),
                max(x1 - pad_x, 0) : min(x2 + pad_x, width),
            ].copy()
        )
    return crops


class PlateReader:
    def __init__(
        self,
        workspace: str,
        workflow: str,
        confirmation_reads: int,
        cooldown_seconds: float,
        plate_pattern: str = DEFAULT_PLATE_PATTERN,
    ) -> None:
        self.workspace = workspace
        self.workflow = workflow
        self.confirmation_reads = confirmation_reads
        self.cooldown_seconds = cooldown_seconds
        # Un patrón vacío desactiva la validación del formato.
        self.pattern = re.compile(plate_pattern) if plate_pattern else None
        self.reads: deque[str] = deque(maxlen=12)
        # placa confirmada -> momento hasta el que no se vuelve a confirmar.
        self.cooldowns: dict[str, float] = {}
        self._cached_client: Any = None

    @staticmethod
    def normalize(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.upper().strip()
        value = re.sub(r"[^A-Z0-9]", "", value)
        return value or None

    def validate(self, text: str) -> str | None:
        """Devuelve la placa si tiene el formato esperado, corrigiendo O/0, I/1, etc."""
        if self.pattern is None:
            return text
        if self.pattern.fullmatch(text):
            return text

        options = []
        for char in text:
            swap = LETTER_TO_DIGIT.get(char) or DIGIT_TO_LETTER.get(char)
            options.append((char, swap) if swap else (char,))
        if sum(len(option) > 1 for option in options) > MAX_AMBIGUOUS_CHARACTERS:
            return None

        # Se prefiere la corrección que cambia menos caracteres, y se permite
        # cambiar como máximo un tercio para no convertir basura en una placa.
        max_changes = max(1, len(text) // 3)
        candidates = sorted(
            (
                (sum(a != b for a, b in zip(candidate, text)), candidate)
                for candidate in map("".join, itertools.product(*options))
            ),
        )
        return next(
            (
                candidate
                for changes, candidate in candidates
                if changes <= max_changes and self.pattern.fullmatch(candidate)
            ),
            None,
        )

    def extract_texts(self, output: Any) -> list[str]:
        """Todos los textos normalizados que devolvió Roboflow."""
        if isinstance(output, list):
            return [text for item in output for text in self.extract_texts(item)]
        if not isinstance(output, dict):
            text = self.normalize(output)
            return [text] if text else []

        values = output.get("plate_text", [])
        if isinstance(values, str):
            values = [values]
        return [text for text in map(self.normalize, values) if text]

    def valid_plates(self, texts: Iterable[str]) -> list[str]:
        plates = []
        for text in texts:
            plate = self.validate(text)
            if plate and plate not in plates:
                plates.append(plate)
        return plates

    def extract_text(self, output: Any) -> str | None:
        plates = self.valid_plates(self.extract_texts(output))
        return plates[0] if plates else None

    def _client(self) -> Any:
        if self._cached_client is None:
            self._cached_client = self._build_client()
        return self._cached_client

    def _build_client(self) -> Any:
        try:
            sdk = import_module("inference_sdk")
        except ImportError as error:
            raise RuntimeError(
                "inference-sdk no está disponible. Usa el entorno Python 3.12."
            ) from error

        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("ROBOFLOW_API_KEY no está configurada")
        if not self.workspace or not self.workflow:
            raise RuntimeError("Falta ROBOFLOW_WORKSPACE o ROBOFLOW_WORKFLOW")

        return sdk.InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        ).configure(sdk.InferenceConfiguration(api_key_transport="header"))

    def read_texts(self, frame: np.ndarray) -> list[str]:
        """Envía la imagen a Roboflow y devuelve todos los textos, válidos o no."""
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temporary:
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not ok:
                    return []
                temporary.write(encoded.tobytes())
                temporary_path = temporary.name

            result = self._client().run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow,
                images={"image": temporary_path},
                use_cache=True,
            )
            return self.extract_texts(result)
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    def read_all(self, frame: np.ndarray) -> list[str]:
        return self.valid_plates(self.read_texts(frame))

    def read(self, frame: np.ndarray) -> str | None:
        plates = self.read_all(frame)
        return plates[0] if plates else None

    def confirm(self, plate: str) -> tuple[str, int] | None:
        """Confirma una placa cuando hay suficientes lecturas parecidas.

        Lecturas que difieren en un solo carácter (BRF509 y BRF508) cuentan
        como la misma placa, y se publica la versión más repetida. Las
        lecturas de otros vehículos se conservan para seguir confirmándolos.
        """
        if self.in_cooldown(plate):
            return None
        self.reads.append(plate)
        group = [read for read in self.reads if similar(read, plate)]
        if len(group) < self.confirmation_reads:
            return None

        # La más repetida; en empate, la que se parece a más lecturas del grupo
        # (entre BRF50, RF509 y BRF509 se elige BRF509).
        counts = Counter(group)
        confirmed = max(
            counts,
            key=lambda read: (counts[read], sum(similar(read, other) for other in group)),
        )
        self.reads = deque(
            (read for read in self.reads if not similar(read, plate)),
            maxlen=self.reads.maxlen,
        )
        return confirmed, len(group)

    def reset(self) -> None:
        self.reads.clear()

    def in_cooldown(self, plate: str) -> bool:
        """True si esta placa (o una casi igual) se confirmó hace poco."""
        now = time.monotonic()
        self.cooldowns = {p: until for p, until in self.cooldowns.items() if until > now}
        return any(similar(plate, recent) for recent in self.cooldowns)

    def start_cooldown(self, plate: str) -> None:
        self.cooldowns[plate] = time.monotonic() + self.cooldown_seconds
