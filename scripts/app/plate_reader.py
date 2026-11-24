from __future__ import annotations

import itertools
import logging
import os
import re
import tempfile
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


log = logging.getLogger(__name__)

# Colombia: carros ABC123, motos ABC12D.
DEFAULT_PLATE_PATTERN = r"^([A-Z]{3}[0-9]{3}|[A-Z]{3}[0-9]{2}[A-Z])$"
# Con el tipo de vehículo que dio YOLO se sabe cuál de los dos formatos esperar: así
# "YZA916" en una moto se corrige a "YZA91G" (el OCR confunde la G final con un 6).
KIND_PATTERNS = {"carro": re.compile(r"^[A-Z]{3}[0-9]{3}$"), "moto": re.compile(r"^[A-Z]{3}[0-9]{2}[A-Z]$")}
MOTORCYCLE_CLASS_ID = 3

# Motores de lectura: "local" (modelos abiertos en este equipo, sin créditos),
# "roboflow" (API externa) e "hibrido" (local y, solo si no logra leer, Roboflow).
ENGINES = ("local", "roboflow", "hibrido")
DEFAULT_LOCAL_DETECTOR = "yolo-v9-t-640-license-plate-end2end"
DEFAULT_LOCAL_OCR = "cct-xs-v2-global-model"

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


def vote_characters(reads: Iterable[str], fallback: str, strict: bool = False) -> str | None:
    """Placa que sale de votar posición por posición entre lecturas del mismo vehículo.

    Un error del OCR suele afectar un solo carácter y en lecturas distintas cae en posiciones
    distintas (BRF509, BRF508, BRF589 -> BRF509). Solo cuentan las lecturas con el largo de
    `fallback` y hacen falta al menos tres. Si en una posición hay empate se queda el carácter
    de `fallback` (la lectura más repetida), o, con `strict`, no hay resultado (None).
    """
    same = [read for read in reads if len(read) == len(fallback)]
    if len(same) < 3:
        return None if strict else fallback
    voted = []
    for position, current in enumerate(fallback):
        counts = Counter(read[position] for read in same)
        best = max(counts.values())
        leaders = sorted(c for c, n in counts.items() if n == best)
        if len(leaders) > 1 and strict:
            return None
        voted.append(current if counts[current] == best else leaders[0])
    return "".join(voted)


def similar(first: str, second: str) -> bool:
    return edit_distance(first, second) <= 1


def vehicle_kind(vehicle: Any) -> str:
    """Formato de placa que se espera según la clase de YOLO del vehículo."""
    return "moto" if getattr(vehicle, "class_id", None) == MOTORCYCLE_CLASS_ID else "carro"


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
    return [crop for crop, _ in vehicle_crops_with_kinds(frame, vehicles, max_vehicles, minimum_width, skip_parked, margin)]


def vehicle_crops_with_kinds(
    frame: np.ndarray,
    vehicles: Iterable[Any],
    max_vehicles: int,
    minimum_width: int,
    skip_parked: bool = False,
    margin: float = 0.1,
) -> list[tuple[np.ndarray, str]]:
    """Como `vehicle_crops`, pero cada recorte va con el tipo de vehículo ("carro" o "moto")."""
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
        crops.append((
            frame[
                max(y1 - pad_y, 0) : min(y2 + pad_y, height),
                max(x1 - pad_x, 0) : min(x2 + pad_x, width),
            ].copy(),
            vehicle_kind(vehicle),
        ))
    return crops


# Zona donde va la placa: la parte baja del vehículo (la trasera de carros y motos).
ZONE_BOTTOM_FRACTION = 0.6
ZONE_MIN_WIDTH = 640
ZONE_SPLIT_WIDTH = 700
ZONE_TILE_FRACTION = 0.6


def plate_zones(image: np.ndarray) -> list[np.ndarray]:
    """Recortes de la parte baja del vehículo donde la placa se ve más grande.

    El detector de placas reduce la imagen a ~640 px; en un carro ancho (más de 1000 px)
    la placa queda diminuta, y en una moto el vehículo entero sobra. Por eso se prueba: la
    franja baja completa y, si es ancha, sus mitades izquierda y derecha (con solape), todas
    ampliadas a un ancho mínimo.
    """
    height, width = image.shape[:2]
    if height < 20 or width < 20:
        return []
    low = image[int(height * (1 - ZONE_BOTTOM_FRACTION)) :, :]
    pieces = [low]
    if width > ZONE_SPLIT_WIDTH:
        tile = int(width * ZONE_TILE_FRACTION)
        pieces += [low[:, :tile], low[:, width - tile :]]

    zones = []
    for piece in pieces:
        piece_width = piece.shape[1]
        if piece_width < ZONE_MIN_WIDTH:
            scale = ZONE_MIN_WIDTH / piece_width
            piece = cv2.resize(piece, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        zones.append(piece)
    return zones


def normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.upper().strip()
    value = re.sub(r"[^A-Z0-9]", "", value)
    return value or None


@dataclass
class LocalRead:
    """Lo que dio el lector local: los textos leídos y cuántas placas vio el detector."""

    texts: list[str]
    found: int


class LocalPlateEngine:
    """Detector y OCR de placas abiertos que corren en este equipo, sin internet ni créditos.

    Usa FastALPR (licencia MIT): un YOLOv9 entrenado para encontrar placas y un OCR
    entrenado solo con placas (fast-plate-ocr), ambos en ONNX para CPU. Los modelos se
    descargan la primera vez (unos 28 MB) y quedan en la caché del usuario.
    """

    def __init__(self, detector_model: str = DEFAULT_LOCAL_DETECTOR, ocr_model: str = DEFAULT_LOCAL_OCR) -> None:
        self.detector_model = detector_model
        self.ocr_model = ocr_model
        self._alpr: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._alpr is None:
            try:
                alpr_module = import_module("fast_alpr")
            except ImportError as error:
                raise RuntimeError(
                    "fast-alpr no está instalado. Instálalo con: "
                    "pip install --no-deps -r requirements-placas.txt"
                ) from error
            self._alpr = alpr_module.ALPR(
                detector_model=self.detector_model, ocr_model=self.ocr_model, ocr_device="cpu"
            )
        return self._alpr

    def read(self, image: np.ndarray) -> LocalRead:
        # Una sola lectura a la vez: el hilo de placas y el endpoint de prueba comparten el modelo.
        with self._lock:
            results = self._load().predict(image)
        texts = [text for text in (normalize_text(result.ocr.text) for result in results if result.ocr) if text]
        return LocalRead(texts=texts, found=len(results))


class PlateReader:
    def __init__(
        self,
        workspace: str,
        workflow: str,
        confirmation_reads: int,
        cooldown_seconds: float,
        plate_pattern: str = DEFAULT_PLATE_PATTERN,
        request_timeout: float = 30.0,
        engine: str = "local",
        local_detector: str = DEFAULT_LOCAL_DETECTOR,
        local_ocr: str = DEFAULT_LOCAL_OCR,
        local_engine: Any = None,
        zone_retry: bool = False,
    ) -> None:
        if engine not in ENGINES:
            raise ValueError(f"Motor de placas desconocido: {engine!r} (usa {', '.join(ENGINES)})")
        self.engine = engine
        # Si la lectura normal no da placa válida, se reintenta con la zona baja del vehículo.
        self.zone_retry = zone_retry
        self.local = local_engine or LocalPlateEngine(local_detector, local_ocr)
        # Cuántas lecturas hizo cada motor: muestra cuántas llamadas a la API se ahorran.
        self.stats: Counter[str] = Counter()
        self.workspace = workspace
        self.request_timeout = request_timeout
        self.workflow = workflow
        self.confirmation_reads = confirmation_reads
        self.cooldown_seconds = cooldown_seconds
        # Un patrón vacío desactiva la validación del formato.
        self.pattern = re.compile(plate_pattern) if plate_pattern else None
        self.reads: deque[str] = deque(maxlen=12)
        # placa confirmada -> momento hasta el que no se vuelve a confirmar.
        self.cooldowns: dict[str, float] = {}
        # El hilo de lectura y el que consulta al backend tocan lecturas y esperas.
        self._lock = threading.RLock()
        self._cached_client: Any = None

    @staticmethod
    def normalize(value: Any) -> str | None:
        return normalize_text(value)

    def validate(self, text: str, kind: str | None = None) -> str | None:
        """Devuelve la placa si tiene el formato esperado, corrigiendo O/0, I/1, etc.

        Con `kind` ("carro" o "moto") y el formato colombiano, primero se intenta el
        formato de ese vehículo cambiando a lo sumo un carácter; si no se puede, se
        valida como cualquier placa.
        """
        if self.pattern is None:
            return text
        if kind in KIND_PATTERNS and self.pattern.pattern == DEFAULT_PLATE_PATTERN:
            plate = self._correct(text, KIND_PATTERNS[kind], max_changes=1)
            if plate:
                return plate
        return self._correct(text, self.pattern, max_changes=max(1, len(text) // 3))

    @staticmethod
    def _correct(text: str, pattern: re.Pattern[str], max_changes: int) -> str | None:
        """La versión de `text` con ese formato que cambia menos caracteres confundibles."""
        if pattern.fullmatch(text):
            return text

        options = []
        for char in text:
            swap = LETTER_TO_DIGIT.get(char) or DIGIT_TO_LETTER.get(char)
            options.append((char, swap) if swap else (char,))
        if sum(len(option) > 1 for option in options) > MAX_AMBIGUOUS_CHARACTERS:
            return None

        # Se prefiere la corrección que cambia menos caracteres, y se permite cambiar
        # como máximo `max_changes` (un tercio, en general) para no convertir basura en una placa.
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
                if changes <= max_changes and pattern.fullmatch(candidate)
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

    def valid_plates(self, texts: Iterable[str], kind: str | None = None) -> list[str]:
        plates = []
        for text in texts:
            plate = self.validate(text, kind)
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

            result = self._run_workflow(temporary_path)
            return self.extract_texts(result)
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    def _run_workflow(self, image_path: str) -> Any:
        """Llama a Roboflow con un tiempo máximo.

        inference_sdk no le pone tiempo máximo a la petición: si Roboflow no
        responde, la llamada se queda esperando para siempre y, como solo hay una
        lectura a la vez, la lectura de placas se detiene sin dar error. Aquí la
        llamada corre en un hilo aparte; si no termina a tiempo se abandona (el
        hilo muere solo cuando la conexión se cierre) y se sigue con el siguiente intento.
        """
        outcome: dict[str, Any] = {}

        def call() -> None:
            try:
                outcome["result"] = self._client().run_workflow(
                    workspace_name=self.workspace,
                    workflow_id=self.workflow,
                    images={"image": image_path},
                    use_cache=True,
                )
            except BaseException as error:  # se relanza en el hilo que espera
                outcome["error"] = error

        worker = threading.Thread(target=call, name="roboflow-request", daemon=True)
        worker.start()
        worker.join(self.request_timeout)
        if worker.is_alive():
            raise TimeoutError(f"Roboflow no respondió en {self.request_timeout:.0f} s")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["result"]

    def roboflow_configured(self) -> bool:
        return bool(self.workspace and self.workflow and os.getenv("ROBOFLOW_API_KEY"))

    def read_candidates(self, frame: np.ndarray, kind: str | None = None) -> tuple[list[str], str]:
        """Textos leídos (válidos o no) y el motor que los dio ("local" o "roboflow").

        En modo híbrido se consulta Roboflow solo si el detector local vio una placa pero
        el OCR no dio ninguna válida: la confianza del OCR no sirve para decidirlo, porque
        sus lecturas erradas también salen con confianza alta, y esas ya las filtra la
        confirmación con varias lecturas.
        """
        if self.engine == "roboflow":
            self.stats["roboflow"] += 1
            return self.read_texts(frame), "roboflow"
        local = self.local.read(frame)
        self.stats["local"] += 1
        if self.zone_retry and not self.valid_plates(local.texts, kind):
            zone_texts, zone = self.read_zones(frame, kind)
            if zone:
                return zone_texts, "local-zona"
        if (self.engine == "local" or not local.found or self.valid_plates(local.texts, kind)
                or not self.roboflow_configured()):
            return local.texts, "local"
        try:
            texts = self.read_texts(frame)
        except Exception as error:  # sin Roboflow, se sigue con lo que leyó el modelo local
            log.warning("Roboflow no respondió en modo híbrido: %s", error)
            return local.texts, "local"
        self.stats["roboflow"] += 1
        return texts, "roboflow"

    def read_zones(self, image: np.ndarray, kind: str | None = None) -> tuple[list[str], int]:
        """Lee las zonas del vehículo hasta dar con una placa válida.

        Devuelve los textos leídos y el número (desde 1) de la zona que dio la placa válida,
        o 0 si ninguna la dio (en ese caso los textos son todo lo que se leyó, sin validar).
        """
        texts: list[str] = []
        for number, zone in enumerate(plate_zones(image), start=1):
            self.stats["local_zona"] += 1
            found = self.local.read(zone).texts
            if self.valid_plates(found, kind):
                return found, number
            texts += found
        return texts, 0

    def read_all(self, frame: np.ndarray, kind: str | None = None) -> list[str]:
        return self.valid_plates(self.read_candidates(frame, kind)[0], kind)

    def read(self, frame: np.ndarray, kind: str | None = None) -> str | None:
        plates = self.read_all(frame, kind)
        return plates[0] if plates else None

    def confirm(self, plate: str) -> tuple[str, int] | None:
        """Confirma una placa cuando hay suficientes lecturas parecidas.

        Lecturas que difieren en un solo carácter (BRF509 y BRF508) cuentan
        como la misma placa, y se publica la versión más repetida. Las
        lecturas de otros vehículos se conservan para seguir confirmándolos.

        Si hay un empate real (TZA91G y YZA91G, una vez cada una) se espera otra
        lectura para desempatar, en vez de quedarse con la que llegó primero: si
        esa era la errada, la espera de la placa confirmada bloquearía la correcta.
        Tras 2 × `confirmation_reads` + 1 lecturas se confirma igual, para no esperar siempre.
        """
        with self._lock:
            if self.in_cooldown(plate):
                return None
            self.reads.append(plate)
            group = [read for read in self.reads if similar(read, plate)]
            if len(group) < self.confirmation_reads:
                return None

            # La más repetida; en empate, la que se parece a más lecturas del grupo
            # (entre BRF50, RF509 y BRF509 se elige BRF509).
            counts = Counter(group)

            def score(read: str) -> tuple[int, int]:
                return counts[read], sum(similar(read, other) for other in group)

            ranked = sorted(counts, key=score, reverse=True)
            tied = len(ranked) > 1 and score(ranked[0]) == score(ranked[1])
            confirmed = ranked[0]
            if tied and len(group) < 2 * self.confirmation_reads + 1:
                # Empate entre lecturas completas: solo se confirma si, posición por posición,
                # hay una mayoría clara; si no, se espera otra lectura.
                decided = vote_characters(group, confirmed, strict=True)
                if decided is None or (self.pattern is not None and not self.pattern.fullmatch(decided)):
                    return None
                self._forget(plate)
                return decided, len(group)
            voted = vote_characters(group, confirmed)
            if voted != confirmed and (self.pattern is None or self.pattern.fullmatch(voted)):
                confirmed = voted
            self._forget(plate)
            return confirmed, len(group)

    def _forget(self, plate: str) -> None:
        self.reads = deque(
            (read for read in self.reads if not similar(read, plate)),
            maxlen=self.reads.maxlen,
        )

    def reset(self) -> None:
        self.reads.clear()

    def in_cooldown(self, plate: str) -> bool:
        """True si esta placa (o una casi igual) se confirmó hace poco."""
        with self._lock:
            now = time.monotonic()
            self.cooldowns = {p: until for p, until in self.cooldowns.items() if until > now}
            return any(similar(plate, recent) for recent in self.cooldowns)

    def start_cooldown(self, plate: str, seconds: float | None = None) -> None:
        """No volver a confirmar esta placa durante `seconds` (por defecto, cooldown_seconds)."""
        # También descarta sus lecturas pendientes: si el backend la reconoció en la
        # primera lectura, esa lectura no debe contar para otra confirmación después.
        with self._lock:
            wait = self.cooldown_seconds if seconds is None else seconds
            self.cooldowns[plate] = time.monotonic() + wait
            self._forget(plate)
