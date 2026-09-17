from __future__ import annotations

import os
import re
import tempfile
from collections import Counter, deque
from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class PlateReader:
    def __init__(
        self,
        workspace: str,
        workflow: str,
        confirmation_reads: int,
        cooldown_seconds: float,
    ) -> None:
        self.workspace = workspace
        self.workflow = workflow
        self.confirmation_reads = confirmation_reads
        self.cooldown_seconds = cooldown_seconds
        self.reads: deque[str] = deque(maxlen=8)
        self.next_attempt_at = 0.0

    @staticmethod
    def normalize(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.upper().strip()
        value = re.sub(r"[^A-Z0-9]", "", value)
        return value or None

    def extract_text(self, output: Any) -> str | None:
        if isinstance(output, list):
            for item in output:
                plate = self.extract_text(item)
                if plate:
                    return plate
            return None
        if not isinstance(output, dict):
            return self.normalize(output)

        values = output.get("plate_text", [])
        if isinstance(values, str):
            values = [values]
        for value in values:
            plate = self.normalize(value)
            if plate:
                return plate
        return None

    def _client(self) -> Any:
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

    def read(self, frame: np.ndarray) -> str | None:
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temporary:
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    return None
                temporary.write(encoded.tobytes())
                temporary_path = temporary.name

            result = self._client().run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow,
                images={"image": temporary_path},
                use_cache=True,
            )
            return self.extract_text(result)
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    def confirm(self, plate: str) -> tuple[str, int] | None:
        self.reads.append(plate)
        confirmed, count = Counter(self.reads).most_common(1)[0]
        if count < self.confirmation_reads:
            return None
        self.reads.clear()
        return confirmed, count

    def reset(self) -> None:
        self.reads.clear()
