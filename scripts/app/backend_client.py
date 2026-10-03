from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import Settings


class BackendError(RuntimeError):
    """Cualquier falla al hablar con educore-backend (red, login, datos)."""


@dataclass
class AssignmentResult:
    assignment_id: int
    student_id: int
    parking_space_id: int | None
    parking_id: int | None
    status: str | None
    raw: dict[str, Any]


class BackendClient:
    """Sesión contra educore-backend para registrar lo que detecta la cámara.

    Se autentica como una cuenta de servicio (un usuario con rol Operador,
    que ya tiene ASSIGNMENTS_MANAGE y PLATES_VIEW) y reutiliza la sesión
    (cookie JWT + CSRF) entre llamadas, re-logueándose si expira.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self._logged_in = False
        self._students_cache: list[dict[str, Any]] = []
        self._students_cache_at: float = 0.0

    # -- sesión -----------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.settings.backend_base_url.rstrip('/')}{path}"

    def _csrf_token(self) -> str | None:
        return self.session.cookies.get("XSRF-TOKEN")

    def _fetch_csrf(self) -> None:
        response = self.session.get(
            self._url("/api/auth/csrf"), timeout=self.settings.backend_request_timeout_seconds
        )
        response.raise_for_status()

    def login(self) -> None:
        if not self.settings.backend_service_user_code:
            raise BackendError(
                "Falta configurar la cuenta de servicio (BACKEND_SERVICE_USER_CODE, "
                "BACKEND_SERVICE_IDENTITY_DOCUMENT, BACKEND_SERVICE_PASSWORD)"
            )
        try:
            self._fetch_csrf()
            response = self.session.post(
                self._url("/api/auth/login"),
                json={
                    "userCode": self.settings.backend_service_user_code,
                    "identityDocument": self.settings.backend_service_identity_document,
                    "password": self.settings.backend_service_password,
                },
                headers={"X-XSRF-TOKEN": self._csrf_token() or ""},
                timeout=self.settings.backend_request_timeout_seconds,
            )
        except requests.RequestException as error:
            raise BackendError(f"No se pudo conectar con el backend: {error}") from error

        if response.status_code != 200:
            raise BackendError(
                f"Login contra el backend falló ({response.status_code}): {response.text[:200]}"
            )
        self._logged_in = True

    def _ensure_login(self) -> None:
        if not self._logged_in:
            self.login()

    def _request(self, method: str, path: str, *, retry: bool = True, **kwargs: Any) -> requests.Response:
        self._ensure_login()
        try:
            response = self.session.request(
                method, self._url(path), timeout=self.settings.backend_request_timeout_seconds, **kwargs
            )
        except requests.RequestException as error:
            raise BackendError(f"Error de red llamando a {path}: {error}") from error

        if response.status_code == 401 and retry:
            # La sesión expiró: se reintenta una sola vez con un login nuevo.
            self._logged_in = False
            self.login()
            headers = kwargs.get("headers")
            if headers and "X-XSRF-TOKEN" in headers:
                # El login nuevo cambió la cookie CSRF; el reintento debe usarla.
                kwargs["headers"] = {**headers, "X-XSRF-TOKEN": self._csrf_token() or ""}
            return self._request(method, path, retry=False, **kwargs)
        return response

    def _mutating_request(self, method: str, path: str, *, json: dict[str, Any]) -> requests.Response:
        # El backend invalida la cookie XSRF-TOKEN en cuanto se hace una
        # petición GET (el mismo comportamiento por el que el front tiene su
        # propio reintento): hay que pedirla de nuevo justo antes de usarla,
        # no basta con la que quedó del login.
        self._ensure_login()
        try:
            self._fetch_csrf()
        except requests.RequestException as error:
            raise BackendError(f"No se pudo refrescar el token CSRF: {error}") from error
        headers = {"X-XSRF-TOKEN": self._csrf_token() or ""}
        return self._request(method, path, json=json, headers=headers)

    # -- consultas ----------------------------------------------------------

    def get_vehicle_by_plate(self, plate: str) -> dict[str, Any] | None:
        response = self._request("GET", f"/api/vehicles/plate/{plate.upper()}")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise BackendError(
                f"No se pudo consultar el vehículo {plate} ({response.status_code}): {response.text[:200]}"
            )
        return response.json().get("data")

    def get_camera_cells(self, camera_id: int) -> dict[str, Any]:
        """Celdas que el administrador trazó para esta cámara (coordenadas relativas 0 a 1)."""
        response = self._request("GET", f"/api/monitoring/cameras/{camera_id}/cells")
        if response.status_code != 200:
            raise BackendError(
                f"No se pudieron consultar las celdas de la cámara {camera_id} "
                f"({response.status_code}): {response.text[:200]}"
            )
        return response.json().get("data", {"version": "", "cells": []})

    def list_students(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        age = time.monotonic() - self._students_cache_at
        if not force_refresh and self._students_cache and age < self.settings.backend_students_cache_seconds:
            return self._students_cache

        response = self._request("GET", "/api/assignments/students")
        if response.status_code != 200:
            raise BackendError(
                f"No se pudo consultar la lista de estudiantes ({response.status_code}): {response.text[:200]}"
            )
        self._students_cache = response.json().get("data", [])
        self._students_cache_at = time.monotonic()
        return self._students_cache

    def find_student_id_by_code(self, student_code: str) -> int | None:
        for student in self.list_students():
            if student.get("studentCode", "").upper() == student_code.upper():
                return student.get("id")
        # El código no apareció: puede ser un estudiante nuevo, se refresca una vez.
        for student in self.list_students(force_refresh=True):
            if student.get("studentCode", "").upper() == student_code.upper():
                return student.get("id")
        return None

    # -- acción principal -----------------------------------------------------

    def auto_assign(self, student_id: int, parking_id: int | None = None) -> AssignmentResult:
        response = self._mutating_request(
            "POST",
            "/api/assignments/auto",
            json={"studentId": student_id, "parkingId": parking_id},
        )
        if response.status_code != 201:
            raise BackendError(
                f"El backend rechazó la asignación automática ({response.status_code}): {response.text[:300]}"
            )
        data = response.json().get("data", {})
        return AssignmentResult(
            assignment_id=data.get("id"),
            student_id=student_id,
            parking_space_id=data.get("parkingSpaceId"),
            parking_id=data.get("parkingId"),
            status=data.get("status"),
            raw=data,
        )

    def sync_plate_detection(self, plate: str) -> AssignmentResult | None:
        """Punto de entrada único: placa confirmada -> vehículo -> estudiante -> asignación.

        Devuelve None si la placa no corresponde a un vehículo activo registrado
        en el backend (no es un error, simplemente no hay nada que sincronizar).
        """
        vehicle = self.get_vehicle_by_plate(plate)
        if not vehicle or not vehicle.get("active", True):
            return None

        student_code = vehicle.get("studentCode")
        if not student_code:
            return None

        student_id = self.find_student_id_by_code(student_code)
        if student_id is None:
            raise BackendError(
                f"El vehículo {plate} está asociado al código {student_code}, "
                "pero no se encontró ese estudiante en el backend"
            )

        return self.auto_assign(student_id, self.settings.backend_parking_id)
