from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import Settings


RELOGIN_ON_403_SECONDS = 60


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


@dataclass
class PlateEntryResult:
    """Respuesta de POST /api/assignments/plate-entry.

    outcome: PROMPTED (se le avisó al estudiante para que elija tiempo y puesto en la
    plataforma), ALREADY_PROMPTED, ASSIGNED, NOT_REGISTERED, STUDENT_INACTIVE,
    ALREADY_ASSIGNED, NO_SPACE o NO_LIVE_DATA. Solo ASSIGNED (estudiante sin usuario en
    la plataforma) trae los datos del puesto; los demás casos solo se informan.
    """

    outcome: str
    plate: str
    assignment_id: int | None
    parking_space_id: int | None
    parking_id: int | None
    space_number: str | None
    parking_name: str | None
    notified: bool
    raw: dict[str, Any]

    @property
    def assigned(self) -> bool:
        return self.outcome == "ASSIGNED"


class BackendClient:
    """Sesión contra educore-backend para registrar lo que detecta la cámara.

    Se autentica como una cuenta de servicio (un usuario con rol Operador,
    que ya tiene ASSIGNMENTS_MANAGE y PLATES_VIEW) y reutiliza la sesión
    (cookie JWT + CSRF) entre llamadas, re-logueándose si expira.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        # Los hilos de ocupación, celdas y placas comparten la sesión, y el
        # backend cambia la cookie CSRF en cada petición: si dos hilos se
        # cruzan, el POST sale con un token viejo y el backend responde 403.
        self._lock = threading.RLock()
        self._logged_in = False
        self._logged_in_at = float("-inf")
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
        self._logged_in_at = time.monotonic()

    def _ensure_login(self) -> None:
        if not self._logged_in:
            self.login()

    def _request(self, method: str, path: str, *, retry: bool = True, **kwargs: Any) -> requests.Response:
        with self._lock:
            return self._request_unlocked(method, path, retry=retry, **kwargs)

    def _request_unlocked(self, method: str, path: str, *, retry: bool, **kwargs: Any) -> requests.Response:
        self._ensure_login()
        try:
            response = self.session.request(
                method, self._url(path), timeout=self.settings.backend_request_timeout_seconds, **kwargs
            )
        except requests.RequestException as error:
            raise BackendError(f"Error de red llamando a {path}: {error}") from error

        if retry and self._session_expired(response.status_code):
            # La sesión expiró: se reintenta una sola vez con un login nuevo.
            self._logged_in = False
            self.login()
            headers = kwargs.get("headers")
            if headers and "X-XSRF-TOKEN" in headers:
                # El login nuevo cambió la cookie CSRF; el reintento debe usarla.
                kwargs["headers"] = {**headers, "X-XSRF-TOKEN": self._csrf_token() or ""}
            return self._request(method, path, retry=False, **kwargs)
        return response

    def _session_expired(self, status_code: int) -> bool:
        """El backend responde 403 (no 401) cuando el JWT vence, al cabo de una hora.

        Un 403 también puede ser falta de permisos, así que solo se toma como
        sesión vencida si el último login no es reciente: de lo contrario se
        haría un login nuevo en cada petición rechazada.
        """
        if status_code == 401:
            return True
        return status_code == 403 and time.monotonic() - self._logged_in_at > RELOGIN_ON_403_SECONDS

    def _mutating_request(self, method: str, path: str, *, json: dict[str, Any]) -> requests.Response:
        # El backend invalida la cookie XSRF-TOKEN en cuanto se hace una
        # petición GET (el mismo comportamiento por el que el front tiene su
        # propio reintento): hay que pedirla de nuevo justo antes de usarla,
        # no basta con la que quedó del login.
        with self._lock:
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

    def report_occupancy(self, camera_id: int, occupancy: dict[int, bool]) -> dict[str, Any]:
        """Le dice al backend qué puestos vinculados a esta cámara tienen un vehículo.

        El backend decide cómo cambia cada puesto: no libera uno asignado a un
        estudiante ni toca los reservados o en mantenimiento.
        """
        response = self._mutating_request(
            "POST",
            f"/api/monitoring/cameras/{camera_id}/occupancy",
            json={
                "cells": [
                    {"parkingSpaceId": space_id, "occupied": occupied}
                    for space_id, occupied in occupancy.items()
                ]
            },
        )
        if response.status_code != 200:
            raise BackendError(
                f"El backend rechazó la ocupación de la cámara {camera_id} "
                f"({response.status_code}): {response.text[:200]}"
            )
        return response.json().get("data", {})

    def report_layout(self, camera_id: int, version: str, cells: list[dict[str, Any]]) -> None:
        """Envía el plano de planta de los puestos de esta cámara (para el mapa 3D del estudiante)."""
        response = self._mutating_request(
            "PUT",
            f"/api/monitoring/cameras/{camera_id}/layout",
            json={"version": version, "cells": cells},
        )
        if response.status_code != 200:
            raise BackendError(
                f"El backend rechazó el plano de la cámara {camera_id} "
                f"({response.status_code}): {response.text[:200]}"
            )

    @staticmethod
    def _services_from(response: requests.Response) -> dict[str, bool]:
        data = response.json().get("data") or {}
        return {
            "espacios": bool(data.get("spacesEnabled", True)),
            "placas": bool(data.get("platesEnabled", True)),
        }

    def get_services(self) -> dict[str, bool]:
        """Qué dejó encendido el operador en la plataforma (Monitoreo)."""
        response = self._request("GET", "/api/monitoring/services")
        if response.status_code != 200:
            raise BackendError(
                f"No se pudieron consultar los servicios de detección "
                f"({response.status_code}): {response.text[:200]}"
            )
        return self._services_from(response)

    def set_services(self, spaces: bool | None, plates: bool | None) -> dict[str, bool]:
        """Guarda en la plataforma un cambio hecho desde el panel local (None = no cambiar)."""
        response = self._mutating_request(
            "PATCH",
            "/api/monitoring/services",
            json={"spacesEnabled": spaces, "platesEnabled": plates},
        )
        if response.status_code != 200:
            raise BackendError(
                f"El backend rechazó el cambio de servicios ({response.status_code}): {response.text[:200]}"
            )
        return self._services_from(response)

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

    def register_plate_entry(self, plate: str, parking_id: int | None = None) -> PlateEntryResult:
        """Le pasa al backend la placa leída en la entrada.

        El backend valida el vehículo y le avisa al estudiante para que él
        elija parqueadero, tiempo y puesto en la plataforma. Si el estudiante
        no tiene usuario allá, le asigna un puesto libre al azar.
        """
        response = self._mutating_request(
            "POST",
            "/api/assignments/plate-entry",
            json={"plate": plate.upper(), "parkingId": parking_id},
        )
        if response.status_code != 200:
            raise BackendError(
                f"El backend rechazó la lectura de la placa {plate} ({response.status_code}): {response.text[:300]}"
            )
        data = response.json().get("data", {})
        assignment = data.get("assignment") or {}
        return PlateEntryResult(
            outcome=data.get("outcome", ""),
            plate=data.get("plate", plate.upper()),
            assignment_id=assignment.get("id"),
            parking_space_id=assignment.get("parkingSpaceId"),
            parking_id=assignment.get("parkingId"),
            space_number=data.get("spaceNumber"),
            parking_name=data.get("parkingName"),
            notified=bool(data.get("notified")),
            raw=data,
        )

    def sync_plate_detection(self, plate: str) -> PlateEntryResult:
        """Punto de entrada único: placa confirmada en la entrada -> puesto + ticket."""
        return self.register_plate_entry(plate, self.settings.backend_parking_id)
