from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.backend_client import BackendClient, BackendError


def make_settings(**overrides: Any) -> SimpleNamespace:
    defaults = dict(
        backend_base_url="http://backend.test",
        backend_service_user_code="OP001",
        backend_service_identity_document="1100000003",
        backend_service_password="Operador2026*",
        backend_parking_id=None,
        backend_request_timeout_seconds=5,
        backend_students_cache_seconds=300,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeCookies(dict):
    def get(self, key, default=None):  # pragma: no cover - trivial
        return super().get(key, default)


class FakeSession:
    """Sustituye a requests.Session: registra llamadas y devuelve respuestas guionadas."""

    def __init__(self) -> None:
        self.cookies = FakeCookies()
        self.calls: list[tuple[str, str, dict]] = []
        self.script: dict[str, FakeResponse] = {}
        self.login_attempts = 0

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/api/auth/csrf"):
            self.cookies["XSRF-TOKEN"] = "csrf-token-1"
            return FakeResponse(200, {})
        return self._resolve(url)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/api/auth/login"):
            self.login_attempts += 1
            self.cookies["access_token"] = "jwt-token"
            return FakeResponse(200, {"success": True})
        return self._resolve(url)

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        return self._resolve(url)

    def _resolve(self, url: str) -> FakeResponse:
        for suffix, response in self.script.items():
            if url.endswith(suffix):
                return response
        raise AssertionError(f"No hay respuesta guionada para {url}")


@pytest.fixture
def client() -> tuple[BackendClient, FakeSession]:
    backend = BackendClient(make_settings())
    fake_session = FakeSession()
    backend.session = fake_session
    return backend, fake_session


def test_login_requires_service_account_configured():
    backend = BackendClient(make_settings(backend_service_user_code=""))
    with pytest.raises(BackendError):
        backend.login()


def test_login_sends_csrf_header_and_credentials(client):
    backend, session = client
    backend.login()

    assert session.login_attempts == 1
    method, url, kwargs = session.calls[-1]
    assert url.endswith("/api/auth/login")
    assert kwargs["headers"]["X-XSRF-TOKEN"] == "csrf-token-1"
    assert kwargs["json"] == {
        "userCode": "OP001",
        "identityDocument": "1100000003",
        "password": "Operador2026*",
    }
    assert backend._logged_in is True


def test_get_vehicle_by_plate_returns_data(client):
    backend, session = client
    session.script["/api/vehicles/plate/ABC123"] = FakeResponse(
        200, {"success": True, "data": {"plate": "ABC123", "studentCode": "1151002", "active": True}}
    )

    vehicle = backend.get_vehicle_by_plate("ABC123")

    assert vehicle == {"plate": "ABC123", "studentCode": "1151002", "active": True}


def test_get_vehicle_by_plate_returns_none_when_not_found(client):
    backend, session = client
    session.script["/api/vehicles/plate/ZZZ999"] = FakeResponse(404, {}, text="not found")

    assert backend.get_vehicle_by_plate("ZZZ999") is None


def test_find_student_id_by_code_matches_case_insensitively(client):
    backend, session = client
    session.script["/api/assignments/students"] = FakeResponse(
        200,
        {
            "success": True,
            "data": [
                {"id": 1, "studentCode": "1151001", "fullName": "Keiner"},
                {"id": 2, "studentCode": "1151002", "fullName": "Edinson"},
            ],
        },
    )

    assert backend.find_student_id_by_code("1151002") == 2
    assert backend.find_student_id_by_code("no-existe") is None


def test_students_list_is_cached_between_calls(client):
    backend, session = client
    session.script["/api/assignments/students"] = FakeResponse(
        200, {"success": True, "data": [{"id": 1, "studentCode": "1151001", "fullName": "Keiner"}]}
    )

    backend.list_students()
    backend.list_students()

    calls_to_students = [c for c in session.calls if c[1].endswith("/api/assignments/students")]
    assert len(calls_to_students) == 1


def test_auto_assign_sends_csrf_and_parses_response(client):
    backend, session = client
    session.script["/api/assignments/auto"] = FakeResponse(
        201,
        {
            "success": True,
            "data": {"id": 55, "studentId": 2, "parkingSpaceId": 9, "parkingId": 1, "status": "ACTIVE"},
        },
    )

    result = backend.auto_assign(student_id=2, parking_id=1)

    assert result.assignment_id == 55
    assert result.parking_space_id == 9
    assert result.status == "ACTIVE"
    method, url, kwargs = session.calls[-1]
    assert method == "POST"
    assert kwargs["json"] == {"studentId": 2, "parkingId": 1}
    assert kwargs["headers"]["X-XSRF-TOKEN"] == "csrf-token-1"


def test_auto_assign_raises_on_rejection(client):
    backend, session = client
    session.script["/api/assignments/auto"] = FakeResponse(
        400, {"success": False, "message": "No hay espacios disponibles"}
    )

    with pytest.raises(BackendError):
        backend.auto_assign(student_id=2)


def test_sync_plate_detection_posts_plate_entry(client):
    backend, session = client
    session.script["/api/assignments/plate-entry"] = FakeResponse(
        200,
        {
            "success": True,
            "data": {
                "outcome": "ASSIGNED",
                "plate": "ABC123",
                "assignment": {"id": 55, "studentId": 2, "parkingSpaceId": 9, "parkingId": 1, "status": "ACTIVE"},
                "parkingName": "Central",
                "spaceNumber": "A-07",
                "zone": "A",
                "notified": True,
            },
        },
    )

    result = backend.sync_plate_detection("abc123")

    assert result.assigned
    assert result.assignment_id == 55
    assert result.parking_space_id == 9
    assert result.space_number == "A-07"
    assert result.notified is True
    method, url, kwargs = session.calls[-1]
    assert method == "POST"
    assert kwargs["json"] == {"plate": "ABC123", "parkingId": None}
    assert kwargs["headers"]["X-XSRF-TOKEN"] == "csrf-token-1"


@pytest.mark.parametrize("outcome", ["NOT_REGISTERED", "ALREADY_ASSIGNED", "NO_SPACE"])
def test_sync_plate_detection_reports_ignored_outcomes(client, outcome):
    backend, session = client
    session.script["/api/assignments/plate-entry"] = FakeResponse(
        200,
        {"success": True, "data": {"outcome": outcome, "plate": "ZZZ999", "assignment": None, "notified": False}},
    )

    result = backend.sync_plate_detection("ZZZ999")

    assert not result.assigned
    assert result.outcome == outcome
    assert result.assignment_id is None


def test_sync_plate_detection_raises_on_rejection(client):
    backend, session = client
    session.script["/api/assignments/plate-entry"] = FakeResponse(403, {}, text="forbidden")

    with pytest.raises(BackendError):
        backend.sync_plate_detection("ABC123")


def test_request_relogs_in_once_on_401(client):
    backend, session = client
    calls = {"n": 0}

    def resolve(url: str) -> FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(401, {}, text="expired")
        return FakeResponse(200, {"success": True, "data": {"plate": "ABC123", "active": True, "studentCode": "1"}})

    session._resolve = resolve  # type: ignore[assignment]
    backend._logged_in = True  # ya había una sesión "vieja"

    vehicle = backend.get_vehicle_by_plate("ABC123")

    assert vehicle["plate"] == "ABC123"
    assert session.login_attempts == 1


def test_concurrent_threads_never_send_a_stale_csrf_token(client):
    """El backend cambia la cookie CSRF en cada petición; dos hilos cruzados daban 403."""
    import threading
    import time

    backend, session = client
    counter = iter(range(1_000_000))
    mismatches = []

    def rotate_cookie():
        time.sleep(0.001)  # agranda la ventana en la que otro hilo podría colarse
        session.cookies["XSRF-TOKEN"] = f"csrf-{next(counter)}"

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        rotate_cookie()
        return FakeResponse(200, {})

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        time.sleep(0.002)  # viaje de red: aquí otro hilo podía cambiar la cookie
        if method == "POST" and kwargs["headers"]["X-XSRF-TOKEN"] != session.cookies["XSRF-TOKEN"]:
            mismatches.append(url)
        rotate_cookie()
        return FakeResponse(200, {"data": {"outcome": "NO_SPACE", "version": "", "cells": []}})

    session.get = fake_get
    session.request = fake_request
    backend._logged_in = True

    def report_occupancy():
        for _ in range(30):
            backend.report_occupancy(4, {1: True})

    def poll_cells():
        for _ in range(30):
            backend.get_camera_cells(4)

    workers = [threading.Thread(target=report_occupancy), threading.Thread(target=poll_cells)]
    for worker in workers:
        worker.start()
    for _ in range(30):
        backend.sync_plate_detection("ABC123")
    for worker in workers:
        worker.join()

    assert mismatches == []
