"""Lector de placas local (sin API) y modo híbrido, con un motor falso para no cargar modelos."""

import numpy as np
import pytest

from app.parking_detector import Vehicle
from app.plate_reader import LocalRead, PlateReader, vehicle_crops_with_kinds

IMAGE = np.zeros((40, 40, 3), dtype=np.uint8)


class FakeLocal:
    def __init__(self, texts, found=None) -> None:
        self.texts = texts
        self.found = len(texts) if found is None else found
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return LocalRead(texts=list(self.texts), found=self.found)


class FakeClient:
    def __init__(self, result=None, error=None) -> None:
        self.result, self.error, self.calls = result, error, 0

    def run_workflow(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def make_reader(engine, local, client=None):
    reader = PlateReader("workspace", "workflow", 2, cooldown_seconds=30, engine=engine, local_engine=local)
    reader._cached_client = client
    return reader


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "clave")


def test_local_engine_reads_without_calling_roboflow(api_key):
    client = FakeClient(result=[{"plate_text": ["ZZZ999"]}])
    reader = make_reader("local", FakeLocal(["ABC123"]), client)

    assert reader.read_all(IMAGE) == ["ABC123"]
    assert client.calls == 0
    assert reader.stats == {"local": 1}


def test_hybrid_asks_roboflow_only_when_a_plate_was_seen_but_not_read(api_key):
    client = FakeClient(result=[{"plate_text": ["ABC123"]}])

    # Leyó bien: no se gasta la API.
    reader = make_reader("hibrido", FakeLocal(["XYZ987"]), client)
    assert reader.read_all(IMAGE) == ["XYZ987"]
    assert client.calls == 0

    # Vio una placa pero el OCR no dio nada válido: se pregunta a Roboflow.
    reader = make_reader("hibrido", FakeLocal(["12"], found=1), client)
    assert reader.read_all(IMAGE) == ["ABC123"]
    assert client.calls == 1
    assert reader.stats == {"local": 1, "roboflow": 1}

    # No vio ninguna placa: no hay nada que preguntar.
    reader = make_reader("hibrido", FakeLocal([], found=0), client)
    assert reader.read_all(IMAGE) == []
    assert client.calls == 1


def test_hybrid_keeps_working_when_roboflow_fails(api_key):
    reader = make_reader("hibrido", FakeLocal(["12"], found=1), FakeClient(error=RuntimeError("sin créditos")))
    assert reader.read_all(IMAGE) == []


def test_hybrid_without_roboflow_credentials_stays_local(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    client = FakeClient(result=[{"plate_text": ["ABC123"]}])
    reader = make_reader("hibrido", FakeLocal(["12"], found=1), client)

    assert reader.read_all(IMAGE) == []
    assert client.calls == 0


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError):
        PlateReader("", "", 2, cooldown_seconds=30, engine="otro")


def test_vehicle_kind_picks_the_plate_format():
    reader = PlateReader("", "", 2, cooldown_seconds=30, local_engine=FakeLocal([]))
    # Sin saber el vehículo, YZA916 es una placa de carro válida.
    assert reader.validate("YZA916") == "YZA916"
    # En una moto, la última debe ser letra: el 6 era una G.
    assert reader.validate("YZA916", "moto") == "YZA91G"
    # En un carro, la última debe ser número.
    assert reader.validate("ABC12S", "carro") == "ABC125"
    # Si el formato del vehículo exige cambiar más de un carácter, se valida como cualquier placa.
    assert reader.validate("ABC123", "moto") == "ABC123"


def test_vehicle_crops_carry_the_vehicle_kind():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    car = Vehicle(class_id=2, confidence=0.9, box=(0, 0, 300, 200))
    moto = Vehicle(class_id=3, confidence=0.9, box=(300, 0, 500, 200))

    kinds = [kind for _, kind in vehicle_crops_with_kinds(frame, [car, moto], 2, 100)]
    assert kinds == ["carro", "moto"]


def test_a_tie_waits_for_another_read_instead_of_keeping_the_first():
    reader = PlateReader("", "", 2, cooldown_seconds=30, local_engine=FakeLocal([]))
    # La primera lectura vino errada: con un empate 1 a 1 no se confirma todavía.
    assert reader.confirm("TZA91G") is None
    assert reader.confirm("YZA91G") is None
    # La tercera desempata a favor de la correcta.
    assert reader.confirm("YZA91G") == ("YZA91G", 3)


def test_a_tie_that_never_breaks_still_confirms_eventually():
    reader = PlateReader("", "", 2, cooldown_seconds=30, local_engine=FakeLocal([]))
    outcomes = [reader.confirm(plate) for plate in ["ABC123", "ABC124", "ABC125", "ABC126", "ABC127"]]
    assert outcomes[:4] == [None, None, None, None]
    assert outcomes[4] is not None
