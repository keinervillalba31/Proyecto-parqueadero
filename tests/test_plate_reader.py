import numpy as np

from app.parking_detector import Vehicle
from app.plate_reader import PlateReader, edit_distance, vehicle_crops


def make_reader(confirmation_reads: int = 2, **kwargs) -> PlateReader:
    return PlateReader("workspace", "workflow", confirmation_reads, cooldown_seconds=30, **kwargs)


def test_normalize_removes_spaces_and_symbols():
    assert PlateReader.normalize(" abc-123 ") == "ABC123"
    assert PlateReader.normalize("---") is None
    assert PlateReader.normalize(None) is None


def test_validate_accepts_colombian_car_and_motorcycle_plates():
    reader = make_reader()
    assert reader.validate("ABC123") == "ABC123"
    assert reader.validate("ABC12D") == "ABC12D"


def test_validate_rejects_text_that_is_not_a_plate():
    # Lecturas reales de la prueba con el video: señales, letreros y basura.
    reader = make_reader()
    for text in ["100", "1", "PARKING", "MADEINCHINA", "BCDEFGHIJKLMNOPQRSTUVWXYZ", "50999"]:
        assert reader.validate(text) is None, text


def test_validate_fixes_confused_characters():
    reader = make_reader()
    assert reader.validate("A8C123") == "ABC123"  # 8 en posición de letra
    assert reader.validate("ABCI23") == "ABC123"  # I en posición de número
    # Demasiadas correcciones: se descarta en vez de inventar una placa.
    assert reader.validate("8801Z3") is None


def test_empty_pattern_accepts_everything():
    reader = make_reader(plate_pattern="")
    assert reader.validate("PARKING") == "PARKING"


def test_custom_pattern_for_other_countries():
    # Alemania: B-PZ 1081 -> BPZ1081
    reader = make_reader(plate_pattern=r"^[A-Z]{2,5}[0-9]{1,4}[EH]?$")
    assert reader.validate("BPZ1081") == "BPZ1081"
    assert reader.validate("100") is None


def test_extract_text_skips_invalid_texts():
    reader = make_reader()
    assert reader.extract_text([{"plate_text": ["100", "xyz 987"]}]) == "XYZ987"
    assert reader.extract_text({"plate_text": "ABC123"}) == "ABC123"
    assert reader.extract_text([{"plate_text": ["PARKING"]}]) is None
    assert reader.extract_texts([{"plate_text": ["100"]}, {"plate_text": "abc123"}]) == [
        "100",
        "ABC123",
    ]


def test_edit_distance():
    assert edit_distance("BRF509", "BRF509") == 0
    assert edit_distance("BRF509", "BRF508") == 1
    assert edit_distance("BRF509", "BRF50") == 1
    assert edit_distance("BRF509", "XYZ987") == 6


def test_confirm_accepts_reads_that_differ_in_one_character():
    reader = make_reader(confirmation_reads=3)
    assert reader.confirm("BRF509") is None
    assert reader.confirm("BRF508") is None
    assert reader.confirm("BRF509") == ("BRF509", 3)


def test_confirm_tie_picks_the_most_central_read():
    reader = make_reader(confirmation_reads=3, plate_pattern="")
    reader.confirm("BRF50")
    reader.confirm("RF509")
    assert reader.confirm("BRF509") == ("BRF509", 3)


def test_confirm_keeps_reads_from_other_vehicles():
    reader = make_reader(confirmation_reads=2)
    assert reader.confirm("ABC123") is None
    assert reader.confirm("XYZ987") is None
    assert reader.confirm("ABC123") == ("ABC123", 2)
    # La lectura de XYZ987 sigue guardada y se confirma con una más.
    assert reader.confirm("XYZ987") == ("XYZ987", 2)


def test_cooldown_only_blocks_the_confirmed_plate():
    reader = make_reader(confirmation_reads=1)
    assert reader.confirm("ABC123") == ("ABC123", 1)
    reader.start_cooldown("ABC123")
    assert reader.in_cooldown("ABC123")
    assert reader.in_cooldown("ABC128")  # casi igual: mismo vehículo
    assert reader.confirm("ABC123") is None
    # Un vehículo distinto se confirma de inmediato.
    assert not reader.in_cooldown("XYZ987")
    assert reader.confirm("XYZ987") == ("XYZ987", 1)


def test_vehicle_crops_prefers_the_largest_vehicles():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    small = Vehicle(2, 0.9, (0, 0, 100, 100))
    big = Vehicle(2, 0.9, (200, 200, 600, 500))
    medium = Vehicle(2, 0.9, (700, 700, 900, 900))
    parked = Vehicle(2, 0.9, (100, 600, 600, 990), in_cell=True)

    crops = vehicle_crops(frame, [small, big, medium, parked], 2, 150, skip_parked=True)
    # El pequeño no alcanza el ancho mínimo y el estacionado se omite.
    assert [crop.shape[:2] for crop in crops] == [(360, 480), (240, 240)]
