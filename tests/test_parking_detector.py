import numpy as np

CAR, MOTORCYCLE = 2, 3
# Caja de 40x40 cuyo punto inferior central cae dentro de Celda_1.
CAR_IN_CELL_1 = (CAR, 0.9, (30, 50, 70, 90))


def blank_frame() -> np.ndarray:
    return np.zeros((120, 220, 3), dtype=np.uint8)


def run(detector, detections, reserved=None):
    detector.vehicles._detect_with_tiles = lambda frame: detections
    return detector.detect(blank_frame(), reserved)[1]


def test_vehicle_occupies_the_cell_under_its_footpoint(make_detector):
    state = run(make_detector(), [CAR_IN_CELL_1])
    assert state["celdas"]["Celda_1"]["ocupado"] is True
    assert state["celdas"]["Celda_2"]["ocupado"] is False
    assert state["ocupadas"] == 1
    assert state["congestion_porcentaje"] == 50.0


def test_small_boxes_are_ignored_except_motorcycles(make_detector):
    tiny_car = (CAR, 0.9, (30, 80, 45, 90))  # 150 px² < 500
    tiny_moto = (MOTORCYCLE, 0.9, (130, 80, 145, 90))  # 150 px² >= 100
    state = run(make_detector(), [tiny_car, tiny_moto])
    assert state["vehiculos_detectados"] == 1
    assert state["celdas"]["Celda_1"]["ocupado"] is False
    assert state["celdas"]["Celda_2"]["ocupado"] is True


def test_smoothing_ignores_a_single_missed_frame(make_detector):
    detector = make_detector(occupancy_frames=3)
    for _ in range(3):
        state = run(detector, [CAR_IN_CELL_1])
    assert state["celdas"]["Celda_1"]["ocupado"] is True

    # Un cuadro sin detección no libera la celda...
    assert run(detector, [])["celdas"]["Celda_1"]["ocupado"] is True
    assert run(detector, [CAR_IN_CELL_1])["celdas"]["Celda_1"]["ocupado"] is True
    # ...pero tres seguidos sí.
    for _ in range(3):
        state = run(detector, [])
    assert state["celdas"]["Celda_1"]["ocupado"] is False


def test_reserved_free_cell_is_reported(make_detector):
    detector = make_detector()
    state = run(detector, [], reserved={"Celda_2": "ABC123"})
    assert state["celdas"]["Celda_2"]["estado"] == "reservado"
    assert detector.first_free_space(state, {"Celda_2"}) == "Celda_1"


def test_tile_starts_cover_the_whole_axis(make_detector):
    assert make_detector(tile_grid=2).vehicles._tile_starts(1000, 675) == [0, 325]
    assert make_detector(tile_grid=3).vehicles._tile_starts(900, 405) == [0, 248, 495]
