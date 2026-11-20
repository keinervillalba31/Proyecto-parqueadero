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


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def occupied_over(detector, clock, frames, seconds_per_frame=1.0):
    """Analiza un cuadro por segundo con las detecciones dadas y devuelve si Celda_1 quedó ocupada en cada uno."""
    result = []
    for detections in frames:
        result.append(run(detector, detections)["celdas"]["Celda_1"]["ocupado"])
        clock.now += seconds_per_frame
    return result


def test_a_car_counts_as_parked_only_after_being_seen_for_a_while(make_detector):
    clock = Clock()
    detector = make_detector(occupy_seconds=2, free_seconds=8, clock=clock)
    assert occupied_over(detector, clock, [[CAR_IN_CELL_1]] * 4) == [False, False, True, True]


def test_a_car_detected_only_now_and_then_keeps_its_space(make_detector):
    # Como un carro del fondo que YOLO pierde y recupera: el puesto no parpadea.
    clock = Clock()
    detector = make_detector(occupy_seconds=2, free_seconds=8, clock=clock)
    occupied_over(detector, clock, [[CAR_IN_CELL_1]] * 3)
    flickering = [[], [], [CAR_IN_CELL_1], [], [], [], [CAR_IN_CELL_1], []] * 3
    assert all(occupied_over(detector, clock, flickering))


def test_a_space_is_freed_after_the_car_is_gone_for_long_enough(make_detector):
    clock = Clock()
    detector = make_detector(occupy_seconds=2, free_seconds=8, clock=clock)
    occupied_over(detector, clock, [[CAR_IN_CELL_1]] * 3)
    states = occupied_over(detector, clock, [[]] * 10)
    assert states[:8] == [True] * 8
    assert states[8:] == [False, False]


def test_a_passing_car_does_not_take_the_space(make_detector):
    clock = Clock()
    detector = make_detector(occupy_seconds=2, free_seconds=8, clock=clock)
    passing = [[], [CAR_IN_CELL_1], [], [CAR_IN_CELL_1], []]
    assert not any(occupied_over(detector, clock, passing))


def test_the_timing_does_not_depend_on_how_fast_frames_are_analyzed(make_detector):
    clock = Clock()
    detector = make_detector(occupy_seconds=2, free_seconds=8, clock=clock)
    # A 10 cuadros por segundo hacen falta los mismos 2 segundos, no 3 cuadros.
    states = occupied_over(detector, clock, [[CAR_IN_CELL_1]] * 25, seconds_per_frame=0.1)
    assert states.index(True) == 20


def test_reserved_free_cell_is_reported(make_detector):
    detector = make_detector()
    state = run(detector, [], reserved={"Celda_2": "ABC123"})
    assert state["celdas"]["Celda_2"]["estado"] == "reservado"
    assert detector.first_free_space(state, {"Celda_2"}) == "Celda_1"


def test_tile_starts_cover_the_whole_axis(make_detector):
    assert make_detector(tile_grid=2).vehicles._tile_starts(1000, 675) == [0, 325]
    assert make_detector(tile_grid=3).vehicles._tile_starts(900, 405) == [0, 248, 495]


def test_cells_follow_the_image_when_the_camera_moves(make_detector):
    class FixedMove:
        source = "guardada"
        # La cámara se corrió 100 px a la derecha: lo que ahora está en x vuelve a x - 100.
        last_transform = np.array([[1.0, 0.0, -100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        last_shift = (100.0, 0.0)

        def transform(self, frame):
            return self.last_transform

        def consider(self, frame, fit):
            pass

        def cells_changed(self):
            pass

    detector = make_detector()
    detector.aligner = FixedMove()
    # El carro de Celda_1 aparece donde estaba Celda_2.
    shifted_car = (CAR, 0.9, (130, 50, 170, 90))
    state = run(detector, [shifted_car])
    assert state["celdas"]["Celda_1"]["ocupado"] is True
    assert state["celdas"]["Celda_2"]["ocupado"] is False
    assert state["alineacion"] == {"referencia": "guardada", "compensando": True, "dx": 100.0, "dy": 0.0}


def test_a_partial_box_of_a_car_cut_by_a_tile_is_not_a_second_car(make_detector):
    # Caja completa en Celda_1 y la parte que vio un recorte, con su punto de apoyo en Celda_2.
    full = (CAR, 0.9, (40, 20, 150, 90))
    partial = (CAR, 0.6, (105, 25, 150, 88))
    # Dos carros vecinos de verdad se superponen poco y se mantienen.
    neighbour = (CAR, 0.8, (150, 30, 200, 95))
    from app.parking_detector import _drop_partial_boxes

    assert _drop_partial_boxes([partial, full]) == [full]
    assert sorted(_drop_partial_boxes([full, neighbour])) == sorted([full, neighbour])
