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


def test_a_car_behind_another_in_the_full_frame_is_kept():
    # En una fila vista en perspectiva, la caja del carro de atrás queda casi entera dentro de la del de
    # adelante: si las dos salen del cuadro completo son dos carros, no una caja parcial.
    from app.parking_detector import _drop_partial_boxes

    front = (CAR, 0.9, (40, 20, 150, 90))
    behind = (CAR, 0.7, (60, 15, 140, 60))  # 89 % dentro de la de adelante
    assert sorted(_drop_partial_boxes([front, behind], [False, False])) == sorted([front, behind])
    assert _drop_partial_boxes([front, behind], [False, True]) == [front]
    # Pero una caja de la parte de arriba del mismo carro, del todo dentro, sí es un duplicado.
    roof = (CAR, 0.6, (45, 22, 120, 55))
    assert _drop_partial_boxes([front, roof], [False, False]) == [front]


def test_the_far_cells_get_a_zoomed_second_look():
    from app.parking_detector import far_zone

    def cell(x, y, w, h):
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    near = [cell(100, 500, 200, 120), cell(320, 500, 200, 120)]
    far = [cell(500, 300, 50, 20), cell(560, 300, 50, 20)]
    x0, y0, x1, y1 = far_zone(near + far, 1280, 720)
    assert x0 <= 500 and x1 >= 610 and y0 < 300 and y1 >= 320
    assert x1 - x0 < 300  # solo las lejanas: así YOLO las amplía
    # Todas del mismo tamaño (cámara cenital): no hay nada que ampliar.
    assert far_zone(near, 1280, 720) is None
    assert far_zone([], 1280, 720) is None


def test_a_vehicle_cut_by_the_border_of_the_zoomed_zone_is_ignored():
    from types import SimpleNamespace

    from app.parking_detector import VehicleDetector

    def boxes(*items):
        return [SimpleNamespace(boxes=[
            SimpleNamespace(cls=[cls], conf=[0.8], xyxy=[np.array(box, float)]) for cls, box in items
        ])]

    calls = []

    def model(source, **options):
        calls.append(source.shape[:2])
        if len(calls) == 1:
            return boxes()  # el cuadro completo no ve nada
        # En la zona: una moto entera y un carro cortado por el borde izquierdo.
        return boxes((MOTORCYCLE, (40, 20, 60, 45)), (CAR, (0, 10, 30, 50)))

    detector = VehicleDetector(None, 0.1, 1280, 1, 150, 50, model=model)
    detector.zoom_zones = [(500, 300, 700, 400)]
    vehicles = detector.detect(np.zeros((720, 1280, 3), np.uint8))
    assert calls == [(720, 1280), (100, 200)]
    assert [(v.class_id, v.box) for v in vehicles] == [(MOTORCYCLE, (540, 320, 560, 345))]


# --- Reglas contra falsos positivos ---


def test_a_vehicle_must_stand_still_before_it_takes_a_space(make_detector):
    clock = Clock()
    detector = make_detector(clock=clock, still_seconds=3)
    assert occupied_over(detector, clock, [[CAR_IN_CELL_1]] * 5) == [False, False, False, True, True]


def test_a_vehicle_driving_through_the_cell_never_takes_it(make_detector):
    # Un carro que cruza la celda: su punto de apoyo avanza 12 px por segundo.
    clock = Clock()
    detector = make_detector(clock=clock, still_seconds=3)
    moving = [[(CAR, 0.9, (5 + 12 * i, 50, 45 + 12 * i, 90))] for i in range(6)]
    assert not any(occupied_over(detector, clock, moving))


def test_box_jitter_and_short_gaps_do_not_restart_the_stillness_count(make_detector):
    clock = Clock()
    detector = make_detector(clock=clock, still_seconds=3)
    jitter = [[(CAR, 0.9, (31, 50, 71, 91))], [], [(CAR, 0.9, (29, 49, 69, 89))], [CAR_IN_CELL_1]]
    assert occupied_over(detector, clock, jitter)[-1] is True


def test_near_cells_are_freed_sooner_than_far_ones(make_detector, tmp_path):
    from app import parking_detector

    clock = Clock()
    detector = make_detector(clock=clock, free_seconds=8, free_seconds_near=2)
    # Celda_2 pasa a ser lejana: mucho más chica que Celda_1.
    detector.set_relative_cells({
        "Celda_1": {"points": [[0, 0], [0.45, 0], [0.45, 0.8], [0, 0.8]]},
        "Celda_2": {"points": [[0.6, 0.1], [0.7, 0.1], [0.7, 0.3], [0.6, 0.3]]},
    })
    near = (CAR, 0.9, (30, 50, 70, 90))
    far = (MOTORCYCLE, 0.9, (140, 25, 150, 35))
    run(detector, [near, far])
    states = []
    for _ in range(4):
        clock.now += 1
        states.append(run(detector, [])["celdas"])
    assert [s["Celda_1"]["ocupado"] for s in states] == [True, True, False, False]
    assert all(s["Celda_2"]["ocupado"] for s in states)
    assert parking_detector.far_cells([np.array(c["points"]) for c in detector.cells.values()]) == [1]


def test_the_footpoint_is_lifted_out_of_the_space_in_front(make_detector):
    # La caja llega 6 px dentro de la celda de abajo (sombra y defensa); subida un 15 % queda en la suya.
    cells = {
        "Atras": {"points": [[0, 0], [1, 0], [1, 0.5], [0, 0.5]]},
        "Adelante": {"points": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]},
    }
    box = (CAR, 0.9, (50, 26, 110, 66))  # 220x120: la mitad es y = 60
    plain, lifted = make_detector(), make_detector(footpoint_lift=0.15)
    for detector in (plain, lifted):
        detector.set_relative_cells(cells)
    assert run(plain, [box])["celdas"]["Adelante"]["ocupado"] is True
    state = run(lifted, [box])["celdas"]
    assert state["Atras"]["ocupado"] is True and state["Adelante"]["ocupado"] is False


def test_a_vehicle_on_a_shared_edge_occupies_only_one_cell(make_detector):
    # El punto de apoyo cae justo en el borde x = 100 entre Celda_1 y Celda_2.
    state = run(make_detector(), [(CAR, 0.9, (80, 60, 120, 90))])
    assert sum(cell["ocupado"] for cell in state["celdas"].values()) == 1
