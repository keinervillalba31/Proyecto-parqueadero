import numpy as np
import pytest

from app.calibration import Row, Stall, calibrate, suggested_corners, top_down
from app.parking_layout import apply

WIDTH, HEIGHT = 1280, 720


def camera(position, target, focal=900.0) -> np.ndarray:
    """Homografía suelo (m) -> imagen relativa de una cámara en `position` que mira a `target`."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    translation = -rotation @ position
    intrinsics = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]])
    to_relative = np.diag([1 / WIDTH, 1 / HEIGHT, 1])
    return to_relative @ intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])


# Vistas distintas del mismo parqueadero (centrado cerca de (12, 2)): desde atrás, girada
# 90 grados, desde una esquina y casi cenital.
POSITIONS = {
    "atras": ([12, -30, 12], 900.0),
    "lado": ([50, 2, 10], 900.0),
    "esquina": ([-20, -25, 14], 900.0),
    "cenital": ([12, -6, 35], 700.0),
}
CAMERAS = {name: camera(position, [12, 2, 0], focal) for name, (position, focal) in POSITIONS.items()}

# Filas reales en el suelo: esquinas en el orden de marcado y sus puestos.
CARS = [Stall(f"A-{i:02d}", "CAR") for i in range(1, 11)]
FACING = [Stall(f"B-{i:02d}", "CAR") for i in range(1, 9)]
COLUMN = [Stall(f"C-{i:02d}", "CAR") for i in range(1, 9)]
MIXED = [Stall(f"M-{i:02d}", "MOTORCYCLE") for i in range(1, 5)] + [Stall("D-01", "CAR"), Stall("D-02", "CAR")]
GROUND_ROWS = [
    ("A", [[0, 0], [25, 0], [25, -5], [0, -5]], CARS),
    ("B", [[0, 7], [20, 7], [20, 12], [0, 12]], FACING),
    ("C", [[-3, -8], [-3, 12], [-8, 12], [-8, -8]], COLUMN),
    ("D", [[28, -5], [28, 4], [33, 4], [33, -5]], MIXED),
]


def stall_corners(front_start, front_end, back_start, stalls) -> list[np.ndarray]:
    """Puestos perpendiculares reales de una fila, repartidos según su ancho."""
    front_start, front_end, back_start = (np.asarray(p, float) for p in (front_start, front_end, back_start))
    along = (front_end - front_start) / np.linalg.norm(front_end - front_start)
    inward = (back_start - front_start) / np.linalg.norm(back_start - front_start)
    widths = {"CAR": 2.5, "MOTORCYCLE": 1.0}
    lengths = {"CAR": 5.0, "MOTORCYCLE": 2.5}
    quads, position = [], 0.0
    for stall in stalls:
        a = front_start + along * position
        b = a + along * widths[stall.type]
        quads.append(np.array([a, b, b + inward * lengths[stall.type], a + inward * lengths[stall.type]]))
        position += widths[stall.type]
    return quads


def marked_rows(view: np.ndarray, noise_px: float = 0.0, seed: int = 0) -> list[Row]:
    rng = np.random.default_rng(seed)
    rows = []
    for label, corners, stalls in GROUND_ROWS:
        image = apply(view, np.array(corners, float))
        image = image + rng.normal(0, noise_px, image.shape) / [WIDTH, HEIGHT]
        rows.append(Row(label, image, tuple(stalls)))
    return rows


def true_cells(view: np.ndarray) -> dict[str, np.ndarray]:
    cells = {}
    for _, corners, stalls in GROUND_ROWS:
        for stall, quad in zip(stalls, stall_corners(corners[0], corners[1], corners[3], stalls)):
            cells[stall.label] = apply(view, quad)
    return cells


def sizes(calibration) -> dict[str, tuple[float, float]]:
    """Ancho (frente) y largo (hacia el fondo) de cada celda generada, en metros."""
    return {
        cell.label: (
            float(np.linalg.norm(cell.ground[1] - cell.ground[0])),
            float(np.linalg.norm(cell.ground[3] - cell.ground[0])),
        )
        for cell in calibration.cells
    }


@pytest.mark.parametrize("name", CAMERAS)
def test_the_generated_cells_match_the_real_stalls_from_any_view(name):
    view = CAMERAS[name]
    calibration = calibrate(marked_rows(view))

    expected = true_cells(view)
    assert len(calibration.cells) == len(expected)
    for cell in calibration.cells:
        # Media milésima de la imagen: menos de un píxel.
        assert cell.image == pytest.approx(expected[cell.label], abs=5e-4), cell.label
    assert calibration.verified
    assert calibration.error_m < 0.01


@pytest.mark.parametrize("name", CAMERAS)
def test_hand_marked_rows_still_put_each_stall_in_place(name):
    view = CAMERAS[name]
    calibration = calibrate(marked_rows(view, noise_px=1.5, seed=7))

    # El centro real de cada puesto, llevado al plano con la calibración, queda a menos de un cuarto
    # del ancho del puesto del centro de su celda, aun en filas a 50 m de la cámara.
    expected = true_cells(view)
    for cell in calibration.cells:
        real_center = apply(calibration.homography, expected[cell.label]).mean(axis=0)
        assert np.linalg.norm(real_center - cell.ground.mean(axis=0)) < 0.6, cell.label
    # Las filas quedan donde están en el suelo: A y B a 7 m de pasillo.
    a, b = calibration.rows[0].ground, calibration.rows[1].ground
    assert np.linalg.norm(b[0] - a[0]) == pytest.approx(7.0, abs=0.35)
    assert calibration.error_m < 0.3


def test_one_row_is_enough_to_calibrate_but_not_to_verify():
    view = CAMERAS["esquina"]
    calibration = calibrate(marked_rows(view)[:1])

    assert not calibration.verified
    # Con la cámara calibrada, las filas no marcadas también caen en su lugar.
    other = np.array(GROUND_ROWS[1][1], float)
    distance = np.linalg.norm(np.diff(apply(calibration.homography, apply(view, other))[:3], axis=0), axis=1)
    assert distance == pytest.approx([20.0, 5.0], abs=1e-3)


def test_the_plan_is_the_lot_seen_from_above_not_its_mirror():
    view = CAMERAS["lado"]
    calibration = calibrate(marked_rows(view))

    # Visto desde arriba con y hacia abajo en pantalla, el suelo real (y hacia el fondo) es (x, -y).
    real = np.array([np.mean(q, axis=0) for _, c, s in GROUND_ROWS for q in stall_corners(c[0], c[1], c[3], s)])
    real[:, 1] *= -1
    fitted = np.array([np.mean(cell.ground, axis=0) for cell in calibration.cells])
    real, fitted = real - real.mean(axis=0), fitted - fitted.mean(axis=0)
    u, _, vt = np.linalg.svd(fitted.T @ real)
    rotation = u @ vt
    assert np.linalg.det(rotation) > 0
    assert fitted @ rotation == pytest.approx(real, abs=1e-3)


@pytest.mark.parametrize("name", ["atras", "lado", "esquina"])
def test_the_plan_puts_what_is_near_the_camera_at_the_bottom(name):
    calibration = calibrate(marked_rows(CAMERAS[name]))

    camera_xy = np.array(POSITIONS[name][0][:2], float)
    distance = {label: np.linalg.norm(np.mean(corners, axis=0) - camera_xy) for label, corners, _ in GROUND_ROWS}
    plan_y = {row.label: row.ground[:, 1].mean() for row in calibration.rows}
    nearest, farthest = min(distance, key=distance.get), max(distance, key=distance.get)
    assert plan_y[nearest] == max(plan_y.values())
    assert plan_y[farthest] == min(plan_y.values())


def test_a_row_with_the_wrong_number_of_stalls_stands_out():
    rows = marked_rows(CAMERAS["atras"], noise_px=1.0)
    rows[1] = Row("B", rows[1].corners, rows[1].stalls[:6])   # tiene 8 puestos, se dijo que 6

    calibration = calibrate(rows)

    errors = {row.label: row.error_m for row in calibration.rows}
    assert errors["B"] > 1.0
    assert max(errors[label] for label in "ACD") < errors["B"] / 3


def test_marking_a_row_from_the_other_end_gives_the_same_cells():
    view = CAMERAS["esquina"]
    rows = marked_rows(view)
    a = rows[0]
    reversed_a = Row("A", a.corners[[1, 0, 3, 2]], tuple(reversed(a.stalls)))

    original = {cell.label: cell.image for cell in calibrate(rows).cells}
    flipped = {cell.label: cell.image for cell in calibrate([reversed_a, *rows[1:]]).cells}

    for label, quad in original.items():
        # Las mismas cuatro esquinas, aunque empiecen por otra.
        assert sorted(map(tuple, np.round(flipped[label], 4))) == sorted(map(tuple, np.round(quad, 4)))


def test_angled_stalls():
    view = CAMERAS["atras"]
    angle = np.radians(60)
    width, length = 2.5, 5.0
    pitch, shift, depth = width / np.sin(angle), length * np.cos(angle), length * np.sin(angle)
    count = 6
    total = count * pitch + shift
    ground = np.array([[0, 0], [total, 0], [total, -depth], [0, -depth]])
    stalls = tuple(Stall(f"E-{i}") for i in range(count))
    # Diagonales: el fondo de cada puesto corrido `shift` hacia el final de la fila.
    expected = [
        apply(view, np.array([[i * pitch, 0], [(i + 1) * pitch, 0], [(i + 1) * pitch + shift, -depth], [i * pitch + shift, -depth]]))
        for i in range(count)
    ]

    calibration = calibrate([Row("E", apply(view, ground), stalls, angle=60)])

    for cell, quad in zip(calibration.cells, expected):
        assert cell.image == pytest.approx(quad, abs=5e-4)


@pytest.mark.parametrize(
    "corners, stalls, message",
    [
        ([[0.1, 0.1], [0.5, 0.1], [0.5, 0.3]], (Stall("X"),), "cuatro esquinas"),
        ([[0.1, 0.1], [0.5, 0.3], [0.5, 0.1], [0.1, 0.3]], (Stall("X"),), "cuadrilátero"),
        ([[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.1, 0.3]], (), "no tiene puestos"),
        ([[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.1, 0.3]], (Stall("X", "BUS"),), "sin medidas"),
    ],
)
def test_invalid_rows_are_rejected(corners, stalls, message):
    with pytest.raises(ValueError, match=message):
        calibrate([Row("X", np.array(corners, float), stalls)])


def test_custom_stall_sizes():
    view = CAMERAS["atras"]
    ground = np.array([[0, 0], [10, 0], [10, -4], [0, -4]], float)   # 4 puestos de 2,5 x 4 m

    calibration = calibrate(
        [Row("F", apply(view, ground), tuple(Stall(f"F-{i}") for i in range(4)))],
        sizes={"CAR": (2.5, 4.0)},
    )

    assert sizes(calibration)["F-0"] == pytest.approx((2.5, 4.0))
    assert calibration.cells[0].image == pytest.approx(apply(view, np.array([[0, 0], [2.5, 0], [2.5, -4], [0, -4]], float)), abs=5e-4)


def test_top_down_view_places_the_ground_where_the_matrix_says():
    view = CAMERAS["esquina"]
    calibration = calibrate(marked_rows(view))
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    # Una marca blanca en el centro del primer puesto de la fila A.
    target = np.mean(stall_corners([0, 0], [25, 0], [0, -5], CARS)[0], axis=0)
    x, y = (apply(view, target[None])[0] * [WIDTH, HEIGHT]).astype(int)
    frame[y - 3:y + 4, x - 3:x + 4] = 255

    image, meters_to_pixels = top_down(frame, calibration, max_side=800)

    assert max(image.shape[:2]) == pytest.approx(800, abs=1)
    cell = next(c for c in calibration.cells if c.label == "A-01")
    px, py = apply(meters_to_pixels, np.mean(cell.ground, axis=0)[None])[0].astype(int)
    assert image[py, px].mean() > 200


@pytest.mark.parametrize("name", CAMERAS)
def test_fitting_a_badly_marked_row_puts_its_corners_back_in_place(name):
    view = CAMERAS[name]
    rows = marked_rows(view)
    # La esquina 3 de la fila B se marca corrida: una punta de la fila queda más grande que la otra.
    bad = rows[1].corners.copy()
    bad[2] += (bad[2] - bad[1]) * 0.35
    rows[1] = Row(rows[1].label, bad, rows[1].stalls)

    calibration = calibrate(rows)
    fitted = suggested_corners(rows)
    real = apply(view, np.array(GROUND_ROWS[1][1], float))

    # El error se nota en esa fila, y la sugerencia devuelve sus esquinas a donde están en realidad.
    assert calibration.rows[1].error_m > 0.3
    assert fitted["B"].corners == pytest.approx(real, abs=1e-3)
    assert fitted["B"].deviation_m > 1.0
    # Las filas bien marcadas no se mueven.
    for row in rows[:1] + rows[2:]:
        assert fitted[row.label].corners == pytest.approx(row.corners, abs=1e-3), row.label
        assert fitted[row.label].deviation_m < 0.05, row.label


def test_with_a_single_row_there_is_nothing_to_compare():
    assert suggested_corners(marked_rows(CAMERAS["atras"])[:1]) == {}


@pytest.mark.parametrize("name", list(CAMERAS))
def test_the_ground_has_positive_homogeneous_coordinate(name):
    # La esquina (0, 0) de la imagen suele ser cielo: si la escala dejara negativo el suelo,
    # el editor descartaría todos los puntos como si estuvieran detrás de la cámara.
    calibration = calibrate(marked_rows(CAMERAS[name]))
    corners = np.vstack([row.corners for row in marked_rows(CAMERAS[name])])
    w = corners @ calibration.homography[2, :2] + calibration.homography[2, 2]
    assert np.all(w > 0)


def test_top_down_view_fills_what_the_camera_does_not_see():
    view = CAMERAS["esquina"]
    calibration = calibrate(marked_rows(view))
    frame = np.full((HEIGHT, WIDTH, 3), 120, np.uint8)

    image, _ = top_down(frame, calibration, max_side=600)

    # Sin triángulos negros: lo que queda fuera se prolonga más oscuro, no en negro.
    assert image.min() > 30
