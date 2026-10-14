import numpy as np
import pytest

from app.parking_layout import apply, build_layout, rectify

# Cámara inventada que ve el suelo en diagonal: lo lejano queda arriba y más pequeño.
GROUND_TO_IMAGE = np.array([
    [38.0, 14.0, 640.0],
    [2.0, 9.0, 430.0],
    [0.0, -0.022, 1.0],
])


def stall(x: float, y: float, width: float, depth: float) -> np.ndarray:
    """Esquinas en la imagen de un puesto cuyo rectángulo real empieza en (x, y) metros."""
    corners = np.array([[x, y], [x + width, y], [x + width, y + depth], [x, y + depth]])
    return apply(GROUND_TO_IMAGE, corners)


def lot() -> list[np.ndarray]:
    """Dos columnas de carros atravesados (5 x 2,5 m) y una fila de fondo con carros de frente."""
    quads = [stall(x, y, 5.0, 2.5) for x in (-12.0, 4.0) for y in np.arange(-8.0, 9.5, 2.5)]
    quads += [stall(x, -14.0, 2.5, 5.0) for x in (-3.0, -0.5, 2.0)]
    return quads


def test_rectify_recovers_real_stall_sizes():
    quads = np.array(lot())
    _, rect = rectify(quads)
    widths, depths = rect[:, 1] - rect[:, 0], rect[:, 3] - rect[:, 2]

    columns, back_row = slice(0, 14), slice(14, 17)
    assert widths[columns] == pytest.approx(5.0, abs=0.15)
    assert depths[columns] == pytest.approx(2.5, abs=0.15)
    assert widths[back_row] == pytest.approx(2.5, abs=0.25)
    assert depths[back_row] == pytest.approx(5.0, abs=0.4)


def test_rectify_keeps_the_real_arrangement():
    _, rect = rectify(np.array(lot()))
    centers_x = (rect[:, 0] + rect[:, 1]) / 2
    centers_y = (rect[:, 2] + rect[:, 3]) / 2

    # Las dos columnas quedan rectas y separadas 16 m, como en el suelo.
    assert np.ptp(centers_x[0:7]) < 0.2 and np.ptp(centers_x[7:14]) < 0.2
    assert centers_x[7] - centers_x[0] == pytest.approx(16.0, abs=0.5)
    # Dentro de una columna los puestos van cada 2,5 m.
    assert np.diff(centers_y[0:7]) == pytest.approx(2.5, abs=0.12)
    # La fila del fondo queda más lejos de la cámara (y menor) que las columnas.
    assert centers_y[14:17].max() < centers_y[0:14].min()


def test_rectify_does_not_depend_on_the_order_the_corners_were_drawn():
    quads = np.array(lot())
    shuffled = np.array([np.roll(quad, i % 4, axis=0) for i, quad in enumerate(quads)])

    assert rectify(shuffled)[1] == pytest.approx(rectify(quads)[1], abs=1e-6)


def test_rectify_needs_enough_car_stalls():
    with pytest.raises(ValueError):
        rectify(np.array(lot()[:2]))


def test_build_layout_only_returns_cells_linked_to_a_space():
    width, height = 1280, 720
    cells = {
        f"C{i}": {"points": (quad / [width, height]).tolist(), "puesto_id": None if i == 3 else 100 + i}
        for i, quad in enumerate(lot())
    }

    layout = build_layout(cells, width, height)

    assert len(layout) == 16
    assert {cell["parkingSpaceId"] for cell in layout} == {100 + i for i in range(17)} - {103}
    first = next(cell for cell in layout if cell["label"] == "C0")
    assert first["x1"] - first["x0"] == pytest.approx(5.0, abs=0.15)
