import numpy as np
import pytest

from app.calibration import layout_from_rows
from app.parking_layout import apply

from test_calibration import CAMERAS

# Filas en el suelo (metros): esquinas en el orden de marcado del editor, puestos de carro
# y motos que siguen la fila por fuera de lo marcado. C va girada 90 grados respecto a A y B.
ROWS = [
    {"corners": [[0, 0], [25, 0], [25, -5], [0, -5]], "count": 10, "motosEnd": 2},
    {"corners": [[0, 7], [20, 7], [20, 12], [0, 12]], "count": 8, "motosEnd": 0},
    {"corners": [[-3, -8], [-3, 12], [-8, 12], [-8, -8]], "count": 8, "motosEnd": 0},
]
MOTO_WIDTH = 1.0


def stalls(corners, count, motos_end):
    """Puestos reales de la fila en el suelo: (tipo, cuadrilátero)."""
    a, b, _, d = (np.asarray(p, float) for p in corners)
    along = (b - a) / np.linalg.norm(b - a)
    inward = (d - a) / np.linalg.norm(d - a)
    widths = [("CAR", 2.5)] * count + [("MOTORCYCLE", MOTO_WIDTH)] * motos_end
    quads, position = [], 0.0
    for kind, width in widths:
        start = a + along * position
        quads.append((kind, np.array([start, start + along * width, start + along * width + inward * 5, start + inward * 5])))
        position += width
    return quads


def editor_data(view):
    """Filas y celdas como las guarda la plataforma, vistas por la cámara `view`."""
    rows, cells, truth = [], {}, {}
    for index, row in enumerate(ROWS):
        rows.append({
            "corners": apply(view, np.array(row["corners"], float)).tolist(),
            "type": "CAR", "count": row["count"], "angle": 90,
            "motosStart": 0, "motosEnd": row["motosEnd"], "motoWidth": MOTO_WIDTH,
        })
        for number, (kind, quad) in enumerate(stalls(row["corners"], row["count"], row["motosEnd"])):
            label = f"{index}-{number}"
            cells[label] = {"points": apply(view, quad).tolist(), "puesto_id": len(cells) + 1, "fila": index}
            truth[label] = (kind, quad)
    return rows, cells, truth


def centers(layout):
    return {cell["label"]: np.array([(cell["x0"] + cell["x1"]) / 2, (cell["y0"] + cell["y1"]) / 2]) for cell in layout}


@pytest.mark.parametrize("name", CAMERAS)
def test_each_stall_gets_its_real_size(name):
    rows, cells, truth = editor_data(CAMERAS[name])
    layout = layout_from_rows(cells, rows)

    assert len(layout) == len(cells)
    for cell in layout:
        kind, _ = truth[cell["label"]]
        size = sorted([cell["x1"] - cell["x0"], cell["y1"] - cell["y0"]])
        expected = sorted([2.5 if kind == "CAR" else MOTO_WIDTH, 5.0])
        assert size == pytest.approx(expected, abs=0.05), cell["label"]


@pytest.mark.parametrize("name", CAMERAS)
def test_the_plan_keeps_the_real_distances_between_stalls(name):
    rows, cells, truth = editor_data(CAMERAS[name])
    plan = centers(layout_from_rows(cells, rows))

    real = {label: quad.mean(axis=0) for label, (_, quad) in truth.items()}
    labels = list(real)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            assert np.linalg.norm(plan[a] - plan[b]) == pytest.approx(np.linalg.norm(real[a] - real[b]), abs=0.05)


def test_stalls_do_not_overlap():
    rows, cells, _ = editor_data(CAMERAS["esquina"])
    layout = layout_from_rows(cells, rows)

    for i, a in enumerate(layout):
        for b in layout[i + 1:]:
            overlap_x = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            overlap_y = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            assert overlap_x <= 0.03 or overlap_y <= 0.03, (a["label"], b["label"])


def test_hand_drawn_cells_and_cells_without_space_are_handled():
    view = CAMERAS["atras"]
    rows, cells, _ = editor_data(view)
    # Una celda trazada a mano fuera de las filas y otra sin puesto vinculado.
    cells["suelta"] = {"points": apply(view, np.array([[30, 7], [32.5, 7], [32.5, 12], [30, 12]], float)).tolist(),
                       "puesto_id": 999, "fila": None}
    cells["sin-puesto"] = {**cells["0-0"], "puesto_id": None}
    layout = {cell["label"]: cell for cell in layout_from_rows(cells, rows)}

    assert "sin-puesto" not in layout
    loose = layout["suelta"]
    assert sorted([loose["x1"] - loose["x0"], loose["y1"] - loose["y0"]]) == pytest.approx([2.5, 5.0], abs=0.1)


def test_without_rows_it_falls_back():
    _, cells, _ = editor_data(CAMERAS["atras"])
    assert layout_from_rows(cells, []) is None
    assert layout_from_rows(cells, [{"corners": "mal"}]) is None
