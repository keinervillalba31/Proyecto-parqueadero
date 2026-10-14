"""Plano de planta del parqueadero a partir de las celdas trazadas sobre la cámara.

La cámara ve el parqueadero en diagonal. Para el mapa 3D de la plataforma hace
falta saber dónde queda cada puesto visto desde arriba y en metros, así que se
"endereza" la imagen:

1. Los bordes de las celdas siguen dos direcciones del suelo (a lo ancho y a lo
   largo) y en la foto convergen a dos puntos de fuga: con ellos se obtiene una
   primera transformación.
2. Se afina pidiendo que cada celda quede como un rectángulo alineado y que los
   carros midan lo mismo estén cerca o lejos de la cámara.
3. La escala se fija con medidas reales: un carro mide unos 5 m de largo y los
   puestos de una columna están cada 2,5 m.

Los trazos a mano tienen error; con muchas celdas el ajuste lo promedia.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

CAR_LENGTH_M = 5.0
STALL_PITCH_M = 2.5
# Con menos celdas de carro "atravesadas" no hay con qué fijar la perspectiva.
MINIMUM_COLUMN_CELLS = 4


def _line(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    line = np.cross([*a, 1.0], [*b, 1.0])
    return line / np.linalg.norm(line[:2])


def _vanishing_point(lines: list[np.ndarray]) -> np.ndarray:
    """Punto que mejor cumple estar sobre todas las rectas (mínimos cuadrados)."""
    _, _, vt = np.linalg.svd(np.array(lines))
    return vt[-1]


def apply(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.c_[points, np.ones(len(points))] @ h.T
    return projected[:, :2] / projected[:, 2:3]


def ordered(quad: np.ndarray) -> np.ndarray:
    """Esquinas en orden fijo: arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda."""
    by_y = quad[np.argsort(quad[:, 1])]
    top, bottom = by_y[:2], by_y[2:]
    top, bottom = top[np.argsort(top[:, 0])], bottom[np.argsort(bottom[:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]])


def _initial_homography(quads: np.ndarray) -> np.ndarray:
    across = [_line(q[a], q[b]) for q in quads for a, b in ((0, 1), (3, 2))]
    along = [_line(q[a], q[b]) for q in quads for a, b in ((0, 3), (1, 2))]
    vx, vy = _vanishing_point(across), _vanishing_point(along)

    horizon = np.cross(vx, vy)
    horizon = horizon / horizon[2]
    projective = np.array([[1, 0, 0], [0, 1, 0], horizon])
    dx, dy = (projective @ vx)[:2], (projective @ vy)[:2]
    affine = np.eye(3)
    affine[:2, :2] = np.linalg.inv(np.column_stack([dx / np.linalg.norm(dx), dy / np.linalg.norm(dy)]))
    return affine @ projective


def _fit_homography(quads: np.ndarray, in_column: np.ndarray) -> np.ndarray:
    start = _initial_homography(quads)
    first = apply(start, quads.reshape(-1, 2)).reshape(-1, 4, 2)
    widths = np.abs(first[:, 1, 0] - first[:, 0, 0])
    depths = np.abs(first[:, 3, 1] - first[:, 0, 1])
    start = np.diag([
        CAR_LENGTH_M / np.median(widths[in_column]) * np.sign(first[0, 1, 0] - first[0, 0, 0]),
        STALL_PITCH_M / np.median(depths[in_column]) * np.sign(first[0, 3, 1] - first[0, 0, 1]),
        1.0,
    ]) @ start
    start = start / start[2, 2]

    def residuals(params: np.ndarray) -> np.ndarray:
        h = np.append(params, 1.0).reshape(3, 3)
        p = apply(h, quads.reshape(-1, 2)).reshape(-1, 4, 2)
        aligned = np.concatenate([
            p[:, 0, 1] - p[:, 1, 1], p[:, 3, 1] - p[:, 2, 1],  # bordes de arriba y abajo, horizontales
            p[:, 0, 0] - p[:, 3, 0], p[:, 1, 0] - p[:, 2, 0],  # bordes laterales, verticales
        ])
        c = p[in_column]
        width = (c[:, 1, 0] + c[:, 2, 0] - c[:, 0, 0] - c[:, 3, 0]) / 2
        depth = (c[:, 3, 1] + c[:, 2, 1] - c[:, 0, 1] - c[:, 1, 1]) / 2
        return np.concatenate([aligned, width - CAR_LENGTH_M, 0.5 * (depth - STALL_PITCH_M)])

    fit = least_squares(residuals, start.flatten()[:8], loss="soft_l1", f_scale=0.5, x_scale="jac")
    return np.append(fit.x, 1.0).reshape(3, 3)


def _rectangles(h: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Cada celda como rectángulo alineado: x0, x1, y0, y1."""
    p = apply(h, quads.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.column_stack([
        (p[:, 0, 0] + p[:, 3, 0]) / 2, (p[:, 1, 0] + p[:, 2, 0]) / 2,
        (p[:, 0, 1] + p[:, 1, 1]) / 2, (p[:, 3, 1] + p[:, 2, 1]) / 2,
    ])


def rectify(quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (h, rect): la transformación imagen -> suelo en metros y el rectángulo de cada celda.

    quads: celdas en píxeles, forma (n, 4, 2). El origen queda en el centro del
    parqueadero, x hacia la derecha de la imagen e y hacia la cámara.
    """
    quads = np.array([ordered(np.asarray(quad, dtype=float)) for quad in quads])

    # Los carros de las columnas están atravesados respecto a la cámara: en la foto su borde
    # de arriba (5 m) se ve mucho más largo que el lateral (2,5 m y en fuga). Los de una fila
    # del fondo, que apuntan hacia la cámara, y las motos se ven casi cuadrados o angostos.
    top_edge = np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1)
    side_edge = np.linalg.norm(quads[:, 3] - quads[:, 0], axis=1)
    in_column = top_edge / np.maximum(side_edge, 1e-6) > 2.0
    if in_column.sum() < MINIMUM_COLUMN_CELLS:
        raise ValueError(
            f"Se necesitan al menos {MINIMUM_COLUMN_CELLS} celdas de carro en columna para calcular el plano"
        )

    h = _fit_homography(quads, in_column)
    rect = _rectangles(h, quads)

    # El ajuste deja bien la perspectiva, pero la escala a lo largo queda floja (encoger ese
    # eje también "alinea" los bordes). Se fija con la separación entre puestos vecinos de
    # una misma columna, que en la realidad es el ancho del puesto.
    cx, cy = (rect[:, 0] + rect[:, 1]) / 2, (rect[:, 2] + rect[:, 3]) / 2
    pitches = []
    for i in np.flatnonzero(in_column):
        ahead = in_column & (np.abs(cx - cx[i]) < CAR_LENGTH_M / 2) & (cy > cy[i] + 1e-6)
        if ahead.any():
            pitches.append((cy[ahead] - cy[i]).min())
    if not pitches:
        raise ValueError("No hay dos puestos seguidos en una misma columna para fijar la escala")
    h = np.diag([
        CAR_LENGTH_M / np.median((rect[:, 1] - rect[:, 0])[in_column]),
        STALL_PITCH_M / np.median(pitches),
        1.0,
    ]) @ h
    rect = _rectangles(h, quads)

    origin = [(rect[:, 0].min() + rect[:, 1].max()) / 2, (rect[:, 2].min() + rect[:, 3].max()) / 2]
    shift = np.eye(3)
    shift[:2, 2] = -np.array(origin)
    h = shift @ h
    rect = _rectangles(h, quads)
    if not np.isfinite(rect).all():
        raise ValueError("El plano calculado no es válido")
    return h, rect


def build_layout(cells: dict[str, dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    """Plano de las celdas de la plataforma, listo para enviar al backend.

    cells: nombre -> {"points": [[x, y] x4 en 0..1], "puesto_id": int | None}, como las entrega
    la plataforma. Devuelve una celda por puesto vinculado, con su rectángulo en metros.
    """
    labels = list(cells)
    quads = np.array([[[x * width, y * height] for x, y in cells[label]["points"]] for label in labels])
    _, rect = rectify(quads)
    return [
        {
            "label": label,
            "parkingSpaceId": cells[label].get("puesto_id"),
            "x0": round(float(r[0]), 2), "x1": round(float(r[1]), 2),
            "y0": round(float(r[2]), 2), "y1": round(float(r[3]), 2),
        }
        for label, r in zip(labels, rect)
        if cells[label].get("puesto_id") is not None
    ]
