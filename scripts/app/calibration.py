"""Calibración de una cámara a partir de filas de puestos de medida conocida.

Calibrar es encontrar la homografía que lleva un punto del suelo en la imagen a
metros sobre el plano del parqueadero. Solo se supone que el suelo es plano y
que los puestos miden lo que dicen sus medidas; nada sobre dónde está la cámara,
así que sirve para cualquier vista.

El administrador marca cada fila con sus cuatro esquinas en la imagen, en este
orden: 0 -> 1 recorre el frente de la fila (el lado del pasillo) y 3 es la
esquina del fondo que queda detrás de 0. Como cada fila es un rectángulo de
medidas conocidas (n puestos de ancho x largo), una sola fila ya calibra la
cámara; con varias se ajustan todas a la vez y lo que no encaja se ve como error
de esa fila.

Las coordenadas de imagen son relativas (0..1), igual que las celdas de la
plataforma, para que la calibración no dependa de la resolución.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .parking_layout import apply

# Ancho y largo del puesto en metros, por tipo.
STALL_SIZES: dict[str, tuple[float, float]] = {"CAR": (2.5, 5.0), "MOTORCYCLE": (1.0, 2.5)}
# Con dos filas o más hay con qué comprobar la calibración.
MINIMUM_ROWS_TO_VERIFY = 2

# Se ajusta con la imagen centrada en (0.5, 0.5): el centro casi siempre es suelo,
# así que el término h[2, 2] nunca queda cerca de cero.
_CENTER = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Stall:
    label: str
    type: str = "CAR"


@dataclass(frozen=True)
class Row:
    label: str
    corners: np.ndarray          # (4, 2) en la imagen, relativas
    stalls: tuple[Stall, ...]
    angle: float = 90.0          # grados entre los puestos y el pasillo (90 = perpendiculares)


@dataclass(frozen=True)
class Cell:
    label: str
    type: str
    row: str
    image: np.ndarray            # (4, 2) en la imagen, relativas
    ground: np.ndarray           # (4, 2) en metros


@dataclass(frozen=True)
class RowFit:
    label: str
    ground: np.ndarray           # (4, 2) esquinas de la fila en metros
    error_m: float               # distancia media entre lo marcado y el rectángulo ajustado


@dataclass(frozen=True)
class Calibration:
    homography: np.ndarray       # imagen relativa -> metros
    rows: tuple[RowFit, ...]
    cells: tuple[Cell, ...]

    @property
    def verified(self) -> bool:
        return len(self.rows) >= MINIMUM_ROWS_TO_VERIFY

    @property
    def error_m(self) -> float:
        return max(row.error_m for row in self.rows)

    def to_image(self, ground: np.ndarray) -> np.ndarray:
        return apply(np.linalg.inv(self.homography), ground)


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _validate(row: Row, sizes: dict[str, tuple[float, float]]) -> None:
    corners = np.asarray(row.corners, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError(f"La fila {row.label} necesita cuatro esquinas")
    if not row.stalls:
        raise ValueError(f"La fila {row.label} no tiene puestos")
    unknown = {stall.type for stall in row.stalls} - set(sizes)
    if unknown:
        raise ValueError(f"La fila {row.label} tiene puestos de un tipo sin medidas: {', '.join(sorted(unknown))}")
    if not 20.0 <= row.angle <= 160.0:
        raise ValueError(f"El ángulo de la fila {row.label} debe estar entre 20 y 160 grados")
    # Las cuatro esquinas deben formar un cuadrilátero convexo: los giros van todos al mismo lado.
    edges = np.roll(corners, -1, axis=0) - corners
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if abs(_signed_area(corners)) < 1e-6 or not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        raise ValueError(f"Las esquinas de la fila {row.label} no forman un cuadrilátero; revise su orden")


def _row_shape(row: Row, sizes: dict[str, tuple[float, float]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rectángulo ideal de la fila y el polígono de cada puesto, en metros, con la fila en el origen.

    u avanza por el frente de la fila y v hacia el fondo. Si las esquinas se
    marcaron girando al revés que el rectángulo ideal, el fondo queda en v < 0;
    la homografía no refleja el suelo, así que así se respeta lo marcado.
    """
    angle = np.radians(row.angle)
    pitches, shifts, depths = [], [], []
    for stall in row.stalls:
        width, length = sizes[stall.type]
        pitches.append(width / np.sin(angle))         # lo que ocupa a lo largo del frente
        shifts.append(length * np.cos(angle))         # cuánto se corre el fondo si van en diagonal
        depths.append(length * np.sin(angle))
    start = max(0.0, -min(shifts))
    total = start + sum(pitches) + max(0.0, max(shifts))
    depth = max(depths)
    side = 1.0
    ideal = np.array([[0.0, 0.0], [total, 0.0], [total, depth], [0.0, depth]])
    if np.sign(_signed_area(np.asarray(row.corners, dtype=float))) != np.sign(_signed_area(ideal)):
        side = -1.0
        ideal[:, 1] *= side

    polygons = []
    position = start
    for pitch, shift, stall_depth in zip(pitches, shifts, depths):
        back = side * stall_depth
        polygons.append(np.array([
            [position, 0.0], [position + pitch, 0.0],
            [position + pitch + shift, back], [position + shift, back],
        ]))
        position += pitch
    return ideal, polygons


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _place(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ _rotation(pose[2]).T + pose[:2]


def calibrate(rows: list[Row], sizes: dict[str, tuple[float, float]] | None = None) -> Calibration:
    """Calibra la cámara con las filas marcadas y genera las celdas de cada puesto.

    La primera fila sirve de punto de partida; las demás se ubican donde mejor
    encajen. El plano queda como se vería la imagen desde arriba: lo cercano a la
    cámara abajo, y el origen en el centro del conjunto de filas.
    """
    sizes = {**STALL_SIZES, **(sizes or {})}
    if not rows:
        raise ValueError("Se necesita al menos una fila para calibrar")
    labels = [row.label for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("Hay filas con el mismo nombre")
    for row in rows:
        _validate(row, sizes)

    shapes = [_row_shape(row, sizes) for row in rows]
    corners = [apply(_CENTER, np.asarray(row.corners, dtype=float)) for row in rows]

    start = cv2.getPerspectiveTransform(corners[0].astype(np.float32), shapes[0][0].astype(np.float32))
    start = start / start[2, 2]
    poses = [np.zeros(3)]
    for points, (ideal, _) in zip(corners[1:], shapes[1:]):
        mapped = apply(start, points)
        edge = mapped[1] - mapped[0]
        poses.append(np.array([*mapped[0], np.arctan2(edge[1], edge[0])]))

    def unpack(params: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        h = np.append(params[:8], 1.0).reshape(3, 3)
        return h, [np.zeros(3)] + [params[8 + 3 * k: 11 + 3 * k] for k in range(len(rows) - 1)]

    def residuals(params: np.ndarray) -> np.ndarray:
        h, fitted = unpack(params)
        return np.concatenate([
            (apply(h, points) - _place(ideal, pose)).ravel()
            for points, (ideal, _), pose in zip(corners, shapes, fitted)
        ])

    params = np.concatenate([start.ravel()[:8], *poses[1:]])
    if len(rows) > 1:
        params = least_squares(residuals, params, loss="soft_l1", f_scale=0.3, x_scale="jac").x
    h, fitted = unpack(params)

    # Todas las esquinas deben quedar del lado visible del horizonte.
    every_corner = np.vstack(corners)
    w = every_corner @ h[2, :2] + h[2, 2]
    if not np.isfinite(h).all() or not (np.all(w > 0) or np.all(w < 0)):
        raise ValueError("Las filas no son coherentes entre sí; revise las esquinas marcadas")

    placed = [_place(ideal, pose) for (ideal, _), pose in zip(shapes, fitted)]
    to_plan = _plan_frame(h, np.vstack(placed))
    homography = to_plan @ h @ _CENTER
    homography = homography / homography[2, 2]
    inverse = np.linalg.inv(homography)

    row_fits, cells = [], []
    for row, points, (_, polygons), pose, ground in zip(rows, corners, shapes, fitted, placed):
        error = float(np.mean(np.linalg.norm(apply(h, points) - ground, axis=1)))
        row_fits.append(RowFit(row.label, apply(to_plan, ground), error))
        for stall, polygon in zip(row.stalls, polygons):
            cell_ground = apply(to_plan, _place(polygon, pose))
            cells.append(Cell(stall.label, stall.type, row.label, apply(inverse, cell_ground), cell_ground))
    return Calibration(homography, tuple(row_fits), tuple(cells))


def _plan_frame(h: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """Giro y traslado del plano para verlo como desde la cámara, pero desde arriba.

    Lo que en la imagen va hacia abajo (hacia la cámara) queda hacia abajo en el
    plano, y el origen en el centro de las filas.
    """
    center, below = apply(h, np.array([[0.0, 0.0], [0.0, 0.1]]))
    toward_camera = below - center
    turn = np.pi / 2 - np.arctan2(toward_camera[1], toward_camera[0])
    rotation = np.eye(3)
    rotation[:2, :2] = _rotation(turn)
    turned = apply(rotation, ground)
    origin = (turned.min(axis=0) + turned.max(axis=0)) / 2
    shift = np.array([[1.0, 0.0, -origin[0]], [0.0, 1.0, -origin[1]], [0.0, 0.0, 1.0]])
    return shift @ rotation


def top_down(
    frame: np.ndarray,
    calibration: Calibration,
    margin_m: float = 5.0,
    max_side: int = 1400,
) -> tuple[np.ndarray, np.ndarray]:
    """La imagen de la cámara vista desde arriba, y la matriz metros -> píxeles de esa imagen.

    Solo cubre las filas más un margen: más lejos el suelo se estira tanto que
    deja de servir para editar.
    """
    height, width = frame.shape[:2]
    corners = np.vstack([row.ground for row in calibration.rows])
    low, high = corners.min(axis=0) - margin_m, corners.max(axis=0) + margin_m
    pixels_per_m = max_side / float(max(high - low))
    meters_to_pixels = np.array([
        [pixels_per_m, 0.0, -low[0] * pixels_per_m],
        [0.0, pixels_per_m, -low[1] * pixels_per_m],
        [0.0, 0.0, 1.0],
    ])
    to_relative = np.diag([1.0 / width, 1.0 / height, 1.0])
    size = tuple(int(np.ceil(v)) for v in (high - low) * pixels_per_m)
    image = cv2.warpPerspective(
        frame, meters_to_pixels @ calibration.homography @ to_relative, size,
        flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40),
    )
    return image, meters_to_pixels
