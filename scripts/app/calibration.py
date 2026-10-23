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
from typing import Any

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
    # Escala con el signo que deja positiva la coordenada homogénea del suelo marcado: la
    # esquina (0, 0) suele ser cielo, detrás del horizonte, y ahí quedaría negativa. Así,
    # quien proyecte puntos puede descartar los de detrás de la cámara con w <= 0.
    relative = np.vstack([np.asarray(row.corners, dtype=float) for row in rows])
    if np.mean(relative @ homography[2, :2] + homography[2, 2]) < 0:
        homography = -homography
    inverse = np.linalg.inv(homography)

    row_fits, cells = [], []
    for row, points, (_, polygons), pose, ground in zip(rows, corners, shapes, fitted, placed):
        error = float(np.mean(np.linalg.norm(apply(h, points) - ground, axis=1)))
        row_fits.append(RowFit(row.label, apply(to_plan, ground), error))
        for stall, polygon in zip(row.stalls, polygons):
            cell_ground = apply(to_plan, _place(polygon, pose))
            cells.append(Cell(stall.label, stall.type, row.label, apply(inverse, cell_ground), cell_ground))
    return Calibration(homography, tuple(row_fits), tuple(cells))



def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Giro y traslado que mejor llevan `source` sobre `target`, y la distancia media que queda."""
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = (u @ vt).T
    if np.linalg.det(rotation) < 0:            # sin reflejos: el suelo no se ve en espejo
        vt[-1] *= -1
        rotation = (u @ vt).T
    moved = source @ rotation.T + (target.mean(axis=0) - source.mean(axis=0) @ rotation.T)
    return rotation, float(np.mean(np.linalg.norm(moved - target, axis=1)))


# Si al dejar fuera una esquina (o una fila) el resto encaja así de mejor, esa esquina (o fila) se tomó mal.
_OUTLIER_GAIN = 3.0
_OUTLIER_MIN_M = 0.15


@dataclass(frozen=True)
class Suggestion:
    corners: np.ndarray          # (4, 2) dónde deberían ir las esquinas, en la imagen (relativas)
    deviation_m: float           # cuánto se aleja lo marcado de eso, en la esquina más alejada


def _suggest(row: Row, others: list[Row], sizes: dict[str, tuple[float, float]]) -> Suggestion | None:
    """La fila ubicada según la calibración de las otras filas; None si no se puede."""
    try:
        reference = calibrate(others, sizes)
    except ValueError:
        return None
    ideal, _ = _row_shape(row, sizes)
    ground = apply(reference.homography, np.asarray(row.corners, dtype=float))
    if not np.isfinite(ground).all():
        return None
    keep = np.arange(4)
    _, full_error = _rigid_fit(ideal, ground)
    if full_error > _OUTLIER_MIN_M:
        best_error, worst_corner = min((_rigid_fit(ideal[np.delete(keep, i)], ground[np.delete(keep, i)])[1], i) for i in range(4))
        if best_error * _OUTLIER_GAIN < full_error:
            keep = np.delete(keep, worst_corner)
    rotation, _ = _rigid_fit(ideal[keep], ground[keep])
    fitted = ideal @ rotation.T + (ground[keep].mean(axis=0) - ideal[keep].mean(axis=0) @ rotation.T)
    deviation = float(np.linalg.norm(fitted - ground, axis=1).max())
    return Suggestion(reference.to_image(fitted), deviation)


def suggested_corners(
    rows: list[Row], sizes: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Suggestion]:
    """Dónde deberían ir las esquinas de cada fila para encajar con las demás, en la imagen.

    Cada fila se compara con la calibración de las otras filas, para que sus propios errores no
    la deformen. Si una esquina no encaja con las otras tres (típico: una punta de la fila
    marcada más larga), se ignora y la fila se ubica con las otras tres.
    Hace falta al menos otra fila; las que no se pueden comparar no aparecen.
    """
    sizes = {**STALL_SIZES, **(sizes or {})}
    if len(rows) < 2:
        return {}

    def against(reference: list[Row]) -> dict[str, Suggestion]:
        found = {}
        for index, row in enumerate(rows):
            suggestion = _suggest(row, reference[:index] + reference[index + 1:], sizes)
            if suggestion is not None:
                found[row.label] = suggestion
        return found

    # Una fila mal marcada también descuadra la revisión de las demás: se corrige de a una, la que
    # peor encaja primero, hasta que todas encajen. Así no se "corrigen" filas buenas.
    reference = list(rows)
    cleaned: set[str] = set()
    for _ in range(len(rows) - 1):
        found = against(reference)
        pending = [label for label in found if label not in cleaned]
        if not pending:
            break
        worst = max(pending, key=lambda label: found[label].deviation_m)
        rest = [found[label].deviation_m for label in found if label != worst]
        # Solo se corrige una fila que se sale del patrón de las demás. Si todas se desvían parecido, el
        # problema no es una fila mal marcada (por ejemplo, puestos que no miden lo de siempre).
        if found[worst].deviation_m <= max(_OUTLIER_MIN_M, _OUTLIER_GAIN * float(np.median(rest))):
            return found
        cleaned.add(worst)
        reference = [
            Row(row.label, found[worst].corners, row.stalls, row.angle) if row.label == worst else row
            for row in reference
        ]
    return against(reference)

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
    margin_m: float = 2.0,
    max_side: int = 1600,
) -> tuple[np.ndarray, np.ndarray]:
    """La imagen de la cámara vista desde arriba, como la de un dron, y la matriz metros -> píxeles.

    Solo cubre las filas más un margen: más lejos el suelo se estira tanto que deja de
    servir. Se limpia el grano del video sin borrar los bordes, y lo que la cámara no ve
    se rellena con una continuación borrosa y oscura de la imagen en vez de un fondo gris.
    Los objetos altos (carros, árboles) siguen viéndose inclinados: con una sola cámara de
    lado no se sabe cómo son por arriba.
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
    warp = meters_to_pixels @ calibration.homography @ to_relative
    image = cv2.warpPerspective(frame, warp, size, flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    seen = cv2.warpPerspective(np.full((height, width), 255, np.uint8), warp, size, flags=cv2.INTER_NEAREST)
    # El borde de lo visto se mezcla con el negro de fuera al interpolar: se descarta.
    seen = cv2.erode(seen, np.ones((5, 5), np.uint8))

    image = _fill_unseen(image, seen > 0)
    image = cv2.bilateralFilter(image, 7, 28, 5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.12, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), meters_to_pixels


def _fill_unseen(image: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Rellena lo que la cámara no ve prolongando, borroso y oscuro, el color de los bordes."""
    if seen.all() or not seen.any():
        return image
    mask = seen.astype(np.float32)
    # A un octavo del tamaño, cada vez más borroso, para que el relleno llegue lejos sin costar mucho.
    small_image = cv2.resize(image.astype(np.float32) * mask[..., None], None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    for sigma in (3, 10, 27):
        blurred = cv2.GaussianBlur(small_image, (0, 0), sigma)
        weight = cv2.GaussianBlur(small_mask, (0, 0), sigma)
        known = small_mask > 0.02
        colors = np.where(known[..., None], small_image / np.maximum(small_mask, 1e-4)[..., None],
                          blurred / np.maximum(weight, 1e-4)[..., None])
        small_mask = np.maximum(small_mask, (weight > 1e-3).astype(np.float32))
        small_image = colors * small_mask[..., None]
    colors = small_image / np.maximum(small_mask, 1e-4)[..., None]
    fill = cv2.resize(colors, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    fill = cv2.GaussianBlur(fill, (0, 0), 6) * 0.55
    # Transición suave solo hacia afuera: dentro de lo visto la imagen queda intacta.
    alpha = np.maximum(mask, cv2.GaussianBlur(mask, (0, 0), 3))[..., None]
    outside = np.where(seen[..., None], image.astype(np.float32), fill)
    return np.clip(outside * alpha + fill * (1 - alpha), 0, 255).astype(np.uint8)


# --- Plano de planta para el mapa 3D -------------------------------------------------------

# Puestos extra por cada lado de una fila (motos al principio o al final), como en el editor.
_MAX_ROW_MOTOS = 30


def _editor_row(index: int, raw: Any) -> tuple[Row, dict[str, Any]] | None:
    """Fila tal como la guarda el editor de la plataforma; None si no se puede usar.

    El editor la guarda con: corners (4 esquinas relativas), type, count, angle y las
    motos que la siguen por fuera de lo marcado (motosStart, motosEnd, motoWidth).
    """
    if not isinstance(raw, dict):
        return None
    corners = np.asarray(raw.get("corners"), dtype=float) if isinstance(raw.get("corners"), list) else None
    count = raw.get("count")
    if corners is None or corners.shape != (4, 2) or not isinstance(count, int) or count < 1:
        return None
    kind = "MOTORCYCLE" if raw.get("type") == "MOTORCYCLE" else "CAR"
    angle = raw.get("angle") if isinstance(raw.get("angle"), (int, float)) else 90.0

    def amount(key: str) -> int:
        value = raw.get(key)
        return int(value) if isinstance(value, (int, float)) and 0 <= value <= _MAX_ROW_MOTOS else 0

    moto_width = raw.get("motoWidth")
    extras = {
        "type": kind,
        "count": count,
        "motos_start": amount("motosStart"),
        "motos_end": amount("motosEnd"),
        "moto_width": float(moto_width) if isinstance(moto_width, (int, float)) and 0.5 <= moto_width <= 3 else STALL_SIZES["MOTORCYCLE"][0],
    }
    row = Row(str(index), corners, tuple(Stall(f"{index}-{i}", kind) for i in range(count)), float(angle))
    return row, extras


def _row_stalls_ground(fit: RowFit, row: Row, extras: dict[str, Any]) -> list[np.ndarray]:
    """Cada puesto de la fila en el plano, incluidas las motos de los extremos.

    Se reparte igual que en el editor: a lo largo del frente, con el mismo fondo que la
    fila. Así cada celda que generó el editor tiene aquí su puesto exacto.
    """
    width, length = STALL_SIZES[extras["type"]]
    angle = np.radians(row.angle)
    sin = np.sin(angle)
    pitch, shift, depth = width / sin, length * np.cos(angle), length * sin
    start = max(0.0, -shift)
    total = start + extras["count"] * pitch + max(0.0, shift)
    g0, g1, _, g3 = fit.ground
    along, inward = (g1 - g0) / total, (g3 - g0) / depth

    def stall(position: float, size: float) -> np.ndarray:
        points = [(position, 0.0), (position + size, 0.0), (position + size + shift, depth), (position + shift, depth)]
        return np.array([g0 + u * along + v * inward for u, v in points])

    moto = extras["moto_width"] / sin
    end = start + extras["count"] * pitch
    return (
        [stall(start - (extras["motos_start"] - i) * moto, moto) for i in range(extras["motos_start"])]
        + [stall(start + i * pitch, pitch) for i in range(extras["count"])]
        + [stall(end + i * moto, moto) for i in range(extras["motos_end"])]
    )


def _alignment(fits: tuple[RowFit, ...]) -> np.ndarray:
    """Giro que deja las filas paralelas a los ejes del plano.

    El mapa 3D dibuja cada puesto como un rectángulo alineado con los ejes, así que las
    filas se enderezan: se gira lo mínimo (menos de 45 grados) para que el frente de las
    filas quede horizontal o vertical. Lo cercano a la cámara sigue quedando abajo.
    """
    angles = np.array([np.arctan2(*(fit.ground[1] - fit.ground[0])[::-1]) for fit in fits])
    # Promedio de direcciones módulo 90 grados: se multiplican por 4 para que 0, 90, 180 y 270 coincidan.
    mean = np.arctan2(np.sin(4 * angles).mean(), np.cos(4 * angles).mean()) / 4
    return _rotation(-mean)


def layout_from_rows(
    cells: dict[str, dict[str, Any]],
    rows: list[Any],
) -> list[dict[str, Any]] | None:
    """Plano de planta de las celdas usando las filas calibradas del editor.

    cells: nombre -> {"points": 4 esquinas relativas, "puesto_id": int | None, "fila": int | None}.
    rows: las filas tal como las guardó el editor. Las celdas de una fila toman la medida
    exacta de su puesto; las sueltas se llevan al plano con la misma calibración.
    Devuelve None si no hay filas utilizables, para seguir con el cálculo anterior.
    """
    parsed = {index: entry for index, raw in enumerate(rows or []) if (entry := _editor_row(index, raw))}
    if not parsed:
        return None
    calibration = calibrate([row for row, _ in parsed.values()])
    turn = _alignment(calibration.rows)

    stalls_by_row = {
        index: [polygon @ turn.T for polygon in _row_stalls_ground(fit, row, extras)]
        for (index, (row, extras)), fit in zip(parsed.items(), calibration.rows)
    }
    taken: dict[int, set[int]] = {index: set() for index in stalls_by_row}

    layout = []
    for label, cell in cells.items():
        if cell.get("puesto_id") is None:
            continue
        ground = apply(calibration.homography, np.asarray(cell["points"], dtype=float)) @ turn.T
        stalls = stalls_by_row.get(cell.get("fila"))
        if stalls:
            # La celda es uno de los puestos de su fila: el más cercano que no se haya usado.
            center = ground.mean(axis=0)
            order = sorted(range(len(stalls)), key=lambda i: np.linalg.norm(stalls[i].mean(axis=0) - center))
            chosen = next((i for i in order if i not in taken[cell["fila"]]), order[0])
            taken[cell["fila"]].add(chosen)
            ground = stalls[chosen]
        low, high = ground.min(axis=0), ground.max(axis=0)
        layout.append({
            "label": label,
            "parkingSpaceId": cell["puesto_id"],
            "x0": round(float(low[0]), 2), "x1": round(float(high[0]), 2),
            "y0": round(float(low[1]), 2), "y1": round(float(high[1]), 2),
        })
    return layout
