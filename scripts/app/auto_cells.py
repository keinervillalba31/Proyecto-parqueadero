"""Propone las celdas de una cámara a partir de dónde se estacionan los vehículos.

No se buscan las líneas pintadas, que en un parqueadero real suelen estar borrosas o tapadas:
lo que define un puesto es dónde se detienen los carros. El proceso es:

1. Observar: siluetas de vehículos (segmentación) en muchos cuadros, ya llevadas al cuadro de
   referencia para que un temblor de la cámara no las mueva.
2. Quietos: se siguen entre cuadros y se quedan los que casi no se mueven y aparecen en buena
   parte de los cuadros. Un carro que pasa por el pasillo no es un puesto.
3. Filas: los puntos de apoyo de los carros de una fila quedan sobre una recta (el suelo es
   plano y una recta del suelo sigue siendo recta en la imagen).
4. Perspectiva: las filas paralelas del suelo se cortan en la imagen en un punto de fuga, y
   los separadores (paralelos al largo de los carros) en otro. Con esos dos puntos cada puesto
   queda bien formado aunque esté lejos, sin calibrar la cámara.
5. Puestos: a lo largo de una fila, todos los puestos miden lo mismo en el suelo. Pasando las
   posiciones de la imagen a una escala proyectiva (1 / distancia al punto de fuga), el paso
   vuelve a ser constante: un hueco de dos pasos es un puesto vacío entre dos ocupados.
6. Fila espalda con espalda: un carro suelto pegado a una fila es el comienzo de la fila de
   enfrente, que se deduce con los mismos separadores.

Cada celda dice si se vio un vehículo en ella ("observada") o si se dedujo ("deducida"), para
que el administrador revise primero las deducidas. Las coordenadas son de píxeles del cuadro
de referencia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MOTORCYCLE_CLASS_ID = 3
# Un vehículo es "quieto" si aparece en al menos esta fracción de los cuadros...
MIN_PRESENCE = 0.5
# ...y su centro se mueve menos que esta fracción de su tamaño.
MAX_MOTION = 0.25
# Dos siluetas de cuadros distintos son el mismo vehículo si sus cajas se solapan así.
MATCH_IOU = 0.4
# Distancia a la recta de la fila, en fracciones del tamaño del vehículo.
ROW_TOLERANCE = 0.35
# Hueco máximo entre vecinos de una fila, en fracciones del tamaño del vehículo.
ROW_MAX_GAP = 3.5
# Una fila "apunta" a un punto de fuga si el ángulo hacia él es menor que esto.
VANISHING_TOLERANCE_DEG = 6.0
# El borde inferior de un carro solo dice su largo si se cruza con la fila al menos así.
LENGTH_CUE_MIN_DEG = 30.0
# Percentiles de la silueta que marcan el frente y el fondo del puesto.
DEPTH_PERCENTILES = (3.0, 97.0)
# El puesto es algo más largo que el carro (5 m frente a ~4,3 m): margen a cada lado del fondo.
STALL_MARGIN = 0.08
# Una celda deducida que tiene más de esta fracción encima de otra se descarta.
MAX_DEDUCED_OVERLAP = 0.3
# Un carro suelto abre la fila de enfrente solo si está orientado como los de la fila (grados).
MIRROR_MAX_TURN_DEG = 25.0
# El paso de una fila no baja de esta fracción del hueco mediano entre sus carros.
LATTICE_MIN_STEP = 0.45
# Dos carros de una fila separados por menos que esto (en huecos medianos) son el mismo.
DUPLICATE_GAP = 0.4
# Una celda más grande que esto (en siluetas de su carro más grande) es una fila mal armada.
MAX_CELL_AREA = 4.0
# Qué tanto pueden apartarse los huecos de un número entero de puestos (en pasos).
LATTICE_TOLERANCE = 0.15
# Dos vehículos son de la misma fila si sus franjas se solapan en más de esta fracción.
BAND_OVERLAP = 0.5
# Más puestos vacíos seguidos que esto separan dos filas distintas.
MAX_EMPTY_RUN = 3


@dataclass(frozen=True)
class Observation:
    class_id: int
    polygon: np.ndarray  # (N, 2) en píxeles de la referencia


@dataclass(eq=False)
class Vehicle:
    """Un vehículo quieto: su silueta de consenso y lo que se deriva de ella."""

    class_id: int
    mask: np.ndarray  # bool, tamaño de la imagen
    presence: float
    ground: np.ndarray = field(init=False)  # punto de apoyo (centro del borde inferior)
    size: float = field(init=False)
    length_axis: np.ndarray | None = field(init=False)  # borde inferior de la silueta

    def __post_init__(self) -> None:
        ys, xs = np.nonzero(self.mask)
        x1, x2 = xs.min(), xs.max()
        # Punto de apoyo: el centro del borde inferior, donde el vehículo toca el suelo.
        self.ground = self.contact().mean(axis=0)
        self.size = float(np.sqrt(self.mask.sum()))
        self.length_axis = _bottom_direction(self.mask, x1, x2)

    @property
    def kind(self) -> str:
        return "MOTORCYCLE" if self.class_id == MOTORCYCLE_CLASS_ID else "CAR"

    def pixels(self) -> np.ndarray:
        ys, xs = np.nonzero(self.mask)
        return np.column_stack([xs, ys]).astype(float)

    def contact(self) -> np.ndarray:
        """El borde inferior de la silueta, columna por columna: donde el vehículo toca el suelo.

        El resto de la silueta es carrocería que se levanta del suelo y, en perspectiva, se corre
        hacia un lado; medir el puesto con ella lo desplaza.
        """
        columns = np.nonzero(self.mask.any(axis=0))[0]
        sub = self.mask[:, columns]
        bottoms = sub.shape[0] - 1 - np.argmax(sub[::-1], axis=0)
        return np.column_stack([columns, bottoms]).astype(float)


@dataclass
class Cell:
    corners: np.ndarray  # (4, 2) en orden alrededor del puesto
    kind: str
    row: int
    observed: bool


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) or 1.0)


def _homogeneous(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], point[1], 1.0])


def _line(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Recta homogénea por un punto con una dirección, normalizada a distancias en píxeles."""
    line = np.cross(_homogeneous(point), _homogeneous(point + direction))
    return line / (np.linalg.norm(line[:2]) or 1.0)


def _meet(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    point = np.cross(a, b)
    return point[:2] / point[2]


def _toward(point: np.ndarray, vanishing: np.ndarray) -> np.ndarray:
    """Dirección desde un punto hacia un punto de fuga (que puede estar en el infinito)."""
    return _unit(vanishing[:2] - point * vanishing[2])


def _vanishing_point(lines: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """El punto (posiblemente en el infinito) más cercano a todas las rectas."""
    stack = np.array(lines) * (np.ones(len(lines)) if weights is None else np.array(weights))[:, None]
    _, _, vt = np.linalg.svd(stack)
    point = vt[-1]
    return point / np.linalg.norm(point)


def _angle_to(point: np.ndarray, direction: np.ndarray, vanishing: np.ndarray) -> float:
    """Ángulo (grados, sin sentido) entre una dirección y la que va hacia el punto de fuga."""
    cosine = abs(float(_toward(point, vanishing) @ _unit(direction)))
    return float(np.degrees(np.arccos(min(cosine, 1.0))))


def _bottom_direction(mask: np.ndarray, x1: int, x2: int) -> np.ndarray | None:
    """Dirección del borde inferior de la silueta: donde las llantas tocan el suelo.

    De lado, ese borde sigue el largo del carro; visto de frente o de atrás sigue el ancho.
    """
    columns = np.arange(x1, x2 + 1)
    if len(columns) < 8:
        return None
    sub = mask[:, x1 : x2 + 1]
    has = sub.any(axis=0)
    bottoms = sub.shape[0] - 1 - np.argmax(sub[::-1], axis=0)
    points = np.column_stack([columns[has], bottoms[has]]).astype(np.float32)
    # Solo el tramo central: las puntas del borde se curvan con el parachoques.
    points = points[len(points) // 6 : len(points) - len(points) // 6]
    if len(points) < 5:
        return None
    vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return _unit(np.array([vx, vy]))


# --- 1 y 2: vehículos quietos ------------------------------------------------------------


def _box(polygon: np.ndarray) -> np.ndarray:
    return np.array([*polygon.min(axis=0), *polygon.max(axis=0)])


def _box_ious(items: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """IoU de cada caja de `items` (N, 4) con cada caja de `tracks` (T, 4)."""
    x1 = np.maximum(items[:, None, 0], tracks[None, :, 0])
    y1 = np.maximum(items[:, None, 1], tracks[None, :, 1])
    x2 = np.minimum(items[:, None, 2], tracks[None, :, 2])
    y2 = np.minimum(items[:, None, 3], tracks[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_items = (items[:, 2] - items[:, 0]) * (items[:, 3] - items[:, 1])
    area_tracks = (tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1])
    union = area_items[:, None] + area_tracks[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _consensus(track: list[Observation], shape: tuple[int, int]) -> np.ndarray:
    """Silueta de consenso: los píxeles que cubre al menos la mitad de las observaciones.

    Se vota solo dentro de la caja que abarca el seguimiento, no en la imagen entera, para que
    una observación de horas (miles de siluetas) no tarde minutos.
    """
    height, width = shape
    boxes = np.array([_box(item.polygon) for item in track])
    x1, y1 = np.clip(np.floor(boxes[:, :2].min(axis=0)).astype(int), 0, None)
    x2 = int(min(np.ceil(boxes[:, 2].max()) + 1, width))
    y2 = int(min(np.ceil(boxes[:, 3].max()) + 1, height))
    votes = np.zeros((max(y2 - y1, 1), max(x2 - x1, 1)), np.uint16)
    canvas = np.zeros_like(votes, dtype=np.uint8)
    for item in track:
        canvas[:] = 0
        cv2.fillPoly(canvas, [np.round(item.polygon - [x1, y1]).astype(np.int32)], 1)
        votes += canvas
    mask = np.zeros(shape, bool)
    mask[y1:y2, x1:x2] = votes[: y2 - y1, : x2 - x1] >= max(1, len(track) / 2)
    return mask


def stationary_vehicles(
    frames: list[list[Observation]],
    shape: tuple[int, int],
    min_presence: float = MIN_PRESENCE,
    max_motion: float = MAX_MOTION,
    min_sightings: int = 1,
) -> list[Vehicle]:
    """Sigue las siluetas entre cuadros y devuelve los vehículos que se quedan quietos.

    Un seguimiento junta lo que se ve en un mismo lugar a lo largo del tiempo: si en un puesto se
    estacionan varios carros uno tras otro, es un solo seguimiento (es el mismo puesto). Cuenta si
    se vio en al menos `min_presence` de los cuadros y en al menos `min_sightings` cuadros: en una
    observación de horas, un puesto usado media hora ya cuenta, y un carro que pasa no.
    """
    tracks: list[list[Observation]] = []
    last_boxes = np.zeros((0, 4))  # última caja de cada seguimiento
    for frame in frames:
        if not frame:
            continue
        boxes = np.array([_box(item.polygon) for item in frame])
        matched: set[int] = set()
        if len(tracks):
            ious = _box_ious(boxes, last_boxes)
            taken: set[int] = set()
            for flat in np.argsort(-ious, axis=None):
                position, track = divmod(int(flat), ious.shape[1])
                if ious[position, track] < MATCH_IOU:
                    break
                if position in matched or track in taken:
                    continue
                matched.add(position)
                taken.add(track)
                tracks[track].append(frame[position])
                last_boxes[track] = boxes[position]
        new = [position for position in range(len(frame)) if position not in matched]
        for position in new:
            tracks.append([frame[position]])
        if new:
            last_boxes = np.vstack([last_boxes, boxes[new]])

    vehicles = []
    total = max(len(frames), 1)
    needed = max(min_presence * total, min_sightings)
    for track in tracks:
        if len(track) < needed:
            continue
        track_boxes = np.array([_box(item.polygon) for item in track])
        centers = (track_boxes[:, :2] + track_boxes[:, 2:]) / 2
        size = np.median(np.sqrt((track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])))
        if np.linalg.norm(centers.std(axis=0)) > max_motion * size:
            continue
        mask = _consensus(track, shape)
        if mask.sum() < 50:
            continue
        class_id = int(np.bincount([item.class_id for item in track]).argmax())
        vehicles.append(Vehicle(class_id, mask, len(track) / total))
    return _drop_contained(vehicles)


def _drop_contained(vehicles: list[Vehicle]) -> list[Vehicle]:
    """Una silueta casi entera dentro de otra es el mismo vehículo visto a medias."""
    vehicles = sorted(vehicles, key=lambda vehicle: -vehicle.mask.sum())
    kept: list[Vehicle] = []
    for vehicle in vehicles:
        area = vehicle.mask.sum()
        if any((vehicle.mask & other.mask).sum() > 0.7 * area for other in kept):
            continue
        kept.append(vehicle)
    return kept


# --- 3: filas -------------------------------------------------------------------------------


@dataclass(eq=False)
class Row:
    origin: np.ndarray
    axis: np.ndarray  # unitario, a lo largo de la fila
    members: list[Vehicle]

    def position(self, point: np.ndarray) -> float:
        return float((point - self.origin) @ self.axis)

    def offset(self, point: np.ndarray) -> float:
        return float((point - self.origin) @ np.array([-self.axis[1], self.axis[0]]))

    @property
    def line(self) -> np.ndarray:
        return _line(self.origin, self.axis)

    @property
    def kind(self) -> str:
        return self.members[0].kind


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center)
    return center, _unit(vt[0])


def _chain(row: Row, candidates: list[Vehicle]) -> list[Vehicle]:
    """Los vehículos cerca de la recta que forman una cadena sin huecos enormes."""
    near = [v for v in candidates if abs(row.offset(v.ground)) < ROW_TOLERANCE * v.size]
    near.sort(key=lambda v: row.position(v.ground))
    if not near:
        return []
    best: list[Vehicle] = []
    current = [near[0]]
    for previous, vehicle in zip(near, near[1:]):
        gap = row.position(vehicle.ground) - row.position(previous.ground)
        if gap > ROW_MAX_GAP * (previous.size + vehicle.size) / 2:
            best = max(best, current, key=len)
            current = []
        current.append(vehicle)
    return max(best, current, key=len)


def _make_row(members: list[Vehicle], candidates: list[Vehicle]) -> Row:
    origin, axis = _fit_line(np.array([v.ground for v in members]))
    row = Row(origin, axis, [])
    row.members = _chain(row, candidates) or members
    origin, axis = _fit_line(np.array([v.ground for v in row.members]))
    row.origin, row.axis = origin, axis
    row.members.sort(key=lambda v: row.position(v.ground))
    return row


def find_rows(
    vehicles: list[Vehicle], minimum: int = 3, accept=lambda row: True
) -> tuple[list[Row], list[Vehicle]]:
    """Agrupa los vehículos en filas rectas (RANSAC secuencial). Devuelve filas y sueltos."""
    left = list(vehicles)
    rows: list[Row] = []
    while len(left) >= minimum:
        best: Row | None = None
        for i, a in enumerate(left):
            for b in left[i + 1 :]:
                if a.kind != b.kind:
                    continue
                distance = np.linalg.norm(a.ground - b.ground)
                mean_size = (a.size + b.size) / 2
                if distance < 0.3 * mean_size or distance > ROW_MAX_GAP * mean_size:
                    continue
                trial = Row(a.ground, _unit(b.ground - a.ground), [])
                chain = _chain(trial, [v for v in left if v.kind == a.kind])
                if len(chain) >= minimum and (best is None or len(chain) > len(best.members)):
                    row = _make_row(chain, [v for v in left if v.kind == a.kind])
                    if accept(row):
                        best = row
        if best is None:
            break
        rows.append(best)
        left = [v for v in left if all(v is not member for member in best.members)]
    return rows, left


# --- 4: perspectiva -------------------------------------------------------------------------


@dataclass
class Perspective:
    """Los dos puntos de fuga del suelo: el de las filas y el de los separadores."""

    rows: np.ndarray
    dividers: np.ndarray

    def divider_vanishing(self, row: Row) -> np.ndarray:
        """Una fila que va hacia el punto de los separadores es una fila transversal."""
        center = row.origin
        if _angle_to(center, row.axis, self.rows) <= _angle_to(center, row.axis, self.dividers):
            return self.dividers
        return self.rows

    def row_vanishing(self, row: Row) -> np.ndarray:
        return self.rows if self.divider_vanishing(row) is self.dividers else self.dividers


def _undirected_mean(vectors: list[np.ndarray]) -> np.ndarray:
    """Promedio de direcciones sin sentido (una recta apunta igual hacia los dos lados)."""
    angles = np.array([np.arctan2(v[1], v[0]) for v in vectors])
    angle = np.arctan2(np.sin(2 * angles).sum(), np.cos(2 * angles).sum()) / 2
    return np.array([np.cos(angle), np.sin(angle)])


def _length_cues(row: Row) -> list[np.ndarray]:
    cues = []
    for vehicle in row.members:
        axis = vehicle.length_axis
        if axis is not None and abs(axis @ row.axis) < np.cos(np.radians(LENGTH_CUE_MIN_DEG)):
            cues.append(_line(vehicle.ground, axis))
    return cues


def _orthogonal_vanishing(rows_vp: np.ndarray, center: np.ndarray, focal: float) -> np.ndarray:
    """Punto de fuga de la dirección del suelo perpendicular a la de las filas.

    Con la cámara sin ladear, el horizonte es horizontal y pasa por el punto de las filas; sobre
    él, el punto perpendicular cumple (v1 - c) · (v2 - c) = -f², con c el centro de la imagen.
    """
    if abs(rows_vp[2]) < 1e-9:
        # Filas paralelas en la imagen (cámara mirando hacia abajo): separadores perpendiculares.
        return np.array([-rows_vp[1], rows_vp[0], 0.0])
    v1 = rows_vp[:2] / rows_vp[2] - center
    if abs(v1[0]) < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    x = -(focal**2 + v1[1] ** 2) / v1[0]
    point = np.array([x + center[0], v1[1] + center[1], 1.0])
    return point / np.linalg.norm(point)


def _cue_error(cues: list[tuple[np.ndarray, np.ndarray]], vanishing: np.ndarray) -> float:
    """Desacuerdo mediano (grados) entre las pistas de los carros y un punto de fuga."""
    return float(np.median([min(_angle_to(point, direction, vanishing), 20.0) for point, direction in cues]))


def estimate_perspective(rows: list[Row], shape: tuple[int, int]) -> Perspective:
    """Punto de fuga de las filas (donde se cortan) y de los separadores (perpendicular en el suelo).

    El de las filas sale de las rectas de las filas. El de los separadores se deduce de él,
    suponiendo la cámara sin ladear y una distancia focal razonable; los bordes de los carros
    vistos de lado solo eligen la focal, porque cada uno por separado es ruidoso (±10°).
    """
    height, width = shape
    center = np.array([width / 2, height / 2])
    if not rows:
        return Perspective(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    # La familia de filas más grande que apunta a un mismo punto: RANSAC sobre pares de filas.
    best_vp, best_family = np.array([*rows[0].axis, 0.0]), [rows[0]]
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            vp = _vanishing_point([a.line, b.line])
            family = [r for r in rows if _angle_to(r.origin, r.axis, vp) < VANISHING_TOLERANCE_DEG]
            if sum(len(r.members) for r in family) > sum(len(r.members) for r in best_family):
                best_vp, best_family = vp, family
    if len(best_family) >= 2:
        best_vp = _vanishing_point([r.line for r in best_family], [len(r.members) for r in best_family])

    cues = [
        (vehicle.ground, vehicle.length_axis)
        for row in best_family
        for vehicle in row.members
        if vehicle.length_axis is not None
        and abs(vehicle.length_axis @ row.axis) < np.cos(np.radians(LENGTH_CUE_MIN_DEG))
    ]
    focals = width * np.geomspace(0.5, 2.5, 15)
    if cues:
        focal = min(focals, key=lambda f: _cue_error(cues, _orthogonal_vanishing(best_vp, center, f)))
    else:
        focal = 0.9 * width
    return Perspective(best_vp, _orthogonal_vanishing(best_vp, center, float(focal)))


# --- 5: puestos de una fila -----------------------------------------------------------------


class _Scale:
    """Lleva posiciones de la imagen sobre la recta de la fila a una escala lineal del suelo.

    Con el punto de fuga de la fila en la posición v, w = 1 / (v - t) es proporcional a la
    distancia en el suelo; si el punto está en el infinito, la imagen ya es lineal (w = t).
    """

    def __init__(self, row: Row, vanishing: np.ndarray) -> None:
        self.finite = abs(vanishing[2]) > 1e-9 and np.isfinite(vanishing[:2] / vanishing[2]).all()
        if self.finite:
            point = vanishing[:2] / vanishing[2]
            self.v = row.position(point)
            # Que sea finito pero tan lejos que no cambia nada es lo mismo que el infinito.
            spread = max(abs(row.position(m.ground)) for m in row.members) + 1.0
            self.finite = abs(self.v) < 1e4 * spread
        self.sign = 1.0
        if self.finite and self.v < 0:
            self.sign = -1.0

    def to_w(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if not self.finite:
            return t
        return self.sign / (self.v - t)

    def to_t(self, w: np.ndarray | float) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        if not self.finite:
            return w
        return self.v - self.sign / w


def _one_per_stall(members: list[Vehicle], w: np.ndarray) -> list[Vehicle]:
    """Dos vehículos de una fila casi en la misma posición son uno solo visto en dos siluetas
    (pasa con pocas observaciones): en un puesto no caben dos carros. Se queda la más grande."""
    if len(members) < 3:
        return members
    order = np.argsort(w)
    gaps = np.diff(w[order])
    limit = DUPLICATE_GAP * float(np.median(gaps))
    kept = [order[0]]
    for index, gap in zip(order[1:], gaps):
        if gap < limit:
            if members[index].mask.sum() > members[kept[-1]].mask.sum():
                kept[-1] = index
            continue
        kept.append(index)
    return [members[i] for i in sorted(kept)]


def _lattice(w: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Ajusta w_i = w0 + paso * k_i con k_i enteros: el índice de puesto de cada carro.

    El paso parte del hueco mediano y se refina con los huecos que son un número entero de
    pasos. Nunca baja de LATTICE_MIN_STEP veces el hueco mediano: puede ser la mitad (una fila
    con un carro cada dos puestos) pero no menos, para que el ruido no llene la fila de puestos.
    """
    order = np.argsort(w)
    w = w[order]
    gaps = np.diff(w)
    if len(gaps) == 0:
        return float(w[0]), 0.0, np.zeros(1, dtype=int)
    floor = LATTICE_MIN_STEP * float(np.median(gaps))
    step = float(np.median(gaps))
    for _ in range(5):
        counts = np.maximum(np.round(gaps / step), 1)
        step = max(float(gaps.sum() / counts.sum()), floor)
    counts = np.maximum(np.round(gaps / step), 1).astype(int)
    # Si todos los huecos son múltiplos de 2 (o 3) pasos, nada respalda un paso tan corto: es
    # medio puesto, que aparece con pocas observaciones ruidosas, y llenaría de puestos falsos
    # cada hueco entre carros vecinos.
    common = int(np.gcd.reduce(counts))
    if common > 1:
        counts //= common
    k = np.concatenate([[0], np.cumsum(counts)]).astype(int)
    # Mínimos cuadrados con los índices ya fijos.
    a = np.column_stack([np.ones(len(k)), k])
    (w0, step), *_ = np.linalg.lstsq(a, w, rcond=None)
    indices = np.empty_like(k)
    indices[order] = k
    return float(w0), float(step), indices


def _project_on_transversal(
    points: np.ndarray, row_vp: np.ndarray, transversal: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Lleva puntos, por la recta que los une con el punto de fuga de la fila, a un separador.

    Así un punto lejano y uno cercano que están a la misma distancia de la fila en el suelo
    caen en el mismo sitio del separador de referencia.
    """
    homogeneous = np.column_stack([points, np.ones(len(points))])
    lines = np.cross(homogeneous, row_vp)
    hits = np.cross(lines, transversal)
    hits = hits[:, :2] / hits[:, 2:3]
    return (hits - origin) @ direction


class RowModel:
    """Una fila ya medida: su escala proyectiva, el paso de sus puestos y su banda de fondo."""

    def __init__(self, row: Row, perspective: Perspective) -> None:
        self.row = row
        self.row_vp = perspective.row_vanishing(row)
        self.divider_vp = perspective.divider_vanishing(row)
        self.direction = _toward(row.origin, self.divider_vp)
        # Separador de referencia: el que pasa por el centro de la fila.
        self.transversal = _line(row.origin, self.direction)
        self.scale = _Scale(row, self.row_vp)
        row.members = _one_per_stall(row.members, self.scale.to_w(np.array([row.position(v.ground) for v in row.members])))
        w = self.scale.to_w(np.array([row.position(v.ground) for v in row.members]))
        self.w0, self.step, self.indices = _lattice(w)
        if self.step == 0.0:
            self.step = self._single_step()
        bands = np.array([self.band(v) for v in row.members])
        self.low, self.high = float(np.median(bands[:, 0])), float(np.median(bands[:, 1]))

    @property
    def last(self) -> int:
        return int(self.indices.max())

    def band(self, vehicle: Vehicle) -> tuple[float, float]:
        """Dónde empieza y termina el vehículo de un lado a otro de la fila (sobre el separador)."""
        values = _project_on_transversal(
            vehicle.contact(), self.row_vp, self.transversal, self.row.origin, self.direction
        )
        low, high = np.percentile(values, DEPTH_PERCENTILES)
        return float(low), float(high)

    def slot(self, vehicle: Vehicle) -> float:
        """Índice de puesto (sin redondear) en que cae un vehículo a lo largo de la fila."""
        return float((self.scale.to_w(self.row.position(vehicle.ground)) - self.w0) / self.step)

    def _single_step(self) -> float:
        """Paso de una fila de un solo carro: lo que ocupa el carro a lo largo de la fila."""
        vehicle = self.row.members[0]
        along = vehicle.pixels() @ self.row.axis - self.row.origin @ self.row.axis
        low, high = np.percentile(along, (10, 90))
        return float(abs(self.scale.to_w(high) - self.scale.to_w(low)))

    def cells(self, index: int, near: float, far: float, slots: range, occupied: set[int]) -> list[Cell]:
        """Puestos entre dos rectas de fondo (por el punto de fuga de la fila) y los separadores."""
        margin = STALL_MARGIN * (far - near)
        edges = [
            np.cross(_homogeneous(self.row.origin + self.direction * value), self.row_vp)
            for value in (near - margin, far + margin)
        ]
        cells = []
        for k in slots:
            ts = self.scale.to_t(np.array([self.w0 + self.step * (k - 0.5), self.w0 + self.step * (k + 0.5)]))
            dividers = [np.cross(_homogeneous(self.row.origin + self.row.axis * t), self.divider_vp) for t in ts]
            corners = np.array(
                [
                    _meet(dividers[0], edges[0]),
                    _meet(dividers[1], edges[0]),
                    _meet(dividers[1], edges[1]),
                    _meet(dividers[0], edges[1]),
                ]
            )
            cells.append(Cell(corners, self.row.kind, index, k in occupied))
        return cells

    def own_cells(self, index: int) -> list[Cell]:
        return self.cells(index, self.low, self.high, range(self.last + 1), set(self.indices.tolist()))


def _best_band(models: list[RowModel], vehicle: Vehicle) -> RowModel | None:
    """La fila en cuya banda de fondo cae el vehículo (al menos la mitad), si hay alguna."""
    best, best_overlap = None, 0.5
    for model in models:
        if model.row.kind != vehicle.kind:
            continue
        low, high = model.band(vehicle)
        shortest = max(min(high - low, model.high - model.low), 1e-6)
        overlap = max(0.0, min(high, model.high) - max(low, model.low)) / shortest
        if overlap > best_overlap:
            best, best_overlap = model, overlap
    return best


def _absorb(models: list[RowModel], loose: list[Vehicle]) -> list[Vehicle]:
    """Une a su fila los carros que quedaron fuera porque su punto de apoyo se salía de la recta.

    Pasa en filas apretadas vistas de lado, donde los carros se tapan y el punto de apoyo
    zigzaguea. Un carro es de la fila si cae en su banda de fondo y en un puesto entero de ella.
    """
    left = []
    for vehicle in loose:
        model = _best_band(models, vehicle)
        if model is None:
            left.append(vehicle)
            continue
        slot = model.slot(vehicle)
        k = round(slot)
        if abs(slot - k) < 0.3 and -0.5 <= slot <= model.last + 0.5:
            # Si el puesto ya tiene carro, es el mismo vehículo visto en dos siluetas.
            if k not in model.indices.tolist():
                model.row.members.append(vehicle)
            continue
        left.append(vehicle)
    return left


# --- 6: filas espalda con espalda ---------------------------------------------------------


def _mirror_cells(model: RowModel, index: int, loose: list[Vehicle]) -> tuple[list[Cell], list[Vehicle]]:
    """La fila de enfrente: carros sueltos pegados a la fila, del otro lado de su fondo o frente."""
    depth = model.high - model.low
    parked = [v.length_axis for v in model.row.members if v.length_axis is not None]
    parked_axis = _undirected_mean(parked) if parked else None
    sides: dict[int, list[tuple[Vehicle, int, tuple[float, float]]]] = {1: [], -1: []}
    for vehicle in loose:
        if vehicle.kind != model.row.kind:
            continue
        # Un carro que entra o sale va a lo largo del pasillo, atravesado respecto de los
        # estacionados: no abre una fila de enfrente.
        if parked_axis is None or vehicle.length_axis is None:
            continue
        if abs(float(vehicle.length_axis @ parked_axis)) < np.cos(np.radians(MIRROR_MAX_TURN_DEG)):
            continue
        k = round(model.slot(vehicle))
        if k < -1 or k > model.last + 1:
            continue
        low, high = model.band(vehicle)
        # Del otro lado del fondo (o del frente), casi sin solaparse y sin dejar pasillo.
        if abs(low - model.high) < 0.45 * depth and high - model.high > 0.5 * depth:
            sides[1].append((vehicle, k, (low, high)))
        elif abs(high - model.low) < 0.45 * depth and model.low - low > 0.5 * depth:
            sides[-1].append((vehicle, k, (low, high)))

    cells: list[Cell] = []
    used: list[Vehicle] = []
    for side, found in sides.items():
        if not found:
            continue
        if side == 1:
            near, far = model.high, float(np.median([band[1] for _, _, band in found]))
        else:
            near, far = float(np.median([band[0] for _, _, band in found])), model.low
        ks = [k for _, k, _ in found]
        slots = range(min(0, *ks), max(model.last, *ks) + 1)
        cells.extend(model.cells(index, near, far, slots, set(ks)))
        index += 1
        used.extend(vehicle for vehicle, _, _ in found)
    return cells, used


# --- Todo junto -----------------------------------------------------------------------------


@dataclass
class Proposal:
    cells: list[Cell]
    rows: list[Row]
    loose: list[Vehicle]
    perspective: Perspective


def _sane(cells: list[Cell], members: list[Vehicle], shape: tuple[int, int]) -> bool:
    """Una fila mal armada (carros de filas distintas, paso equivocado) da celdas imposibles:
    cruzadas, mucho más grandes que sus carros o fuera de la imagen."""
    height, width = shape
    biggest = max(float(v.mask.sum()) for v in members)
    for cell in cells:
        corners = cell.corners
        if not np.isfinite(corners).all():
            return False
        if (corners[:, 0] < -0.25 * width).any() or (corners[:, 0] > 1.25 * width).any():
            return False
        if (corners[:, 1] < -0.25 * height).any() or (corners[:, 1] > 1.25 * height).any():
            return False
        contour = corners.astype(np.float32)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) > MAX_CELL_AREA * biggest:
            return False
    return True


def _own_cell(vehicle: Vehicle, index: int) -> Cell:
    """Celda de un carro que no se pudo ubicar en una fila: el rectángulo de su huella."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(vehicle.contact().astype(np.float32))
    # La huella vista de la llanta es delgada: se le da al menos un tercio de su largo de fondo.
    w, h = max(w, h / 3), max(h, w / 3)
    corners = cv2.boxPoints(((cx, cy), (w * (1 + 2 * STALL_MARGIN), h * (1 + 2 * STALL_MARGIN)), angle))
    return Cell(np.array(corners, float), vehicle.kind, index, True)


def propose(vehicles: list[Vehicle], shape: tuple[int, int]) -> Proposal:
    rows, loose = find_rows(vehicles)
    perspective = estimate_perspective(rows, shape)

    # Los carros que se salieron de la recta de su fila vuelven a ella; luego se reajusta.
    loose = _absorb([RowModel(row, perspective) for row in rows], loose)
    for row in rows:
        row.origin, row.axis = _fit_line(np.array([v.ground for v in row.members]))
        row.members.sort(key=lambda v: row.position(v.ground))

    # Con la perspectiva ya conocida, dos carros alcanzan para una fila si apuntan bien.
    def consistent(row: Row) -> bool:
        return min(
            _angle_to(row.origin, row.axis, perspective.rows), _angle_to(row.origin, row.axis, perspective.dividers)
        ) < 2 * VANISHING_TOLERANCE_DEG

    pairs, loose = find_rows(loose, minimum=2, accept=consistent)
    rows += pairs

    cells: list[Cell] = []
    good_rows: list[Row] = []
    # Carros de filas que dieron celdas imposibles: sí están en una fila, pero mal medida.
    broken: list[Vehicle] = []
    for row in rows:
        model = RowModel(row, perspective)
        own = model.own_cells(len(good_rows))
        if _sane(own, row.members, shape):
            cells.extend(own)
            good_rows.append(row)
        else:
            broken.extend(row.members)

    for row in list(good_rows):
        model = RowModel(row, perspective)
        mirrored, used = _mirror_cells(model, max((c.row for c in cells), default=-1) + 1, loose)
        if mirrored and _sane(mirrored, used, shape):
            cells.extend(mirrored)
            loose = [v for v in loose if all(v is not u for u in used)]

    # Los de una fila mal medida reciben la celda de su huella. Un vehículo suelto, lejos de toda
    # fila (uno detenido en el pasillo, un reflejo en un edificio), no: no hay cómo saber que
    # es un puesto, y el administrador lo traza a mano si lo es.
    next_index = max((c.row for c in cells), default=-1) + 1
    for offset, vehicle in enumerate(broken):
        cells.append(_own_cell(vehicle, next_index + offset))
    return Proposal(_without_overlaps(cells), good_rows, loose, perspective)


def _overlap_ratio(a: Cell, b: Cell) -> float:
    """Qué fracción de `a` queda dentro de `b`."""
    area = cv2.contourArea(a.corners.astype(np.float32))
    if area <= 0:
        return 1.0
    inter, _ = cv2.intersectConvexConvex(a.corners.astype(np.float32), b.corners.astype(np.float32))
    return float(inter) / area


def _without_overlaps(cells: list[Cell]) -> list[Cell]:
    """Una celda deducida que se monta sobre otra sobra: manda la que tiene un vehículo visto,
    y entre dos deducidas, la primera."""
    kept = [cell for cell in cells if cell.observed]
    for cell in cells:
        if cell.observed:
            continue
        if all(_overlap_ratio(cell, other) <= MAX_DEDUCED_OVERLAP for other in kept):
            kept.append(cell)
    order = {id(cell): i for i, cell in enumerate(cells)}
    return sorted(kept, key=lambda cell: order[id(cell)])


# --- Evaluación contra celdas trazadas a mano ------------------------------------------------


def evaluate(cells: list[Cell], reference: list[np.ndarray], occupied: list[bool] | None = None) -> dict[str, float]:
    """Compara las celdas propuestas con las reales (trazadas a mano o de una simulación).

    Se emparejan una a una maximizando el solapamiento; un par cuenta como acierto si el centro
    de cada celda cae dentro de la otra (así un vehículo estacionado contaría en el mismo puesto).
    """
    from scipy.optimize import linear_sum_assignment
    from shapely.geometry import Polygon

    real = [Polygon(np.asarray(points, float)).buffer(0) for points in reference]
    mine = [Polygon(cell.corners).buffer(0) for cell in cells]
    iou = np.zeros((len(real), len(mine)))
    for i, a in enumerate(real):
        for j, b in enumerate(mine):
            if a.intersects(b):
                inter = a.intersection(b).area
                iou[i, j] = inter / (a.area + b.area - inter)
    found: list[tuple[int, int]] = []
    if len(real) and len(mine):
        for i, j in zip(*linear_sum_assignment(-iou)):
            if iou[i, j] > 0 and a_contains(real[i], mine[j]) and a_contains(mine[j], real[i]):
                found.append((int(i), int(j)))
    ious = [iou[i, j] for i, j in found]
    occupied = occupied or [False] * len(real)
    return {
        "reference": len(real),
        "proposed": len(mine),
        "found": len(found),
        "proposed_observed": sum(cell.observed for cell in cells),
        "found_observed": sum(cells[j].observed for _, j in found),
        "proposed_deduced": sum(not cell.observed for cell in cells),
        "found_deduced": sum(not cells[j].observed for _, j in found),
        "occupied": sum(occupied),
        "found_occupied": sum(occupied[i] for i, _ in found),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "pairs": found,
    }


def a_contains(outer, inner) -> bool:
    return bool(outer.contains(inner.centroid))


def occupancy_agreement(
    cells: list[Cell], reference: list[np.ndarray], frames: list[list[Observation]], pairs: list[tuple[int, int]]
) -> dict[str, float]:
    """Cuánto se parece la ocupación medida con las celdas propuestas a la de las celdas reales.

    Se usa la misma regla del servicio: un vehículo ocupa la celda donde cae el centro inferior
    de su caja. Se compara el conteo por cuadro y el estado de cada celda real con el de su
    pareja propuesta (sin pareja, la propuesta la da siempre por libre).
    """
    from shapely.geometry import Point, Polygon

    real = [Polygon(np.asarray(points, float)).buffer(0) for points in reference]
    mine = [Polygon(cell.corners).buffer(0) for cell in cells]
    partner = dict(pairs)
    counts_real, counts_mine, same, total = [], [], 0, 0
    for frame in frames:
        feet = [Point((o.polygon[:, 0].min() + o.polygon[:, 0].max()) / 2, o.polygon[:, 1].max()) for o in frame]
        state_real = [any(p.covers(f) for f in feet) for p in real]
        state_mine = [any(p.covers(f) for f in feet) for p in mine]
        counts_real.append(sum(state_real))
        counts_mine.append(sum(state_mine))
        for i, occupied in enumerate(state_real):
            total += 1
            same += occupied == (state_mine[partner[i]] if i in partner else False)
    counts_real, counts_mine = np.array(counts_real), np.array(counts_mine)
    return {
        "occupied_real": float(counts_real.mean()) if len(frames) else 0.0,
        "occupied_proposed": float(counts_mine.mean()) if len(frames) else 0.0,
        "count_error": float(np.abs(counts_real - counts_mine).mean()) if len(frames) else 0.0,
        "cell_agreement": same / total if total else 0.0,
    }
