"""Parqueaderos simulados para medir la detección automática de puestos desde cualquier cámara.

Se arma un parqueadero en el suelo (metros) con filas de puestos de 2,5 x 5 m, se estacionan
carros (cajas de 1,8 x 4,4 x 1,5 m, con algo de desorden) y se "filma" con una cámara puntual.
Cada cuadro entrega las siluetas visibles de los carros, como las daría la segmentación: los
cercanos tapan a los lejanos, algunos no se detectan y por el pasillo pasa un carro.

    python scripts/admin/simular_puestos.py            # tabla de resultados en varias vistas
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auto_cells import Observation, evaluate, propose, stationary_vehicles  # noqa: E402

WIDTH, HEIGHT = 1280, 720
STALL = (2.5, 5.0)
CAR = (1.8, 4.4, 1.5)


def camera(position, target, focal: float = 900.0) -> np.ndarray:
    """Matriz de proyección (3 x 4) de una cámara en `position` que mira a `target`."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    intrinsics = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]])
    return intrinsics @ np.column_stack([rotation, -rotation @ position])


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, float)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    image = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return image[:, :2] / image[:, 2:3]


@dataclass(frozen=True)
class GroundRow:
    """Una fila de puestos: esquina de inicio, dirección a lo largo y hacia el fondo."""

    start: tuple[float, float]
    along: tuple[float, float]
    inward: tuple[float, float]
    stalls: int

    def stall(self, k: int) -> np.ndarray:
        start, along, inward = (np.asarray(v, float) for v in (self.start, self.along, self.inward))
        a = start + along * STALL[0] * k
        b = a + along * STALL[0]
        return np.array([a, b, b + inward * STALL[1], a + inward * STALL[1]])


# Un parqueadero como el real: dos filas laterales y una doble (espalda con espalda) al centro,
# con los pasillos de 6 m. Las filas corren a lo largo del eje y.
LOT = [
    GroundRow((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), 10),
    GroundRow((11.0, 0.0), (0.0, 1.0), (1.0, 0.0), 8),
    GroundRow((21.0, 0.0), (0.0, 1.0), (-1.0, 0.0), 8),
    GroundRow((32.0, 0.0), (0.0, 1.0), (-1.0, 0.0), 10),
]
CENTER = np.array([16.0, 12.0, 0.0])

# Vistas: desde atrás y alto (como el dron), más baja, girada, desde una esquina y casi cenital.
VIEWS = {
    "atras-alta": ([16, -22, 16], 900.0),
    "atras-baja": ([16, -16, 7], 900.0),
    "girada": ([0, -20, 13], 900.0),
    "esquina": ([-12, -14, 12], 800.0),
    "cenital": ([16, 4, 40], 700.0),
}


def _car_corners(center: np.ndarray, along: np.ndarray, inward: np.ndarray, yaw: float) -> np.ndarray:
    """Las 8 esquinas de un carro: su largo va hacia el fondo del puesto."""
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    length_axis, width_axis = rotation @ inward, rotation @ along
    base = [
        center + length_axis * sx * CAR[1] / 2 + width_axis * sy * CAR[0] / 2
        for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
    ]
    return np.array([[*p, z] for z in (0.0, CAR[2]) for p in base])


@dataclass
class Scene:
    matrix: np.ndarray
    cells: list[np.ndarray]  # puestos reales en la imagen
    occupied: list[bool]
    frames: list[list[Observation]]


def simulate(
    matrix: np.ndarray,
    occupancy: float = 0.65,
    frames: int = 30,
    miss_rate: float = 0.08,
    jitter_px: float = 1.5,
    seed: int = 0,
    lot: list[GroundRow] = LOT,
) -> Scene:
    rng = np.random.default_rng(seed)
    cells, occupied, cars = [], [], []
    camera_position = -np.linalg.inv(matrix[:, :3]) @ matrix[:, 3]
    for row in lot:
        along, inward = np.asarray(row.along, float), np.asarray(row.inward, float)
        for k in range(row.stalls):
            quad = row.stall(k)
            cells.append(project(matrix, quad))
            taken = rng.random() < occupancy
            occupied.append(taken)
            if taken:
                center = quad.mean(axis=0) + along * rng.normal(0, 0.12) + inward * rng.normal(0, 0.2)
                cars.append(_car_corners(center, along, inward, rng.normal(0, np.radians(3))))

    observations = []
    for index in range(frames):
        # Un carro que recorre el pasillo central: no es un puesto.
        moving = _car_corners(np.array([16.0, -4.0 + index * 1.2]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0)
        observations.append(_render(matrix, camera_position, [*cars, moving], miss_rate, jitter_px, rng))
    return Scene(matrix, cells, occupied, observations)


def _render(matrix, camera_position, cars, miss_rate, jitter_px, rng) -> list[Observation]:
    """Siluetas visibles: se pintan de lejos a cerca, así los cercanos tapan a los lejanos."""
    canvas = np.zeros((HEIGHT, WIDTH), np.int32)
    order = sorted(range(len(cars)), key=lambda i: -np.linalg.norm(cars[i].mean(axis=0) - camera_position))
    for i in order:
        hull = cv2.convexHull(project(matrix, cars[i]).astype(np.float32))
        cv2.fillPoly(canvas, [np.round(hull).astype(np.int32)], i + 1)
    frame = []
    for i in range(len(cars)):
        if rng.random() < miss_rate:
            continue
        mask = (canvas == i + 1).astype(np.uint8)
        if mask.sum() < 60:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        if len(polygon) < 3:
            continue
        frame.append(Observation(2, polygon + rng.normal(0, jitter_px, polygon.shape)))
    return frame


def run(scene: Scene) -> dict[str, float]:
    vehicles = stationary_vehicles(scene.frames, (HEIGHT, WIDTH))
    proposal = propose(vehicles, (HEIGHT, WIDTH))
    return evaluate(proposal.cells, scene.cells, occupied=scene.occupied)


def main() -> None:
    print(f"{'vista':<12}{'ocupación':>10}{'reales':>8}{'propuestas':>11}{'aciertos':>10}"
          f"{'observ.':>9}{'deduc.':>8}{'IoU':>6}")
    totals = []
    for name, (position, focal) in VIEWS.items():
        matrix = camera(position, CENTER, focal)
        for occupancy in (0.4, 0.65, 0.9):
            for seed in range(3):
                result = run(simulate(matrix, occupancy=occupancy, seed=seed))
                totals.append(result)
                if seed == 0:
                    print(
                        f"{name:<12}{occupancy:>10.0%}{result['reference']:>8}{result['proposed']:>11}"
                        f"{result['found']:>10}{result['found_observed']:>5}/{result['proposed_observed']:<3}"
                        f"{result['found_deduced']:>4}/{result['proposed_deduced']:<3}{result['mean_iou']:>6.2f}"
                    )
    found = sum(r["found"] for r in totals)
    print(
        f"\nTotal ({len(totals)} escenas): {found}/{sum(r['reference'] for r in totals)} puestos encontrados "
        f"({found / sum(r['reference'] for r in totals):.0%}), precisión "
        f"{found / max(sum(r['proposed'] for r in totals), 1):.0%}, "
        f"puestos ocupados encontrados {sum(r['found_occupied'] for r in totals)}/{sum(r['occupied'] for r in totals)}"
    )


if __name__ == "__main__":
    main()
