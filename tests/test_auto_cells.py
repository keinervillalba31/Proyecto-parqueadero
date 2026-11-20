"""Detección automática de puestos: geometría pura y parqueaderos simulados (sin el modelo)."""

import numpy as np
import pytest

from admin.simular_puestos import CENTER, HEIGHT, VIEWS, WIDTH, camera, project, simulate
from app.auto_cells import (
    Cell,
    Observation,
    _lattice,
    _one_per_stall,
    _sane,
    evaluate,
    propose,
    stationary_vehicles,
)

SHAPE = (HEIGHT, WIDTH)


def square(x, y, size=40):
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]], float)


def test_a_car_driving_through_is_not_a_stall():
    parked = square(100, 100)
    frames = [[Observation(2, parked), Observation(2, square(300 + 30 * i, 400))] for i in range(10)]
    vehicles = stationary_vehicles(frames, SHAPE)
    assert len(vehicles) == 1
    assert vehicles[0].ground[0] == pytest.approx(120, abs=2)


def test_a_car_seen_only_a_moment_is_not_a_stall():
    frames = [[Observation(2, square(100, 100))] if i < 3 else [] for i in range(10)]
    assert stationary_vehicles(frames, SHAPE) == []


def test_gaps_of_two_steps_are_an_empty_stall():
    w0, step, indices = _lattice(np.array([0.0, 1.02, 1.98, 4.01, 5.0]))
    assert step == pytest.approx(1.0, abs=0.03)
    assert indices.tolist() == [0, 1, 2, 4, 5]


def test_noise_never_invents_more_than_two_empty_stalls_between_neighbours():
    _, step, indices = _lattice(np.array([0.0, 1.0, 2.37, 3.1, 4.6]))
    assert np.diff(np.sort(indices)).max() <= 3


def test_a_row_that_gives_impossible_cells_is_rejected():
    frames = [[Observation(2, square(100, 100))]] * 4
    vehicle = stationary_vehicles(frames, SHAPE)[0]
    good = Cell(square(95, 95, 50), "CAR", 0, True)
    huge = Cell(square(0, 0, 700), "CAR", 0, True)
    crossed = Cell(np.array([[0, 0], [50, 50], [50, 0], [0, 50]], float), "CAR", 0, True)
    assert _sane([good], [vehicle], SHAPE)
    assert not _sane([good, huge], [vehicle], SHAPE)
    assert not _sane([crossed], [vehicle], SHAPE)


def run(view: str, occupancy: float, seed: int = 0) -> dict:
    position, focal = VIEWS[view]
    scene = simulate(camera(position, CENTER, focal), occupancy=occupancy, frames=12, seed=seed)
    proposal = propose(stationary_vehicles(scene.frames, SHAPE), SHAPE)
    return evaluate(proposal.cells, scene.cells, occupied=scene.occupied)


def test_from_above_almost_every_stall_is_found():
    result = run("cenital", 0.9)
    assert result["found"] >= 0.9 * result["reference"]
    assert result["found"] >= 0.9 * result["proposed"]


def test_empty_stalls_between_parked_cars_are_deduced():
    result = run("cenital", 0.65)
    assert result["proposed_deduced"] >= 3
    assert result["found_deduced"] >= 0.75 * result["proposed_deduced"]


def test_from_a_high_oblique_camera_most_parked_cars_get_their_stall():
    result = run("atras-alta", 0.9)
    assert result["found_occupied"] >= 0.75 * result["occupied"]


def test_the_simulated_camera_sees_the_lot():
    position, focal = VIEWS["atras-alta"]
    corners = project(camera(position, CENTER, focal), np.array([[0, 0], [32, 0], [32, 25], [0, 25]], float))
    assert (corners[:, 0] > 0).all() and (corners[:, 0] < WIDTH).all()
    assert (corners[:, 1] > 0).all() and (corners[:, 1] < HEIGHT).all()


def test_noise_cannot_shrink_the_step_below_half_a_stall():
    # Un carro visto en dos siluetas casi juntas no puede convertir la fila en medios puestos.
    _, step, indices = _lattice(np.array([0.0, 1.0, 1.05, 2.0, 3.0, 4.0]))
    assert step >= 0.45
    assert np.diff(np.sort(indices)).max() <= 2


def test_two_silhouettes_of_one_car_count_as_one_stall():
    frames = [[Observation(2, square(100, 100))]] * 4
    small = [[Observation(2, square(100, 100, 30))]] * 4
    big, little = stationary_vehicles(frames, SHAPE)[0], stationary_vehicles(small, SHAPE)[0]
    others = [stationary_vehicles([[Observation(2, square(x, 100))]] * 4, SHAPE)[0] for x in (300, 500, 700)]
    members = [big, little, *others]
    kept = _one_per_stall(members, np.array([0.0, 0.05, 1.0, 2.0, 3.0]))
    assert kept == [big, *others]
