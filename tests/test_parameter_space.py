from pathlib import Path

import pytest

from optimizer.pipeline.parameter_space import Parameter, ParameterSpace

PROJECT_ROOT = Path(__file__).parent.parent
SPACE_JSON = PROJECT_ROOT / "config" / "parameter_space.json"


@pytest.fixture
def space():
    return ParameterSpace.from_json(SPACE_JSON)


# ---------------------------------------------------------------------------
# from_json + total_combinations
# ---------------------------------------------------------------------------

def test_from_json_loads_params(space):
    assert len(space.parameters) == 3


def test_total_combinations(space):
    assert space.total_combinations() == 180


# ---------------------------------------------------------------------------
# sample_lhs
# ---------------------------------------------------------------------------

def test_sample_lhs_returns_exact_count(space):
    samples = space.sample_lhs(150, seed=42)
    assert len(samples) == 150


def test_sample_lhs_values_within_bounds(space):
    samples = space.sample_lhs(50, seed=1)
    for sample in samples:
        for p in space.parameters:
            val = sample[p.name]
            assert p.min_val <= val <= p.max_val, (
                f"{p.name}={val} out of [{p.min_val}, {p.max_val}]"
            )


def test_sample_lhs_values_snapped_to_grid(space):
    samples = space.sample_lhs(50, seed=2)
    for sample in samples:
        for p in space.parameters:
            val = sample[p.name]
            assert val in p.valid_values(), (
                f"{p.name}={val} not on grid {p.valid_values()}"
            )


# ---------------------------------------------------------------------------
# neighbors
# ---------------------------------------------------------------------------

def test_neighbors_within_bounds(space):
    point = space.sample_lhs(1, seed=99)[0]
    neighbors = space.neighbors(point, radius=1)
    for neighbor in neighbors:
        for p in space.parameters:
            val = neighbor[p.name]
            assert p.min_val <= val <= p.max_val


def test_neighbors_does_not_include_point(space):
    point = space.sample_lhs(1, seed=7)[0]
    neighbors = space.neighbors(point, radius=1)
    # Snap point for fair comparison
    snapped = {p.name: p.snap(point[p.name]) for p in space.parameters}
    assert snapped not in neighbors


def test_neighbors_corner_has_fewer_than_interior(space):
    corner = {p.name: p.snap(p.min_val) for p in space.parameters}
    interior = {p.name: p.snap((p.min_val + p.max_val) / 2) for p in space.parameters}
    corner_n = len(space.neighbors(corner, radius=1))
    interior_n = len(space.neighbors(interior, radius=1))
    assert corner_n < interior_n


# ---------------------------------------------------------------------------
# snap
# ---------------------------------------------------------------------------

def test_snap_int_param():
    p = Parameter(name="X", min_val=5, max_val=15, step=2, param_type="int")
    assert p.snap(6.3) == 7
    assert p.snap(5.0) == 5
    assert p.snap(14.9) == 15
    assert isinstance(p.snap(6.3), int)


def test_snap_float_param():
    p = Parameter(name="Y", min_val=0.75, max_val=0.95, step=0.05, param_type="float")
    assert p.snap(0.826) == 0.85
    assert p.snap(0.777) == 0.80
    assert p.snap(0.749) == 0.75  # clipped to min
    assert p.snap(0.96) == 0.95   # clipped to max
