import decimal
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _decimal_places(value: float) -> int:
    """Number of decimal places implied by a step value."""
    d = decimal.Decimal(str(value))
    _, _, exponent = d.as_tuple()
    return max(0, -exponent)


@dataclass
class Parameter:
    name: str
    min_val: float
    max_val: float
    step: float
    param_type: str  # "int" or "float"

    def __post_init__(self):
        self._precision: int = _decimal_places(self.step)
        self._n_steps: int = round((self.max_val - self.min_val) / self.step)

    def valid_values(self) -> list:
        """Returns all valid grid values for this parameter."""
        vals = [
            round(self.min_val + i * self.step, self._precision)
            for i in range(self._n_steps + 1)
        ]
        if self.param_type == "int":
            return [int(v) for v in vals]
        return vals

    def snap(self, value: float) -> float | int:
        """Snaps an arbitrary float to the nearest valid grid value."""
        idx = round((value - self.min_val) / self.step)
        idx = max(0, min(self._n_steps, idx))
        snapped = round(self.min_val + idx * self.step, self._precision)
        if self.param_type == "int":
            return int(snapped)
        return snapped


class ParameterSpace:
    def __init__(self, parameters: list[Parameter]):
        self.parameters = parameters

    @classmethod
    def from_json(cls, path: str | Path) -> "ParameterSpace":
        """Load parameter space from a JSON config file."""
        data = json.loads(Path(path).read_text())
        params = [
            Parameter(
                name=p["name"],
                min_val=float(p["min"]),
                max_val=float(p["max"]),
                step=float(p["step"]),
                param_type=p["type"],
            )
            for p in data["parameters"]
        ]
        return cls(params)

    def total_combinations(self) -> int:
        """Returns count of all valid grid combinations."""
        result = 1
        for p in self.parameters:
            result *= len(p.valid_values())
        return result

    def all_combinations(self) -> list[dict]:
        """Returns every valid grid combination. Use with caution -- 679,140 for full space."""
        grids = [p.valid_values() for p in self.parameters]
        names = [p.name for p in self.parameters]
        return [dict(zip(names, combo)) for combo in itertools.product(*grids)]

    def sample_lhs(self, n: int, seed: int | None = None) -> list[dict]:
        """Returns n parameter dicts sampled via Latin Hypercube, snapped to grid steps."""
        from scipy.stats.qmc import LatinHypercube

        d = len(self.parameters)
        sampler = LatinHypercube(d=d, seed=seed)
        samples = sampler.random(n=n)  # shape (n, d), values in [0, 1)

        result = []
        for row in samples:
            params = {}
            for i, p in enumerate(self.parameters):
                scaled = p.min_val + row[i] * (p.max_val - p.min_val)
                params[p.name] = p.snap(scaled)
            result.append(params)
        return result

    def neighbors(self, point: dict, radius: int = 1) -> list[dict]:
        """Returns all parameter dicts within radius grid steps (Chebyshev distance).

        Does not include point itself. All returned points are within parameter bounds.
        """
        snapped_point = {p.name: p.snap(point[p.name]) for p in self.parameters}

        per_dim: list[list] = []
        for p in self.parameters:
            vals = p.valid_values()
            val = snapped_point[p.name]
            # Find index of val in grid
            idx = min(range(len(vals)), key=lambda i, v=val, vs=vals: abs(vs[i] - v))
            low = max(0, idx - radius)
            high = min(len(vals) - 1, idx + radius)
            per_dim.append(vals[low : high + 1])

        names = [p.name for p in self.parameters]
        neighbors = []
        for combo in itertools.product(*per_dim):
            candidate = dict(zip(names, combo))
            if candidate != snapped_point:
                neighbors.append(candidate)
        return neighbors

    def to_optuna_distributions(self) -> dict:
        """Returns dict of optuna distribution objects for Bayesian stage."""
        try:
            import optuna.distributions as od
        except ImportError:
            raise ImportError("optuna is required. Install with: pip install optuna")

        dists = {}
        for p in self.parameters:
            if p.param_type == "int":
                dists[p.name] = od.IntDistribution(
                    low=int(p.min_val), high=int(p.max_val), step=int(p.step)
                )
            else:
                dists[p.name] = od.FloatDistribution(
                    low=p.min_val, high=p.max_val, step=p.step
                )
        return dists
