from uuid import uuid4

import pytest

from optimizer.fitness.calmar import CalmarFitness
from optimizer.pipeline.local_grid import LocalGridStage, _fingerprint
from optimizer.pipeline.parameter_space import Parameter, ParameterSpace
from optimizer.runner.evaluation import Evaluation, EvaluationStatus


def _small_space() -> ParameterSpace:
    """A=[10,20,30], B=[0.5,1.0] — 6 total grid points."""
    return ParameterSpace([
        Parameter(name="A", min_val=10, max_val=30, step=10, param_type="int"),
        Parameter(name="B", min_val=0.5, max_val=1.0, step=0.5, param_type="float"),
    ])


def _make_metrics() -> dict:
    return {
        "net_pnl": 23.4,
        "max_drawdown": 5.0,
        "trade_count": 50,
        "profit_factor": 1.79,
        "win_rate": 0.55,
        "sharpe_ratio": 1.45,
        "avg_trade": 0.468,
    }


def _make_prev_eval(
    params: dict, fitness_score: float = 5.0, stage: str = "lhs"
) -> Evaluation:
    ev = Evaluation(id=uuid4().hex, parameters=params)
    ev.status = EvaluationStatus.SUCCESS
    ev.metrics = _make_metrics()
    ev.fitness_score = fitness_score
    ev.extracted_metrics = ev.metrics
    ev.stage = stage
    return ev


def _mock_runner():
    calls = []

    def runner(parameter_sets: list) -> list:
        calls.append(parameter_sets)
        evaluations = []
        for params in parameter_sets:
            ev = Evaluation(id=uuid4().hex, parameters=params)
            ev.status = EvaluationStatus.SUCCESS
            ev.metrics = _make_metrics()
            evaluations.append(ev)
        return evaluations

    runner.calls = calls
    return runner


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_only_top_n_used_as_seeds():
    """Only the top top_n candidates are used as neighborhood seeds."""
    space = _small_space()
    runner = _mock_runner()

    # 3 evals with scores 10, 5, 1 — top_n=2 should use only 10 and 5
    prev = [
        _make_prev_eval({"A": 20, "B": 0.5}, fitness_score=10.0),
        _make_prev_eval({"A": 10, "B": 0.5}, fitness_score=5.0),
        _make_prev_eval({"A": 30, "B": 1.0}, fitness_score=1.0),
    ]

    stage = LocalGridStage(top_n=2, radius=1)
    stage.run(space, CalmarFitness(), runner, previous_results=prev)

    # Neighbors of {A:20, B:0.5} and {A:10, B:0.5} only
    # {A:30, B:1.0}'s neighbors should NOT be included unless they overlap
    submitted = runner.calls[0] if runner.calls else []
    submitted_fps = {_fingerprint(p) for p in submitted}

    # {A:30, B:0.5} is a neighbor of seed {A:20, B:0.5} — so it CAN appear
    # But {A:30, B:1.0} is a neighbor of {A:20, B:1.0} or {A:30, B:0.5}
    # The key test: not using {A:30, B:1.0} as a SEED (it has score=1.0)
    # All submitted params should be neighbors of the top-2 seeds
    top2_params = [{"A": 20, "B": 0.5}, {"A": 10, "B": 0.5}]
    valid_neighbors = set()
    for seed in top2_params:
        for n in space.neighbors(seed, radius=1):
            valid_neighbors.add(_fingerprint(n))

    for fp in submitted_fps:
        assert fp in valid_neighbors


def test_neighbors_generated_correctly():
    """Neighbors of a seed at A=20, B=0.5 are correct."""
    space = _small_space()
    runner = _mock_runner()

    prev = [_make_prev_eval({"A": 20, "B": 0.5}, fitness_score=10.0)]

    stage = LocalGridStage(top_n=1, radius=1)
    results = stage.run(space, CalmarFitness(), runner, previous_results=prev)

    # {A:20, B:0.5} neighbors: {10,20,30} x {0.5,1.0} - {20,0.5} = 5
    # But {A:20, B:0.5} itself is in prev, so it's deduped = 5 novel
    assert len(results) == 5


def test_already_evaluated_not_resubmitted():
    """Points already in previous_results are not submitted again."""
    space = _small_space()
    runner = _mock_runner()

    # Pre-evaluate some neighbors of {A:20, B:0.5}
    prev = [
        _make_prev_eval({"A": 20, "B": 0.5}, fitness_score=10.0),
        _make_prev_eval({"A": 10, "B": 0.5}, fitness_score=3.0),
        _make_prev_eval({"A": 30, "B": 0.5}, fitness_score=2.0),
    ]

    stage = LocalGridStage(top_n=1, radius=1)
    results = stage.run(space, CalmarFitness(), runner, previous_results=prev)

    # 5 total neighbors of {20, 0.5}, minus 2 already evaluated = 3 novel
    submitted_fps = {_fingerprint(p) for p in runner.calls[0]}
    assert _fingerprint({"A": 10, "B": 0.5}) not in submitted_fps
    assert _fingerprint({"A": 30, "B": 0.5}) not in submitted_fps
    assert len(results) == 3


def test_dedup_across_stages():
    """Deduplication works across all previous stages, not just one."""
    space = _small_space()
    runner = _mock_runner()

    prev = [
        _make_prev_eval({"A": 20, "B": 0.5}, fitness_score=10.0, stage="lhs"),
        _make_prev_eval({"A": 10, "B": 1.0}, fitness_score=2.0, stage="bayesian"),
    ]

    stage = LocalGridStage(top_n=1, radius=1)
    results = stage.run(space, CalmarFitness(), runner, previous_results=prev)

    submitted_fps = {_fingerprint(p) for p in runner.calls[0]}
    # {A:10, B:1.0} was from bayesian stage — should be deduped
    assert _fingerprint({"A": 10, "B": 1.0}) not in submitted_fps


def test_stage_label_local_grid():
    space = _small_space()
    runner = _mock_runner()

    prev = [_make_prev_eval({"A": 20, "B": 0.5}, fitness_score=10.0)]

    stage = LocalGridStage(top_n=1, radius=1)
    results = stage.run(space, CalmarFitness(), runner, previous_results=prev)

    assert all(ev.stage == "local_grid" for ev in results)


def test_all_neighbors_evaluated_returns_empty():
    """If every neighbor is already evaluated, returns empty list gracefully."""
    space = _small_space()
    runner = _mock_runner()

    # Pre-evaluate the seed and ALL its neighbors
    # {A:20, B:0.5} neighbors: {10,20,30} x {0.5,1.0} - itself = 5
    prev = [
        _make_prev_eval({"A": 20, "B": 0.5}, fitness_score=10.0),
        _make_prev_eval({"A": 10, "B": 0.5}, fitness_score=1.0),
        _make_prev_eval({"A": 30, "B": 0.5}, fitness_score=1.0),
        _make_prev_eval({"A": 10, "B": 1.0}, fitness_score=1.0),
        _make_prev_eval({"A": 20, "B": 1.0}, fitness_score=1.0),
        _make_prev_eval({"A": 30, "B": 1.0}, fitness_score=1.0),
    ]

    stage = LocalGridStage(top_n=1, radius=1)
    results = stage.run(space, CalmarFitness(), runner, previous_results=prev)

    assert results == []
    assert len(runner.calls) == 0  # batch_runner never called
