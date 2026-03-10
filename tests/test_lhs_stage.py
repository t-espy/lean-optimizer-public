from uuid import uuid4

import pytest

from optimizer.fitness.calmar import CalmarFitness
from optimizer.pipeline.lhs import LHSStage
from optimizer.pipeline.parameter_space import Parameter, ParameterSpace
from optimizer.runner.evaluation import Evaluation, EvaluationStatus


def _small_space() -> ParameterSpace:
    """2-parameter space for fast tests."""
    return ParameterSpace([
        Parameter(name="A", min_val=10, max_val=30, step=10, param_type="int"),
        Parameter(name="B", min_val=0.5, max_val=1.0, step=0.5, param_type="float"),
    ])


def _make_metrics() -> dict:
    """ExtractedMetrics.to_dict() output with valid CalmarFitness values."""
    return {
        "net_pnl": 23.4,
        "max_drawdown": 5.0,
        "trade_count": 50,
        "profit_factor": 1.79,
        "win_rate": 0.55,
        "sharpe_ratio": 1.45,
        "avg_trade": 0.468,
    }


def _mock_runner(fail_indices: set | None = None):
    """Returns a callable that mimics batch_runner behavior."""
    calls = []

    def runner(parameter_sets: list) -> list:
        calls.append(parameter_sets)
        evaluations = []
        for i, params in enumerate(parameter_sets):
            ev = Evaluation(id=uuid4().hex, parameters=params)
            if fail_indices and i in fail_indices:
                ev.status = EvaluationStatus.FAILED
                ev.metrics = None
            else:
                ev.status = EvaluationStatus.SUCCESS
                ev.metrics = _make_metrics()
            evaluations.append(ev)
        return evaluations

    runner.calls = calls
    return runner


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_lhs_calls_batch_runner_with_n_samples():
    space = _small_space()
    fitness = CalmarFitness()
    runner = _mock_runner()

    stage = LHSStage(n_samples=10, seed=42)
    results = stage.run(space, fitness, runner)

    assert len(runner.calls) == 1
    assert len(runner.calls[0]) == 10
    assert len(results) == 10


def test_lhs_evaluations_have_stage_label():
    space = _small_space()
    fitness = CalmarFitness()
    runner = _mock_runner()

    stage = LHSStage(n_samples=5, seed=1)
    results = stage.run(space, fitness, runner)

    assert all(ev.stage == "lhs" for ev in results)


def test_lhs_fitness_score_set_on_all():
    space = _small_space()
    fitness = CalmarFitness()
    runner = _mock_runner()

    stage = LHSStage(n_samples=5, seed=2)
    results = stage.run(space, fitness, runner)

    assert all(ev.fitness_score is not None for ev in results)


def test_lhs_extracted_metrics_attached():
    space = _small_space()
    fitness = CalmarFitness()
    runner = _mock_runner()

    stage = LHSStage(n_samples=5, seed=3)
    results = stage.run(space, fitness, runner)

    for ev in results:
        if ev.metrics is not None:
            assert ev.extracted_metrics is not None
            assert ev.extracted_metrics == ev.metrics


def test_lhs_failed_evals_returned_with_neginf_score():
    space = _small_space()
    fitness = CalmarFitness()
    runner = _mock_runner(fail_indices={0})

    stage = LHSStage(n_samples=3, seed=4)
    results = stage.run(space, fitness, runner)

    assert len(results) == 3
    failed = [ev for ev in results if ev.status == EvaluationStatus.FAILED]
    assert len(failed) == 1
    assert failed[0].fitness_score == float("-inf")
