from uuid import uuid4

import pytest

from optimizer.fitness.calmar import CalmarFitness
from optimizer.pipeline.parameter_space import Parameter, ParameterSpace
from optimizer.pipeline.pipeline import OptimizationPipeline, PipelineResult
from optimizer.runner.evaluation import Evaluation, EvaluationStatus


def _small_space() -> ParameterSpace:
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


def _make_eval(params: dict, fitness: float = 5.0, stage: str = "test") -> Evaluation:
    ev = Evaluation(id=uuid4().hex, parameters=params)
    ev.status = EvaluationStatus.SUCCESS
    ev.metrics = _make_metrics()
    ev.fitness_score = fitness
    ev.extracted_metrics = ev.metrics
    ev.stage = stage
    return ev


class FakeStage:
    """Fake OptimizationStage that returns pre-built evaluations."""

    def __init__(self, name: str, evals: list[Evaluation]):
        self._name = name
        self._evals = evals
        self.received_previous = None

    def run(self, space, fitness_fn, batch_runner, previous_results=None):
        self.received_previous = previous_results
        return self._evals


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_stages_run_in_order():
    """Stages run sequentially in the order given."""
    order = []

    class OrderStage:
        def __init__(self, label):
            self.label = label

        def run(self, space, fitness_fn, batch_runner, previous_results=None):
            order.append(self.label)
            return [_make_eval({"A": 20, "B": 0.5}, stage=self.label)]

    pipeline = OptimizationPipeline(
        stages=[OrderStage("first"), OrderStage("second"), OrderStage("third")]
    )
    pipeline.run(_small_space(), CalmarFitness(), lambda ps: ps)

    assert order == ["first", "second", "third"]


def test_accumulated_results_passed_to_subsequent_stages():
    """Each stage receives accumulated results from all prior stages."""
    ev1 = _make_eval({"A": 10, "B": 0.5}, fitness=3.0, stage="lhs")
    ev2 = _make_eval({"A": 20, "B": 0.5}, fitness=7.0, stage="bayesian")

    stage1 = FakeStage("lhs", [ev1])
    stage2 = FakeStage("bayesian", [ev2])

    pipeline = OptimizationPipeline(stages=[stage1, stage2])
    pipeline.run(_small_space(), CalmarFitness(), lambda ps: ps)

    # stage1 receives None (no previous)
    assert stage1.received_previous is None
    # stage2 receives stage1's results
    assert stage2.received_previous is not None
    assert len(stage2.received_previous) == 1
    assert stage2.received_previous[0].id == ev1.id


def test_best_evaluation_tracked():
    """Pipeline tracks the best evaluation across all main stages."""
    ev1 = _make_eval({"A": 10, "B": 0.5}, fitness=3.0, stage="lhs")
    ev2 = _make_eval({"A": 20, "B": 0.5}, fitness=9.0, stage="bayesian")
    ev3 = _make_eval({"A": 30, "B": 1.0}, fitness=7.0, stage="local_grid")

    pipeline = OptimizationPipeline(stages=[
        FakeStage("lhs", [ev1]),
        FakeStage("bayesian", [ev2]),
        FakeStage("local_grid", [ev3]),
    ])
    result = pipeline.run(_small_space(), CalmarFitness(), lambda ps: ps)

    assert result.best_evaluation is not None
    assert result.best_evaluation.id == ev2.id
    assert result.best_score == 9.0


def test_exception_propagates():
    """If a stage raises, exception propagates to the caller."""

    class FailStage:
        def run(self, space, fitness_fn, batch_runner, previous_results=None):
            raise RuntimeError("stage exploded")

    pipeline = OptimizationPipeline(stages=[FailStage()])

    with pytest.raises(RuntimeError, match="stage exploded"):
        pipeline.run(_small_space(), CalmarFitness(), lambda ps: ps)


def test_all_evaluations_complete():
    """all_evaluations includes results from all stages."""
    ev1 = _make_eval({"A": 10, "B": 0.5}, fitness=3.0, stage="lhs")
    ev2 = _make_eval({"A": 20, "B": 0.5}, fitness=7.0, stage="bayesian")
    ev3 = _make_eval({"A": 30, "B": 1.0}, fitness=5.0, stage="local_grid")

    pipeline = OptimizationPipeline(
        stages=[
            FakeStage("lhs", [ev1]),
            FakeStage("bayesian", [ev2]),
            FakeStage("local_grid", [ev3]),
        ],
    )
    result = pipeline.run(_small_space(), CalmarFitness(), lambda ps: ps)

    assert len(result.all_evaluations) == 3
    all_ids = {ev.id for ev in result.all_evaluations}
    assert all_ids == {ev1.id, ev2.id, ev3.id}


def test_stage_results_dict_keys():
    """stage_results dict is keyed by class name."""
    ev1 = _make_eval({"A": 10, "B": 0.5}, stage="lhs")

    class MyCustomStage:
        def run(self, space, fitness_fn, batch_runner, previous_results=None):
            return [ev1]

    pipeline = OptimizationPipeline(stages=[MyCustomStage()])
    result = pipeline.run(_small_space(), CalmarFitness(), lambda ps: ps)

    assert "MyCustomStage" in result.stage_results
    assert len(result.stage_results["MyCustomStage"]) == 1


def test_empty_stages_no_crash():
    """Pipeline with no stages returns empty result cleanly."""
    pipeline = OptimizationPipeline(stages=[])
    result = pipeline.run(_small_space(), CalmarFitness(), lambda ps: ps)

    assert result.all_evaluations == []
    assert result.best_evaluation is None
    assert result.best_score == float("-inf")


def test_pipeline_result_defaults():
    """PipelineResult has sane defaults."""
    r = PipelineResult()
    assert r.all_evaluations == []
    assert r.stage_results == {}
    assert r.best_evaluation is None
    assert r.best_score == float("-inf")
