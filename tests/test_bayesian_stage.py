from uuid import uuid4

import pytest

from optimizer.fitness.calmar import CalmarFitness
from optimizer.pipeline.bayesian import BayesianStage
from optimizer.pipeline.parameter_space import Parameter, ParameterSpace
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


def _make_prev_eval(params: dict, fitness_score: float = 5.0) -> Evaluation:
    ev = Evaluation(id=uuid4().hex, parameters=params)
    ev.status = EvaluationStatus.SUCCESS
    ev.metrics = _make_metrics()
    ev.fitness_score = fitness_score
    ev.extracted_metrics = ev.metrics
    ev.stage = "lhs"
    return ev


def _mock_runner():
    """Returns a callable that mimics batch_runner."""
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


@pytest.fixture
def mock_optuna(mocker):
    """Mock optuna module inside bayesian.py."""
    mock_mod = mocker.patch("optimizer.pipeline.bayesian.optuna")

    study = mocker.MagicMock()
    mock_mod.create_study.return_value = study

    # trial returned by study.ask() — has params matching small space
    trial = mocker.MagicMock()
    trial.params = {"A": 20, "B": 0.5}
    study.ask.return_value = trial

    return mock_mod, study, trial


def test_warmstart_add_trial_called_before_ask(mock_optuna):
    mod, study, trial = mock_optuna
    runner = _mock_runner()

    prev = [_make_prev_eval({"A": 20, "B": 0.5}, fitness_score=5.0)]

    stage = BayesianStage(n_calls=1)
    stage.run(_small_space(), CalmarFitness(), runner, previous_results=prev)

    # add_trial called during warm-start
    study.add_trial.assert_called()
    # ask called for new evaluation
    study.ask.assert_called()


def test_inf_replaced_with_finite_in_warmstart(mock_optuna):
    mod, study, trial = mock_optuna
    runner = _mock_runner()

    prev = [_make_prev_eval({"A": 20, "B": 0.5}, fitness_score=float("-inf"))]

    stage = BayesianStage(n_calls=1)
    stage.run(_small_space(), CalmarFitness(), runner, previous_results=prev)

    # create_trial should have been called with -1000.0, not -inf
    call_kwargs = mod.trial.create_trial.call_args.kwargs
    assert call_kwargs["values"] == [-1000.0]


def test_n_calls_exact(mock_optuna):
    _, study, _ = mock_optuna
    runner = _mock_runner()

    stage = BayesianStage(n_calls=5)
    results = stage.run(_small_space(), CalmarFitness(), runner)

    assert len(results) == 5
    assert study.ask.call_count == 5
    assert study.tell.call_count == 5


def test_stage_label_bayesian(mock_optuna):
    runner = _mock_runner()

    stage = BayesianStage(n_calls=3)
    results = stage.run(_small_space(), CalmarFitness(), runner)

    assert all(ev.stage == "bayesian" for ev in results)


def test_fitness_score_set_on_all(mock_optuna):
    runner = _mock_runner()

    stage = BayesianStage(n_calls=3)
    results = stage.run(_small_space(), CalmarFitness(), runner)

    assert all(ev.fitness_score is not None for ev in results)


def test_returns_only_new_evals(mock_optuna):
    runner = _mock_runner()

    prev = [_make_prev_eval({"A": 20, "B": 0.5}, fitness_score=5.0)]

    stage = BayesianStage(n_calls=2)
    results = stage.run(_small_space(), CalmarFitness(), runner, previous_results=prev)

    assert len(results) == 2
    # No previous eval in results — all are new
    prev_ids = {ev.id for ev in prev}
    assert all(ev.id not in prev_ids for ev in results)
    assert all(ev.stage == "bayesian" for ev in results)
