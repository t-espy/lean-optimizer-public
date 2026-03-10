from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from optimizer.runner.evaluation import Evaluation, EvaluationStatus


FIVE_PARAM_SETS = [{"ticker": "SPY", "n": str(i)} for i in range(5)]


def _make_mock_pool(worker_name: str = "lean_optimizer_test") -> MagicMock:
    """Return a mock WorkerPool whose acquire() always returns a consistent mock worker."""
    mock_worker = MagicMock()
    mock_worker.worker_id = worker_name
    mock_worker.is_alive.return_value = True

    pool = MagicMock()
    pool.size = 5
    pool.acquire.return_value = mock_worker
    pool._worker = mock_worker  # convenience handle for assertions
    return pool


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_runs_all_five_sets(tmp_path):
    """All 5 parameter sets are evaluated; 5 Evaluations returned."""
    from optimizer.runner import batch_runner

    pool = _make_mock_pool()

    def fake_run(ev):
        ev.status = EvaluationStatus.SUCCESS
        ev.runtime_seconds = 1.0
        return ev

    pool._worker.run_backtest.side_effect = fake_run

    with patch.object(batch_runner.collector, "verify_result", return_value=None):
        results = batch_runner.run_batch(
            parameter_sets=FIVE_PARAM_SETS,
            pool=pool,
            results_root=tmp_path,
            artifacts_container_path="/Artifacts",
        )

    assert len(results) == 5
    assert pool.acquire.call_count == 5
    assert pool.release.call_count == 5


def test_continues_after_failure(tmp_path):
    """Failed evaluations don't stop the batch; all 5 are returned."""
    from optimizer.runner import batch_runner

    pool = _make_mock_pool()
    outcomes = [True, False, True, False, True]
    call_count = [0]

    def fake_run(ev):
        idx = call_count[0] % len(outcomes)
        call_count[0] += 1
        if outcomes[idx]:
            ev.status = EvaluationStatus.SUCCESS
        else:
            ev.status = EvaluationStatus.FAILED
            ev.error_message = "simulated failure"
        ev.runtime_seconds = 0.5
        return ev

    pool._worker.run_backtest.side_effect = fake_run

    with patch.object(batch_runner.collector, "verify_result", return_value=None):
        results = batch_runner.run_batch(
            parameter_sets=FIVE_PARAM_SETS,
            pool=pool,
            results_root=tmp_path,
            artifacts_container_path="/Artifacts",
        )

    assert len(results) == 5
    statuses = [r.status for r in results]
    assert EvaluationStatus.SUCCESS in statuses
    assert EvaluationStatus.FAILED in statuses


def test_accepts_parameter_list_directly(tmp_path):
    """run_batch takes a plain list; no file I/O coupling."""
    from optimizer.runner import batch_runner

    pool = _make_mock_pool()
    custom_params = [{"ticker": "AAPL", "lookback": "30"}]

    def fake_run(ev):
        ev.status = EvaluationStatus.SUCCESS
        ev.runtime_seconds = 1.0
        return ev

    pool._worker.run_backtest.side_effect = fake_run

    with patch.object(batch_runner.collector, "verify_result", return_value=None):
        results = batch_runner.run_batch(
            parameter_sets=custom_params,
            pool=pool,
            results_root=tmp_path,
            artifacts_container_path="/Artifacts",
        )

    assert len(results) == 1
    assert results[0].parameters == {"ticker": "AAPL", "lookback": "30"}


def test_worker_released_even_on_exception(tmp_path):
    """pool.release() is called even when run_backtest raises an exception."""
    from optimizer.runner import batch_runner

    pool = _make_mock_pool()
    pool._worker.run_backtest.side_effect = RuntimeError("container exploded")
    pool._worker.is_alive.return_value = False  # container died

    with patch.object(batch_runner.collector, "verify_result", return_value=None):
        results = batch_runner.run_batch(
            parameter_sets=[{"ticker": "SPY"}],
            pool=pool,
            results_root=tmp_path,
            artifacts_container_path="/Artifacts",
        )

    assert len(results) == 1
    assert results[0].status == EvaluationStatus.FAILED
    assert "container exploded" in results[0].error_message
    pool.release.assert_called_once()
    _, kwargs = pool.release.call_args
    assert kwargs.get("dead") is True or pool.release.call_args.args[1] is True


def test_dead_worker_flagged_when_container_dies(tmp_path):
    """pool.release(dead=True) when worker.is_alive() returns False after exception."""
    from optimizer.runner import batch_runner

    pool = _make_mock_pool()
    pool._worker.run_backtest.side_effect = Exception("exec_run failed")
    pool._worker.is_alive.return_value = False

    with patch.object(batch_runner.collector, "verify_result", return_value=None):
        batch_runner.run_batch(
            parameter_sets=[{"ticker": "SPY"}],
            pool=pool,
            results_root=tmp_path,
            artifacts_container_path="/Artifacts",
        )

    release_args = pool.release.call_args
    dead_flag = (
        release_args.kwargs.get("dead")
        if release_args.kwargs
        else release_args.args[1]
    )
    assert dead_flag is True


def test_alive_worker_not_flagged_dead_on_soft_failure(tmp_path):
    """pool.release(dead=False) when backtest fails but container is still alive."""
    from optimizer.runner import batch_runner

    pool = _make_mock_pool()

    def fake_run(ev):
        ev.status = EvaluationStatus.FAILED
        ev.runtime_seconds = 0.5
        raise RuntimeError("backtest logic error")

    pool._worker.run_backtest.side_effect = fake_run
    pool._worker.is_alive.return_value = True  # container survived

    with patch.object(batch_runner.collector, "verify_result", return_value=None):
        batch_runner.run_batch(
            parameter_sets=[{"ticker": "SPY"}],
            pool=pool,
            results_root=tmp_path,
            artifacts_container_path="/Artifacts",
        )

    release_args = pool.release.call_args
    dead_flag = (
        release_args.kwargs.get("dead")
        if release_args.kwargs
        else release_args.args[1]
    )
    assert dead_flag is False


def test_worker_id_set_on_evaluation(tmp_path):
    """Each evaluation gets the worker_id of the container that ran it."""
    from optimizer.runner import batch_runner

    pool = _make_mock_pool(worker_name="lean_optimizer_abc12345")

    def fake_run(ev):
        ev.status = EvaluationStatus.SUCCESS
        ev.runtime_seconds = 1.0
        return ev

    pool._worker.run_backtest.side_effect = fake_run

    with patch.object(batch_runner.collector, "verify_result", return_value=None):
        results = batch_runner.run_batch(
            parameter_sets=[{"ticker": "SPY"}],
            pool=pool,
            results_root=tmp_path,
            artifacts_container_path="/Artifacts",
        )

    assert results[0].worker_id == "lean_optimizer_abc12345"
