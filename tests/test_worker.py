from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from optimizer.runner.evaluation import Evaluation, EvaluationStatus


def _make_worker(artifact_dir: Path, results_root: Path, harness_dir: Path = None):
    from optimizer.docker.worker import Worker
    if harness_dir is None:
        harness_dir = artifact_dir  # dummy path for tests
    return Worker(artifact_dir=artifact_dir, results_root=results_root, harness_dir=harness_dir)


def _mock_docker_client():
    client = MagicMock()
    container = MagicMock()
    container.short_id = "abc123"
    container.id = "abc123full"
    client.containers.run.return_value = container
    return client, container


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_start_creates_container_with_correct_volumes(tmp_path, monkeypatch):
    """Worker.start() calls containers.run with expected volume bindings including harness."""
    artifact_dir = tmp_path / "artifacts" / "deadbeef"
    artifact_dir.mkdir(parents=True)
    results_root = tmp_path / "results"
    results_root.mkdir()
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()

    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")
    monkeypatch.setenv("LEAN_DATA_CONTAINER_PATH", "/Lean/Data")
    monkeypatch.setenv("ARTIFACTS_CONTAINER_PATH", "/Artifacts")
    monkeypatch.setenv("RESULTS_CONTAINER_PATH", "/Results")

    mock_client, _ = _mock_docker_client()
    mock_proc = MagicMock()

    with patch("optimizer.docker.worker.docker") as mock_docker, \
         patch("optimizer.docker.worker.subprocess") as mock_subprocess:
        mock_docker.from_env.return_value = mock_client
        mock_subprocess.Popen.return_value = mock_proc
        worker = _make_worker(artifact_dir, results_root, harness_dir)
        worker.start()

    mock_client.containers.run.assert_called_once()
    _, kwargs = mock_client.containers.run.call_args
    volumes = kwargs["volumes"]

    assert "/fake/data" in volumes
    assert volumes["/fake/data"]["bind"] == "/Lean/Data"
    assert volumes["/fake/data"]["mode"] == "ro"

    assert str(artifact_dir.resolve()) in volumes
    assert volumes[str(artifact_dir.resolve())]["mode"] == "ro"

    assert str(results_root.resolve()) in volumes
    assert volumes[str(results_root.resolve())]["mode"] == "rw"

    assert str(harness_dir.resolve()) in volumes
    assert volumes[str(harness_dir.resolve())]["bind"] == "/Harness"
    assert volumes[str(harness_dir.resolve())]["mode"] == "ro"

    assert kwargs["entrypoint"] == ["sleep", "infinity"]


def test_run_backtest_uses_harness_protocol(tmp_path, monkeypatch):
    """run_backtest sends JSON to harness stdin and reads JSON response from stdout."""
    import json

    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    results_root = tmp_path / "res"
    results_root.mkdir()

    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")
    monkeypatch.setenv("RESULTS_CONTAINER_PATH", "/Results")
    monkeypatch.setenv("ARTIFACTS_CONTAINER_PATH", "/Artifacts")

    ev = Evaluation(id="abc123def456" * 2, parameters={"ticker": "SPY"})
    response_json = json.dumps({"id": ev.id, "status": "ok"}) + "\n"

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline.return_value = response_json

    with patch("optimizer.docker.worker.config_builder") as mock_cb, \
         patch("optimizer.docker.worker.select") as mock_select:
        mock_cb.build.return_value = results_root / ev.id / "config.json"
        mock_select.select.return_value = ([mock_proc.stdout], [], [])

        worker = _make_worker(artifact_dir, results_root)
        worker._container = MagicMock()
        worker._harness_proc = mock_proc
        result = worker.run_backtest(ev)

    mock_proc.stdin.write.assert_called_once()
    mock_proc.stdin.flush.assert_called_once()
    assert result.status == EvaluationStatus.SUCCESS


def test_run_backtest_handles_harness_error(tmp_path, monkeypatch):
    """run_backtest sets FAILED status when harness returns error."""
    import json

    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    results_root = tmp_path / "res"
    results_root.mkdir()

    monkeypatch.setenv("RESULTS_CONTAINER_PATH", "/Results")
    monkeypatch.setenv("ARTIFACTS_CONTAINER_PATH", "/Artifacts")

    ev = Evaluation(id="abc123def456" * 2, parameters={"ticker": "SPY"})
    response_json = json.dumps({"id": ev.id, "status": "error", "message": "algo crashed"}) + "\n"

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline.return_value = response_json

    with patch("optimizer.docker.worker.config_builder") as mock_cb, \
         patch("optimizer.docker.worker.select") as mock_select:
        mock_cb.build.return_value = results_root / ev.id / "config.json"
        mock_select.select.return_value = ([mock_proc.stdout], [], [])

        worker = _make_worker(artifact_dir, results_root)
        worker._container = MagicMock()
        worker._harness_proc = mock_proc
        result = worker.run_backtest(ev)

    assert result.status == EvaluationStatus.FAILED
    assert "algo crashed" in result.error_message


def test_run_backtest_timeout(tmp_path, monkeypatch):
    """run_backtest fails with timeout error when harness doesn't respond."""
    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    results_root = tmp_path / "res"
    results_root.mkdir()

    monkeypatch.setenv("RESULTS_CONTAINER_PATH", "/Results")
    monkeypatch.setenv("ARTIFACTS_CONTAINER_PATH", "/Artifacts")

    ev = Evaluation(id="abc123def456" * 2, parameters={"ticker": "SPY"})

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()

    with patch("optimizer.docker.worker.config_builder") as mock_cb, \
         patch("optimizer.docker.worker.select") as mock_select:
        mock_cb.build.return_value = results_root / ev.id / "config.json"
        mock_select.select.return_value = ([], [], [])  # nothing ready = timeout

        worker = _make_worker(artifact_dir, results_root)
        worker._container = MagicMock()
        worker._harness_proc = mock_proc
        result = worker.run_backtest(ev)

    assert result.status == EvaluationStatus.FAILED
    assert "timeout" in result.error_message.lower()


def test_stop_closes_harness_then_container(tmp_path):
    """stop() closes harness stdin then stops and removes container."""
    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    results_root = tmp_path / "res"
    results_root.mkdir()

    mock_client, mock_container = _mock_docker_client()
    mock_proc = MagicMock()

    with patch("optimizer.docker.worker.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_client
        worker = _make_worker(artifact_dir, results_root)
        worker._container = mock_container
        worker._harness_proc = mock_proc
        worker.stop()

    mock_proc.stdin.close.assert_called_once()
    mock_proc.wait.assert_called_once()
    mock_container.stop.assert_called_once_with(timeout=5)
    mock_container.remove.assert_called_once_with(force=True)
    assert worker._container is None
    assert worker._harness_proc is None


def test_is_alive_checks_both_container_and_harness(tmp_path):
    """is_alive() returns True only when both container and harness are running."""
    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    results_root = tmp_path / "res"
    results_root.mkdir()

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    worker = _make_worker(artifact_dir, results_root)
    worker._container = mock_container
    worker._harness_proc = mock_proc
    assert worker.is_alive() is True

    # Harness dead
    mock_proc.poll.return_value = 1
    assert worker.is_alive() is False

    # Container dead
    mock_proc.poll.return_value = None
    mock_container.status = "exited"
    assert worker.is_alive() is False


def test_context_manager_calls_stop(tmp_path, monkeypatch):
    """Context manager __exit__ calls stop()."""
    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    results_root = tmp_path / "res"
    results_root.mkdir()

    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")

    mock_client, mock_container = _mock_docker_client()
    mock_proc = MagicMock()

    with patch("optimizer.docker.worker.docker") as mock_docker, \
         patch("optimizer.docker.worker.subprocess") as mock_subprocess:
        mock_docker.from_env.return_value = mock_client
        mock_subprocess.Popen.return_value = mock_proc
        with _make_worker(artifact_dir, results_root) as worker:
            assert worker._container is not None

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
