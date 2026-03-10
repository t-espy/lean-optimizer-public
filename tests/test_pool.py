import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from optimizer.docker.pool import WorkerPool


def _make_pool(tmp_path: Path, n: int = 0) -> WorkerPool:
    return WorkerPool(
        n=n,
        artifact_dir=tmp_path / "arts",
        results_root=tmp_path / "res",
        harness_dir=tmp_path / "harness",
    )


def _mock_worker(name: str = "lean_optimizer_test") -> MagicMock:
    w = MagicMock()
    w.worker_id = name
    return w


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_pool_starts_n_workers(tmp_path, monkeypatch):
    """start() creates exactly N containers."""
    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")

    mock_worker = _mock_worker()

    with patch("optimizer.docker.pool.Worker") as MockWorker:
        MockWorker.return_value = mock_worker
        pool = _make_pool(tmp_path, n=3)
        pool.start()

    assert MockWorker.call_count == 3
    assert mock_worker.start.call_count == 3
    assert pool._queue.qsize() == 3
    assert len(pool._all_workers) == 3


def test_acquire_blocks_when_pool_empty(tmp_path):
    """acquire() blocks when no workers are available."""
    pool = _make_pool(tmp_path, n=0)
    # Don't call start() — queue stays empty

    acquired = []

    def try_acquire():
        acquired.append(pool.acquire())

    t = threading.Thread(target=try_acquire, daemon=True)
    t.start()
    t.join(timeout=0.2)

    assert t.is_alive(), "acquire() should block on empty pool"

    # Unblock it by inserting a worker directly
    mock_worker = _mock_worker()
    pool._queue.put(mock_worker)
    t.join(timeout=1.0)

    assert not t.is_alive()
    assert acquired[0] is mock_worker


def test_release_returns_worker_to_idle(tmp_path):
    """release(worker) puts the worker back in the idle queue."""
    pool = _make_pool(tmp_path, n=0)
    mock_worker = _mock_worker()
    pool._queue.put(mock_worker)

    acquired = pool.acquire()
    assert pool._queue.qsize() == 0

    pool.release(acquired)
    assert pool._queue.qsize() == 1


def test_dead_worker_is_replaced(tmp_path, monkeypatch):
    """release(worker, dead=True) stops the dead worker and starts a replacement."""
    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")

    dead_worker = _mock_worker("lean_optimizer_dead")
    replacement = _mock_worker("lean_optimizer_replacement")

    with patch("optimizer.docker.pool.Worker") as MockWorker:
        MockWorker.return_value = replacement

        pool = _make_pool(tmp_path, n=0)
        with pool._lock:
            pool._all_workers.append(dead_worker)

        pool.release(dead_worker, dead=True)

        # Replacement is async — wait for it to enter the queue
        new_worker = pool._queue.get(timeout=3.0)

    dead_worker.stop.assert_called_once()
    replacement.start.assert_called_once()
    assert new_worker is replacement
    assert dead_worker not in pool._all_workers


def test_stop_shuts_down_all_workers(tmp_path):
    """stop() calls stop() on every registered worker."""
    pool = _make_pool(tmp_path, n=0)

    workers = [_mock_worker(f"lean_optimizer_{i}") for i in range(3)]
    with pool._lock:
        pool._all_workers.extend(workers)
    for w in workers:
        pool._queue.put(w)

    pool.stop()

    for w in workers:
        w.stop.assert_called_once()
    assert pool._all_workers == []


def test_start_raises_if_all_workers_fail(tmp_path, monkeypatch):
    """start() raises RuntimeError if no workers start successfully."""
    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")

    broken_worker = _mock_worker()
    broken_worker.start.side_effect = RuntimeError("Docker not available")

    with patch("optimizer.docker.pool.Worker") as MockWorker:
        MockWorker.return_value = broken_worker
        pool = _make_pool(tmp_path, n=2)

        with pytest.raises(RuntimeError, match="failed to start any"):
            pool.start()


def test_container_prefix_passed_to_workers(tmp_path, monkeypatch):
    """container_prefix is forwarded to Worker constructor."""
    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")

    mock_worker = _mock_worker()

    with patch("optimizer.docker.pool.Worker") as MockWorker:
        MockWorker.return_value = mock_worker
        pool = WorkerPool(
            n=1,
            artifact_dir=tmp_path / "arts",
            results_root=tmp_path / "res",
            harness_dir=tmp_path / "harness",
            container_prefix="lean_optimizer_AAPL",
        )
        pool.start()

    _, kwargs = MockWorker.call_args
    assert kwargs["container_prefix"] == "lean_optimizer_AAPL"


def test_context_manager_starts_and_stops(tmp_path, monkeypatch):
    """__enter__ calls start(), __exit__ calls stop()."""
    monkeypatch.setenv("LEAN_DATA_HOST_PATH", "/fake/data")

    mock_worker = _mock_worker()

    with patch("optimizer.docker.pool.Worker") as MockWorker:
        MockWorker.return_value = mock_worker
        pool = _make_pool(tmp_path, n=2)

        with pool:
            assert pool._queue.qsize() == 2

        # After __exit__, all workers should be stopped
        assert mock_worker.stop.call_count == 2
