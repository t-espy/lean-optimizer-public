import queue
import threading
from pathlib import Path

from loguru import logger

from optimizer.docker.worker import Worker


class WorkerPool:
    """Manages N warm LEAN engine containers for parallel backtest execution.

    Usage (Phase 4+: pool is long-lived across multiple batches):
        with WorkerPool(n=5, artifact_dir=..., results_root=...) as pool:
            run_batch(param_sets_1, pool, ...)
            run_batch(param_sets_2, pool, ...)

    Workers are started in parallel at startup. Callers acquire an idle worker
    (blocking if all are busy), run their backtest, then release it back.
    Dead containers are replaced asynchronously without stalling other workers.
    """

    def __init__(
        self,
        n: int,
        artifact_dir: Path,
        results_root: Path,
        harness_dir: Path,
        container_prefix: str = "lean_optimizer",
    ):
        self._n = n
        self._artifact_dir = artifact_dir.resolve()
        self._results_root = results_root.resolve()
        self._harness_dir = harness_dir.resolve()
        self._container_prefix = container_prefix
        self._queue: queue.Queue[Worker] = queue.Queue()
        self._lock = threading.Lock()
        self._all_workers: list[Worker] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start N workers in parallel; block until all are ready."""
        logger.info(f"[pool] Starting {self._n} workers in parallel...")
        threads = [
            threading.Thread(target=self._start_one_worker, daemon=True)
            for _ in range(self._n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ready = self._queue.qsize()
        logger.info(f"[pool] {ready}/{self._n} workers ready")
        if ready == 0:
            raise RuntimeError("Worker pool failed to start any containers")

    def stop(self) -> None:
        """Stop all registered workers in parallel."""
        with self._lock:
            workers = list(self._all_workers)
        if not workers:
            return
        logger.info(f"[pool] Stopping {len(workers)} workers...")
        threads = [threading.Thread(target=w.stop, daemon=True) for w in workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with self._lock:
            self._all_workers.clear()
        logger.info("[pool] All workers stopped")

    # ------------------------------------------------------------------
    # Worker acquisition / release
    # ------------------------------------------------------------------

    def acquire(self, timeout: float = 300) -> Worker:
        """Block until an idle worker is available and return it."""
        try:
            worker = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"[pool] No worker available after {timeout}s — "
                f"all {self._n} workers may be dead"
            )
        logger.debug(f"[pool] Acquired {worker.worker_id}")
        return worker

    def release(self, worker: Worker, dead: bool = False) -> None:
        """Return worker to the idle pool, or replace it if dead."""
        if dead:
            logger.warning(f"[pool] Worker {worker.worker_id} is dead — spawning replacement")
            threading.Thread(
                target=self._replace_worker, args=(worker,), daemon=True
            ).start()
        else:
            logger.debug(f"[pool] Released {worker.worker_id}")
            self._queue.put(worker)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_one_worker(self) -> None:
        worker = Worker(
            self._artifact_dir, self._results_root, self._harness_dir,
            container_prefix=self._container_prefix,
        )
        try:
            worker.start()
        except Exception as e:
            logger.error(f"[pool] Failed to start worker: {e}")
            return
        with self._lock:
            self._all_workers.append(worker)
        self._queue.put(worker)

    def _replace_worker(self, dead_worker: Worker) -> None:
        with self._lock:
            if dead_worker in self._all_workers:
                self._all_workers.remove(dead_worker)
        try:
            dead_worker.stop()
        except Exception:
            pass
        self._start_one_worker()
        logger.info("[pool] Replacement worker ready")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WorkerPool":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
