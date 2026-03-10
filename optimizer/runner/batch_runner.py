from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

from loguru import logger

from optimizer.docker.pool import WorkerPool
from optimizer.results import collector
from optimizer.runner.evaluation import Evaluation, EvaluationStatus


def run_batch(
    parameter_sets: list,
    pool: WorkerPool,
    results_root: Path,
    artifacts_container_path: str,
) -> list:
    """Run parameter_sets in parallel using the worker pool.

    Args:
        parameter_sets: List of parameter dicts.
        pool: Started WorkerPool instance (lifecycle managed by caller).
        results_root: Host path for results directory.
        artifacts_container_path: Unused; kept for signature compatibility.

    Returns:
        List of Evaluation objects (one per param set, in submission order).
    """

    def run_one(params: dict) -> Evaluation:
        ev = Evaluation(id=uuid4().hex, parameters=params)
        worker = pool.acquire()
        ev.worker_id = worker.worker_id
        dead = False
        try:
            ev = worker.run_backtest(ev)
            result = collector.verify_result(ev, results_root)
            if result is not None:
                ev.metrics = result.to_dict()
        except Exception as exc:
            ev.status = EvaluationStatus.FAILED
            ev.error_message = str(exc)
            logger.error(f"[{ev.id[:8]}] exception during backtest: {exc}")
            dead = not worker.is_alive()
        finally:
            pool.release(worker, dead=dead)

        runtime_str = f"{ev.runtime_seconds:.1f}s" if ev.runtime_seconds is not None else "n/a"
        logger.info(
            f"[{ev.id[:8]}] {ev.status.value}  {runtime_str}"
            f"  worker={worker.worker_id[-8:]}"
        )
        return ev

    with ThreadPoolExecutor(max_workers=pool.size) as executor:
        future_to_idx = {
            executor.submit(run_one, params): i
            for i, params in enumerate(parameter_sets)
        }
        evaluations = [None] * len(parameter_sets)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            evaluations[idx] = future.result()

    return evaluations
