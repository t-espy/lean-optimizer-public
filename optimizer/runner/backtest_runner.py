import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from loguru import logger

from optimizer.runner.evaluation import Evaluation, EvaluationStatus


def _backtest_output_dir(params: dict, backtests_root: Path) -> Path:
    symbol = params.get("ticker", "UNKNOWN")
    start = params.get("startDate", "unknown")
    return backtests_root / symbol / f"{start}"


def _collect_backtest_output(
    evaluation: Evaluation,
    results_root: Path,
    backtests_root: Path,
) -> Path | None:
    result_dir = results_root / evaluation.id
    result_path = result_dir / f"{evaluation.id}.json"

    if not result_path.exists():
        logger.warning(f"[{evaluation.id[:8]}] Result file not found: {result_path}")
        return None

    if result_path.stat().st_size == 0:
        logger.warning(f"[{evaluation.id[:8]}] Result file is empty: {result_path}")
        return None

    output_dir = _backtest_output_dir(evaluation.parameters, backtests_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    # Detect strategy name from the generated config
    strategy = "backtest"
    config_path = result_dir / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            dll_name = cfg.get("algorithm-location", "")
            strategy = Path(dll_name).stem or strategy
        except Exception:
            pass
    output_path = output_dir / f"{strategy}_{timestamp}.json"

    try:
        shutil.copy2(result_path, output_path)
        logger.info(f"[{evaluation.id[:8]}] Backtest saved: {output_path}")
    except Exception as exc:
        logger.error(f"[{evaluation.id[:8]}] Failed to copy result: {exc}")
        return None

    try:
        shutil.rmtree(result_dir)
        logger.debug(f"[{evaluation.id[:8]}] Cleaned scratch: {result_dir}")
    except Exception as exc:
        logger.warning(f"[{evaluation.id[:8]}] Could not clean scratch: {exc}")

    return output_path


def run_backtests(
    parameter_sets: list,
    pool,
    results_root: Path,
    backtests_root: Path,
) -> list:
    """Run parameter_sets in parallel and persist full LEAN result JSONs.

    Args:
        parameter_sets: List of parameter dicts.
        pool: Started WorkerPool instance (lifecycle managed by caller).
        results_root: Host path for scratch results directory.
        backtests_root: Host path for final backtests output directory.

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
            if ev.status == EvaluationStatus.SUCCESS:
                ev.result_path = _collect_backtest_output(ev, results_root, backtests_root)
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
        futures = [executor.submit(run_one, params) for params in parameter_sets]
        evaluations = [f.result() for f in futures]

    return evaluations
