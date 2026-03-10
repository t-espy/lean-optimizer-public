import json
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from optimizer.results import extractor as _extractor
from optimizer.results.extractor import ExtractedMetrics
from optimizer.runner.evaluation import Evaluation


def verify_result(evaluation: Evaluation, results_root: Path) -> Optional[ExtractedMetrics]:
    """Verify result JSON exists and is valid; return ExtractedMetrics or None.

    On success: parses JSON, extracts metrics, deletes the result directory (scratch cleanup).
    On any failure: returns None without deleting — leaves directory for inspection.
    """
    result_dir = results_root / evaluation.id
    result_path = result_dir / f"{evaluation.id}.json"

    if not result_path.exists():
        logger.warning(f"[{evaluation.id[:8]}] Result file not found: {result_path}")
        return None

    if result_path.stat().st_size == 0:
        logger.warning(f"[{evaluation.id[:8]}] Result file is empty: {result_path}")
        return None

    try:
        data = json.loads(result_path.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"[{evaluation.id[:8]}] Failed to parse result JSON: {e}")
        return None

    if "runtimeStatistics" not in data or "statistics" not in data:
        logger.warning(
            f"[{evaluation.id[:8]}] Result missing required keys "
            f"(runtimeStatistics/statistics): {result_path}"
        )
        return None

    metrics = _extractor.extract(data)
    if metrics is None:
        logger.warning(f"[{evaluation.id[:8]}] Failed to extract metrics from result")
        return None

    # All checks passed — clean up scratch space
    try:
        shutil.rmtree(result_dir)
        logger.debug(f"[{evaluation.id[:8]}] Result dir cleaned: {result_dir}")
    except Exception as e:
        logger.warning(f"[{evaluation.id[:8]}] Could not clean result dir: {e}")

    logger.info(f"[{evaluation.id[:8]}] Result verified")
    return metrics
