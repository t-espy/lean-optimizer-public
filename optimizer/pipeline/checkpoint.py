"""Incremental checkpoint persistence for optimization pipelines.

Saves a JSON checkpoint after each stage completes, enabling:
- Recovery from ctrl-c (all completed stages preserved)
- Resume with --resume flag (skip completed stages, reuse their results)
"""

import json
from pathlib import Path

from loguru import logger

from optimizer.runner.evaluation import Evaluation


def save_checkpoint(
    path: Path,
    completed_stages: list[str],
    stage_results: dict[str, list],
) -> None:
    """Write checkpoint to disk. Overwrites previous checkpoint at same path.

    Deliberately writes to a temp file then renames — atomic on POSIX,
    so a ctrl-c during write won't corrupt the checkpoint.
    """
    data = {
        "completed_stages": completed_stages,
        "stage_results": {
            name: [ev.to_dict() for ev in evs]
            for name, evs in stage_results.items()
        },
    }

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)

    total = sum(len(evs) for evs in stage_results.values())
    logger.info(
        f"[Checkpoint] Saved {total} evaluations"
        f" ({len(completed_stages)} stages) → {path}"
    )


def load_checkpoint(path: Path) -> dict | None:
    """Load checkpoint from disk. Returns None if file doesn't exist.

    Returns dict with:
        completed_stages: list[str]
        stage_results: dict[str, list[Evaluation]]
    """
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[Checkpoint] Could not load {path}: {e}")
        return None

    stage_results = {}
    for name, ev_dicts in raw.get("stage_results", {}).items():
        stage_results[name] = [Evaluation.from_dict(d) for d in ev_dicts]

    logger.info(
        f"[Checkpoint] Loaded {sum(len(v) for v in stage_results.values())} evaluations"
        f" from {len(raw.get('completed_stages', []))} completed stages"
    )

    return {
        "completed_stages": raw.get("completed_stages", []),
        "stage_results": stage_results,
    }
