"""Per-evaluation JSONL logger for post-hoc fitness function analysis."""

import json
from pathlib import Path

from optimizer.runner.evaluation import EvaluationStatus


class EvalLogger:
    """Appends one JSON line per evaluation to a JSONL file.

    The file is opened/closed per batch write — no persistent file handle.
    Directory creation is lazy (first write).
    """

    def __init__(self, base_dir: Path, symbol: str, tunable_params: list[str]):
        self._path = base_dir / symbol / "evaluations.jsonl"
        self._tunable_params = tunable_params
        self._dir_created = False

    def log_evals(
        self, evals: list, oos_labels: set[str] | None = None
    ) -> None:
        if not evals:
            return

        if not self._dir_created:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._dir_created = True

        with open(self._path, "a") as f:
            for ev in evals:
                metrics = ev.metrics or {}
                record = {
                    "id": ev.id,
                    "stage": ev.stage,
                    "stage_detail": ev.stage_detail,
                    "fitness": ev.fitness_score,
                    "net_pnl": metrics.get("net_pnl"),
                    "trade_count": metrics.get("trade_count"),
                    "win_rate": metrics.get("win_rate"),
                    "profit_factor": metrics.get("profit_factor"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "params": {
                        k: v
                        for k, v in ev.parameters.items()
                        if k in self._tunable_params
                    },
                    "status": (
                        "success"
                        if ev.status == EvaluationStatus.SUCCESS
                        else "failed"
                    ),
                }

                if ev.window_label is not None:
                    record["window"] = ev.window_label
                    record["is_oos"] = (
                        oos_labels is not None
                        and ev.window_label in oos_labels
                    )

                f.write(json.dumps(record) + "\n")
            f.flush()

    def close(self) -> None:
        """No-op — file is opened/closed per batch."""
