from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Evaluation:
    id: str                                     # uuid4 hex
    parameters: dict                            # flat {key: value}
    status: EvaluationStatus = EvaluationStatus.PENDING
    result_path: Optional[Path] = None
    runtime_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Optional[dict] = None              # populated by collector
    worker_id: Optional[str] = None            # set by batch_runner; which container ran this
    fitness_score: Optional[float] = None      # set by pipeline stage
    extracted_metrics: Optional[dict] = None   # set by pipeline stage
    stage: Optional[str] = None                # e.g. "lhs", "bayesian", "local_grid", "genetic"
    stage_detail: Optional[str] = None         # sub-stage label, e.g. "gen_0", "gen_42"
    window_label: Optional[str] = None         # optional time-window identifier

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parameters": self.parameters,
            "status": self.status.value,
            "result_path": str(self.result_path) if self.result_path else None,
            "runtime_seconds": self.runtime_seconds,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "worker_id": self.worker_id,
            "fitness_score": self.fitness_score,
            "extracted_metrics": self.extracted_metrics,
            "stage": self.stage,
            "stage_detail": self.stage_detail,
            "window_label": self.window_label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evaluation":
        ev = cls(id=d["id"], parameters=d["parameters"])
        ev.status = EvaluationStatus(d["status"]) if d.get("status") else EvaluationStatus.PENDING
        ev.result_path = Path(d["result_path"]) if d.get("result_path") else None
        ev.runtime_seconds = d.get("runtime_seconds")
        ev.error_message = d.get("error_message")
        ev.metrics = d.get("metrics")
        ev.worker_id = d.get("worker_id")
        ev.fitness_score = d.get("fitness_score")
        ev.extracted_metrics = d.get("extracted_metrics")
        ev.stage = d.get("stage")
        ev.stage_detail = d.get("stage_detail")
        ev.window_label = d.get("window_label")
        return ev
