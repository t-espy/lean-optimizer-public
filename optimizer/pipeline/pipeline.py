from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from optimizer.fitness.base import FitnessFunction
from optimizer.pipeline.base import BatchRunner, OptimizationStage
from optimizer.pipeline.checkpoint import save_checkpoint
from optimizer.pipeline.parameter_space import ParameterSpace


@dataclass
class PipelineResult:
    """Aggregated results from a full optimization pipeline run."""

    all_evaluations: list = field(default_factory=list)
    stage_results: dict = field(default_factory=dict)
    best_evaluation: object = None
    interrupted: bool = False

    @property
    def best_score(self) -> float:
        if self.best_evaluation is not None:
            return self.best_evaluation.fitness_score
        return float("-inf")


class OptimizationPipeline:
    """Runs optimization stages sequentially, accumulating results.

    Supports:
    - Checkpoint persistence after each stage (survives ctrl-c between stages)
    - KeyboardInterrupt handling (saves partial results mid-stage)
    - Resume from checkpoint (skip completed stages, reuse their results)
    """

    def __init__(
        self,
        stages: list[OptimizationStage],
        checkpoint_path: Path | None = None,
    ):
        self.stages = stages
        self.checkpoint_path = checkpoint_path

    def run(
        self,
        space: ParameterSpace,
        fitness_fn: FitnessFunction,
        batch_runner: BatchRunner,
        resume_from: dict | None = None,
        eval_logger=None,
    ) -> PipelineResult:
        result = PipelineResult()
        accumulated: list = []
        completed_stages: list[str] = []

        # Resume: reload completed stages from checkpoint
        if resume_from is not None:
            completed_stages = list(resume_from.get("completed_stages", []))
            for name, evs in resume_from.get("stage_results", {}).items():
                result.stage_results[name] = evs
                result.all_evaluations.extend(evs)
                accumulated.extend(evs)
                for ev in evs:
                    self._track_best(result, ev)
            logger.info(
                f"[Pipeline] Resumed from checkpoint:"
                f" {len(completed_stages)} stages, {len(accumulated)} evaluations"
            )

        # Run each stage sequentially
        for stage_idx, stage in enumerate(self.stages):
            stage_name = type(stage).__name__

            # Skip if already completed in checkpoint (positional match)
            if stage_idx < len(completed_stages):
                saved_name = completed_stages[stage_idx]
                if saved_name == stage_name:
                    logger.info(f"[Pipeline] Skipping {stage_name} [{stage_idx}] (in checkpoint)")
                    continue
                else:
                    logger.warning(
                        f"[Pipeline] Checkpoint mismatch at index {stage_idx}:"
                        f" expected {saved_name}, got {stage_name}. Running from here."
                    )
                    # Truncate completed_stages — can't trust anything after mismatch
                    completed_stages = completed_stages[:stage_idx]

            logger.info(f"\n{'='*60}")
            logger.info(f"  STAGE: {stage_name}")
            logger.info(f"{'='*60}")

            try:
                stage_evals = stage.run(
                    space=space,
                    fitness_fn=fitness_fn,
                    batch_runner=batch_runner,
                    previous_results=list(accumulated) if accumulated else None,
                )
            except KeyboardInterrupt:
                logger.warning(
                    f"\n[Pipeline] Interrupted during {stage_name}!"
                    f" Saving {len(result.all_evaluations)} evaluations from prior stages."
                )
                result.interrupted = True
                self._save(result, completed_stages)
                return result

            accumulated.extend(stage_evals)
            result.stage_results[stage_name] = stage_evals
            result.all_evaluations.extend(stage_evals)
            completed_stages.append(stage_name)

            if eval_logger is not None:
                eval_logger.log_evals(stage_evals)

            for ev in stage_evals:
                self._track_best(result, ev)

            valid_count = sum(
                1
                for ev in stage_evals
                if ev.fitness_score is not None
                and ev.fitness_score != float("-inf")
            )
            logger.info(
                f"[Pipeline] {stage_name} complete:"
                f" {len(stage_evals)} evaluations, {valid_count} valid"
            )
            if result.best_evaluation is not None:
                logger.info(
                    f"[Pipeline] Best so far: {result.best_score:.4f}"
                    f"  params={result.best_evaluation.parameters}"
                )

            # Checkpoint after each stage
            self._save(result, completed_stages)

        total = len(result.all_evaluations)
        logger.info(f"\n[Pipeline] All stages complete. {total} total evaluations.")
        return result

    def _track_best(self, result: PipelineResult, ev) -> None:
        if (
            ev.fitness_score is not None
            and ev.fitness_score != float("-inf")
            and (
                result.best_evaluation is None
                or ev.fitness_score > result.best_evaluation.fitness_score
            )
        ):
            result.best_evaluation = ev

    def _save(
        self,
        result: PipelineResult,
        completed_stages: list[str],
    ) -> None:
        if self.checkpoint_path is None:
            return
        save_checkpoint(
            path=self.checkpoint_path,
            completed_stages=completed_stages,
            stage_results=result.stage_results,
        )
