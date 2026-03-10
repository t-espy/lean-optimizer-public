from loguru import logger

from optimizer.fitness.base import FitnessFunction
from optimizer.pipeline.base import BatchRunner, OptimizationStage
from optimizer.pipeline.parameter_space import ParameterSpace
from optimizer.pipeline.scoring import score_evaluation


class LHSStage(OptimizationStage):
    """Latin Hypercube Sampling stage — space-filling initial exploration."""

    def __init__(self, n_samples: int = 150, seed: int | None = None):
        self.n_samples = n_samples
        self.seed = seed

    def run(
        self,
        space: ParameterSpace,
        fitness_fn: FitnessFunction,
        batch_runner: BatchRunner,
        previous_results: list | None = None,
    ) -> list:
        # 1. Sample n_samples points via LHS
        param_sets = space.sample_lhs(self.n_samples, seed=self.seed)
        logger.info(f"[LHSStage] Sampled {len(param_sets)} points via LHS")

        # 2. Submit all as a batch
        evaluations = batch_runner(param_sets)

        # 3-4. Score each evaluation, attach metadata
        for ev in evaluations:
            ev.stage = "lhs"
            score_evaluation(ev, fitness_fn)

        # 5. Log top 5 results by fitness score
        valid = [ev for ev in evaluations if ev.fitness_score != float("-inf")]
        top5 = sorted(valid, key=lambda e: e.fitness_score, reverse=True)[:5]
        logger.info(f"[LHSStage] {len(valid)}/{len(evaluations)} valid results")
        for i, ev in enumerate(top5, 1):
            logger.info(
                f"  #{i} [{ev.id[:8]}] score={ev.fitness_score:.4f}"
                f"  params={ev.parameters}"
            )

        # 6. Return all evaluations
        return evaluations
