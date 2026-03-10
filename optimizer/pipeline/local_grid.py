from loguru import logger

from optimizer.fitness.base import FitnessFunction
from optimizer.pipeline.base import BatchRunner, OptimizationStage
from optimizer.pipeline.parameter_space import ParameterSpace
from optimizer.pipeline.scoring import score_evaluation


def _fingerprint(params: dict, keys: set | None = None) -> frozenset:
    """Create hashable fingerprint from parameter dict for O(1) dedup.

    If keys is provided, only those keys are included in the fingerprint.
    This allows matching across dicts that may have extra fields (e.g. base_params).
    """
    if keys is not None:
        return frozenset((k, v) for k, v in params.items() if k in keys)
    return frozenset(params.items())


class LocalGridStage(OptimizationStage):
    """Local grid search around top candidates from previous stages."""

    def __init__(self, top_n: int = 5, radius: int = 1, max_neighbors: int | None = None):
        self.top_n = top_n
        self.radius = radius
        self.max_neighbors = max_neighbors

    def run(
        self,
        space: ParameterSpace,
        fitness_fn: FitnessFunction,
        batch_runner: BatchRunner,
        previous_results: list | None = None,
    ) -> list:
        if not previous_results:
            logger.warning("[LocalGridStage] No previous results — nothing to refine")
            return []

        # Select top candidates (ignore -inf)
        valid = [
            ev
            for ev in previous_results
            if ev.fitness_score is not None and ev.fitness_score != float("-inf")
        ]
        if not valid:
            logger.warning("[LocalGridStage] No valid candidates in previous results")
            return []

        top = sorted(valid, key=lambda e: e.fitness_score, reverse=True)[
            : self.top_n
        ]
        previous_best = top[0].fitness_score
        tunable_names = {p.name for p in space.parameters}

        # Build dedup set from all previous results (tunable params only)
        evaluated = {_fingerprint(ev.parameters, tunable_names) for ev in previous_results}

        # Generate neighbors for each top candidate, dedup as we go
        total_generated = 0
        novel_params: list[dict] = []
        for seed_ev in top:
            neighbors = space.neighbors(seed_ev.parameters, radius=self.radius)
            total_generated += len(neighbors)
            for n in neighbors:
                fp = _fingerprint(n, tunable_names)
                if fp not in evaluated:
                    evaluated.add(fp)  # prevent dupes across seeds
                    novel_params.append(n)

        skipped = total_generated - len(novel_params)

        # Cap if max_neighbors is set
        if self.max_neighbors is not None and len(novel_params) > self.max_neighbors:
            import random
            novel_params = random.sample(novel_params, self.max_neighbors)

        logger.info(
            f"[LocalGridStage] {total_generated} neighbors generated from"
            f" {len(top)} seeds, {skipped} skipped (already evaluated),"
            f" {len(novel_params)} to run"
            + (f" (capped from {total_generated - skipped})" if self.max_neighbors and total_generated - skipped > len(novel_params) else "")
        )

        if not novel_params:
            logger.info("[LocalGridStage] All neighbors already evaluated")
            return []

        # Run batch
        evaluations = batch_runner(novel_params)

        # Score each
        for ev in evaluations:
            ev.stage = "local_grid"
            score_evaluation(ev, fitness_fn)

        # Log improvement
        new_valid = [ev for ev in evaluations if ev.fitness_score != float("-inf")]
        if new_valid:
            new_best_ev = max(new_valid, key=lambda e: e.fitness_score)
            if new_best_ev.fitness_score > previous_best:
                logger.info(
                    f"[LocalGridStage] New best! {new_best_ev.fitness_score:.4f}"
                    f" (was {previous_best:.4f})  params={new_best_ev.parameters}"
                )
            else:
                logger.info(
                    f"[LocalGridStage] Best local: {new_best_ev.fitness_score:.4f}"
                    f"  (previous best: {previous_best:.4f})"
                )

        return evaluations
