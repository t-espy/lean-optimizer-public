from loguru import logger

from optimizer.fitness.base import FitnessFunction
from optimizer.results.extractor import ExtractedMetrics
from optimizer.runner.evaluation import Evaluation


def score_evaluation(ev: Evaluation, fitness_fn: FitnessFunction) -> None:
    """Score an evaluation and attach fitness_score + extracted_metrics in place."""
    if ev.metrics is not None:
        try:
            # Separate trades (list) from scalar metrics for ExtractedMetrics constructor
            m = dict(ev.metrics)
            trades = m.pop("trades", [])
            metrics = ExtractedMetrics(raw={}, trades=trades, **m)
            ev.fitness_score = fitness_fn.score(metrics)
            ev.extracted_metrics = ev.metrics
        except Exception as e:
            logger.warning(f"[{ev.id[:8]}] Could not score: {e}")
            ev.fitness_score = float("-inf")
            ev.extracted_metrics = ev.metrics
    else:
        ev.fitness_score = float("-inf")
