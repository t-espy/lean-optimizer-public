from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from optimizer.fitness.base import FitnessFunction
from optimizer.pipeline.parameter_space import ParameterSpace


@runtime_checkable
class BatchRunner(Protocol):
    """Callable that takes a list of parameter dicts and returns a list of Evaluations."""

    def __call__(self, parameter_sets: list) -> list: ...


class OptimizationStage(ABC):
    """Abstract base for optimization pipeline stages."""

    @abstractmethod
    def run(
        self,
        space: ParameterSpace,
        fitness_fn: FitnessFunction,
        batch_runner: BatchRunner,
        previous_results: list | None = None,
    ) -> list:
        """Runs this optimization stage.

        Args:
            space: The parameter space to search.
            fitness_fn: Fitness function for scoring evaluations.
            batch_runner: Callable that runs backtests for a list of parameter dicts.
            previous_results: Accumulated Evaluations from all prior stages. May be None.

        Returns:
            List of Evaluation objects for runs executed in this stage.
        """
        ...
