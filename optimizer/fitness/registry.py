from typing import Dict, Type

from optimizer.fitness.base import FitnessFunction
from optimizer.fitness.calmar import CalmarFitness
from optimizer.fitness.trailing_stop import TrailingStopFitness

FITNESS_REGISTRY: Dict[str, Type[FitnessFunction]] = {
    "calmar": CalmarFitness,
    "trailing_stop": TrailingStopFitness,
}


def get_fitness(name: str = "calmar", **kwargs) -> FitnessFunction:
    """Instantiate a fitness function by name.

    Args:
        name: Key in FITNESS_REGISTRY (default "calmar").
        **kwargs: Forwarded to the fitness function constructor.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in FITNESS_REGISTRY:
        raise KeyError(f"Unknown fitness function {name!r}. Available: {list(FITNESS_REGISTRY)}")
    return FITNESS_REGISTRY[name](**kwargs)
