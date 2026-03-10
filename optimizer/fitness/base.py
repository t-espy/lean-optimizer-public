from abc import ABC, abstractmethod

from optimizer.results.extractor import ExtractedMetrics


class FitnessFunction(ABC):
    """Abstract base for fitness functions."""

    def score(self, metrics: ExtractedMetrics) -> float:
        """Return fitness score; -inf if metrics fail validity check."""
        if not self.is_valid(metrics):
            return float("-inf")
        return self.compute(metrics)

    @abstractmethod
    def compute(self, metrics: ExtractedMetrics) -> float:
        """Compute raw fitness value (called only when is_valid returns True)."""
        ...

    @abstractmethod
    def is_valid(self, metrics: ExtractedMetrics) -> bool:
        """Return True if metrics meet minimum requirements for this function."""
        ...
