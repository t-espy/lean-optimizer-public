import math
from dataclasses import dataclass
from typing import Optional

from optimizer.fitness.base import FitnessFunction
from optimizer.results.extractor import ExtractedMetrics


@dataclass
class CalmarFitness(FitnessFunction):
    """Calmar-inspired fitness: (net_pnl / max_drawdown) × log10(trade_count / min_trades).

    Rewards high return-to-drawdown ratio while penalising under-traded strategies.
    score() returns -inf whenever is_valid() returns False.
    """

    min_trades: int = 30
    max_drawdown_limit: Optional[float] = None  # None = no cap
    min_profit_factor: float = 1.0

    def is_valid(self, metrics: ExtractedMetrics) -> bool:
        if metrics.trade_count < self.min_trades:
            return False
        if metrics.max_drawdown <= 0:
            return False
        if (
            self.max_drawdown_limit is not None
            and metrics.max_drawdown > self.max_drawdown_limit
        ):
            return False
        if metrics.profit_factor < self.min_profit_factor:
            return False
        return True

    def compute(self, metrics: ExtractedMetrics) -> float:
        calmar = metrics.net_pnl / abs(metrics.max_drawdown)
        return calmar * math.log10(metrics.trade_count / self.min_trades)
