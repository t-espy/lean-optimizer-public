import math

import pytest

from optimizer.fitness.calmar import CalmarFitness
from optimizer.fitness.registry import FITNESS_REGISTRY, get_fitness
from optimizer.results.extractor import ExtractedMetrics


def _metrics(**overrides) -> ExtractedMetrics:
    """Return an ExtractedMetrics instance with sensible defaults."""
    defaults = dict(
        net_pnl=23.4,
        max_drawdown=5.0,
        trade_count=42,
        profit_factor=1.79,
        win_rate=0.55,
        sharpe_ratio=1.45,
        avg_trade=0.557,
    )
    defaults.update(overrides)
    return ExtractedMetrics(**defaults)


# ---------------------------------------------------------------------------
# CalmarFitness.is_valid
# ---------------------------------------------------------------------------

def test_is_valid_passes_on_good_metrics():
    f = CalmarFitness()
    assert f.is_valid(_metrics()) is True


def test_is_valid_fails_below_min_trades():
    f = CalmarFitness(min_trades=30)
    assert f.is_valid(_metrics(trade_count=29)) is False


def test_is_valid_fails_zero_drawdown():
    f = CalmarFitness()
    assert f.is_valid(_metrics(max_drawdown=0.0)) is False


def test_is_valid_fails_negative_drawdown():
    f = CalmarFitness()
    assert f.is_valid(_metrics(max_drawdown=-1.0)) is False


def test_is_valid_fails_drawdown_limit_exceeded():
    f = CalmarFitness(max_drawdown_limit=10.0)
    assert f.is_valid(_metrics(max_drawdown=10.1)) is False


def test_is_valid_passes_drawdown_at_limit():
    f = CalmarFitness(max_drawdown_limit=10.0)
    assert f.is_valid(_metrics(max_drawdown=10.0)) is True


def test_is_valid_fails_low_profit_factor():
    f = CalmarFitness(min_profit_factor=1.0)
    assert f.is_valid(_metrics(profit_factor=0.99)) is False


def test_is_valid_passes_at_min_profit_factor():
    f = CalmarFitness(min_profit_factor=1.0)
    assert f.is_valid(_metrics(profit_factor=1.0)) is True


# ---------------------------------------------------------------------------
# CalmarFitness.compute
# ---------------------------------------------------------------------------

def test_compute_formula():
    f = CalmarFitness(min_trades=30)
    m = _metrics(net_pnl=23.4, max_drawdown=5.0, trade_count=42)
    expected = (23.4 / 5.0) * math.log10(42 / 30)
    assert f.compute(m) == pytest.approx(expected, rel=1e-6)


def test_compute_higher_pnl_gives_higher_score():
    f = CalmarFitness()
    low = f.compute(_metrics(net_pnl=10.0))
    high = f.compute(_metrics(net_pnl=50.0))
    assert high > low


def test_compute_higher_drawdown_gives_lower_score():
    f = CalmarFitness()
    low_dd = f.compute(_metrics(max_drawdown=5.0))
    high_dd = f.compute(_metrics(max_drawdown=20.0))
    assert low_dd > high_dd


def test_compute_more_trades_gives_higher_score():
    f = CalmarFitness(min_trades=30)
    few = f.compute(_metrics(trade_count=31))
    many = f.compute(_metrics(trade_count=300))
    assert many > few


# ---------------------------------------------------------------------------
# CalmarFitness.score (via base class)
# ---------------------------------------------------------------------------

def test_score_returns_neginf_when_invalid():
    f = CalmarFitness(min_trades=30)
    assert f.score(_metrics(trade_count=5)) == float("-inf")


def test_score_returns_compute_when_valid():
    f = CalmarFitness()
    m = _metrics()
    assert f.score(m) == pytest.approx(f.compute(m))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_registry_contains_calmar():
    assert "calmar" in FITNESS_REGISTRY


def test_get_fitness_returns_calmar_instance():
    f = get_fitness("calmar")
    assert isinstance(f, CalmarFitness)


def test_get_fitness_forwards_kwargs():
    f = get_fitness("calmar", min_trades=50, max_drawdown_limit=15.0)
    assert f.min_trades == 50
    assert f.max_drawdown_limit == 15.0


def test_get_fitness_unknown_name_raises():
    with pytest.raises(KeyError):
        get_fitness("nonexistent")
