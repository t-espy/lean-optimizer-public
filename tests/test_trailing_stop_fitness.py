import math

import pytest

from optimizer.fitness.trailing_stop import TrailingStopFitness, _base_score, _bucket_by_quarter
from optimizer.fitness.registry import FITNESS_REGISTRY, get_fitness
from optimizer.results.extractor import ExtractedMetrics


def _make_trade(exit_time: str, profit: float, mfe: float, exit_reason: str = "MaxHoldBars") -> dict:
    return {"exitTime": exit_time, "profit": profit, "mfe": mfe, "exit_reason": exit_reason}


def _make_trades(n: int = 100, ts_pct: float = 0.5) -> list[dict]:
    """Generate n trades with configurable trailing stop percentage."""
    trades = []
    n_ts = int(n * ts_pct)
    n_mh = n - n_ts
    for _ in range(n_ts):
        trades.append(_make_trade("2025-03-01T10:00:00Z", profit=5.0, mfe=10.0, exit_reason="TrailingStop"))
    for _ in range(n_mh):
        trades.append(_make_trade("2025-03-01T10:00:00Z", profit=2.0, mfe=8.0, exit_reason="MaxHoldBars"))
    return trades


def _metrics(trades: list[dict] | None = None, **overrides) -> ExtractedMetrics:
    if trades is None:
        trades = _make_trades()
    defaults = dict(
        net_pnl=10.0,
        max_drawdown=5.0,
        trade_count=len(trades),
        profit_factor=1.5,
        win_rate=0.55,
        sharpe_ratio=1.0,
        avg_trade=0.1,
        trades=trades,
    )
    defaults.update(overrides)
    return ExtractedMetrics(**defaults)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_registry_contains_trailing_stop():
    assert "trailing_stop" in FITNESS_REGISTRY


def test_get_fitness_returns_trailing_stop_instance():
    f = get_fitness("trailing_stop")
    assert isinstance(f, TrailingStopFitness)


def test_get_fitness_forwards_kwargs():
    f = get_fitness("trailing_stop", min_trades=200)
    assert f.min_trades == 200


# ---------------------------------------------------------------------------
# Hard constraints (is_valid)
# ---------------------------------------------------------------------------

def test_valid_passes_good_metrics():
    f = TrailingStopFitness()
    assert f.is_valid(_metrics()) is True


def test_invalid_too_few_trades():
    f = TrailingStopFitness(min_trades=100)
    assert f.is_valid(_metrics(trade_count=99)) is False


def test_invalid_negative_pnl():
    f = TrailingStopFitness()
    assert f.is_valid(_metrics(net_pnl=-5.0)) is False


def test_invalid_zero_pnl():
    f = TrailingStopFitness()
    assert f.is_valid(_metrics(net_pnl=0.0)) is False


def test_invalid_low_profit_factor():
    f = TrailingStopFitness(min_profit_factor=1.0)
    assert f.is_valid(_metrics(profit_factor=0.99)) is False


def test_invalid_no_trades_data():
    f = TrailingStopFitness()
    assert f.is_valid(_metrics(trades=[])) is False


# ---------------------------------------------------------------------------
# compute() = _base_score() on all trades
# ---------------------------------------------------------------------------

def test_compute_equals_base_score():
    """compute() delegates directly to _base_score on all trades."""
    trades = _make_trades(100)
    f = TrailingStopFitness(min_trades=100)
    m = _metrics(trades=trades)
    assert f.compute(m) == pytest.approx(_base_score(trades, 100))


def test_compute_positive_for_good_trades():
    f = TrailingStopFitness(min_trades=50)
    m = _metrics(trades=_make_trades(100))
    assert f.compute(m) > 0


# ---------------------------------------------------------------------------
# _base_score
# ---------------------------------------------------------------------------

def test_base_score_all_trailing_stop():
    trades = [_make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "TrailingStop") for _ in range(100)]
    score = _base_score(trades, min_trades=100)
    # ts_bonus = 2.0, mfe_eff = 0.5, es_penalty = 1.0, confidence = 1.0
    assert score == pytest.approx(0.5 * 2.0 * 1.0 * 1.0, rel=1e-6)


def test_base_score_no_trailing_stop():
    trades = [_make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "MaxHoldBars") for _ in range(100)]
    score = _base_score(trades, min_trades=100)
    # ts_bonus = 1.0, mfe_eff = 0.5, es_penalty = 1.0, confidence = 1.0
    assert score == pytest.approx(0.5 * 1.0 * 1.0 * 1.0, rel=1e-6)


def test_base_score_emergency_stop_penalty():
    """10% emergency stops → penalty applies above 5% threshold."""
    trades = []
    for _ in range(90):
        trades.append(_make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "MaxHoldBars"))
    for _ in range(10):
        trades.append(_make_trade("2025-03-01T10:00:00Z", -5.0, 2.0, "EmergencyStop"))
    score = _base_score(trades, min_trades=100)
    # es_pct = 0.10, es_penalty = max(0, 1 - 3 * max(0, 0.10 - 0.05)) = 1 - 0.15 = 0.85
    es_penalty = 1.0 - 3.0 * 0.05
    assert es_penalty == pytest.approx(0.85)
    assert score > 0


def test_base_score_high_emergency_stop_zeroes():
    """40% emergency stops → penalty clamps to 0."""
    trades = []
    for _ in range(60):
        trades.append(_make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "MaxHoldBars"))
    for _ in range(40):
        trades.append(_make_trade("2025-03-01T10:00:00Z", -5.0, 2.0, "EmergencyStop"))
    score = _base_score(trades, min_trades=100)
    # es_pct = 0.40, 3 * (0.40 - 0.05) = 1.05 → penalty = max(0, 1 - 1.05) = 0
    assert score == pytest.approx(0.0)


def test_base_score_losers_excluded_from_efficiency():
    """Losing trades with positive MFE should not drag efficiency negative."""
    trades = []
    for _ in range(50):
        trades.append(_make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "MaxHoldBars"))  # winner
    for _ in range(50):
        trades.append(_make_trade("2025-03-01T10:00:00Z", -10.0, 5.0, "MaxHoldBars"))  # loser with MFE
    score = _base_score(trades, min_trades=100)
    # Only winners contribute: mfe_eff = 5/10 = 0.5
    assert score > 0


def test_base_score_zero_mfe_excluded():
    """Trades with mfe=0 excluded from efficiency calc, not division by zero."""
    trades = [
        _make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "MaxHoldBars"),
        _make_trade("2025-03-01T10:00:00Z", 0.0, 0.0, "MaxHoldBars"),  # mfe=0
    ]
    score = _base_score(trades, min_trades=1)
    # Only 1 trade contributes to mfe_efficiency: 5/10 = 0.5
    assert score > 0


def test_base_score_all_zero_mfe():
    """All trades with mfe=0 → mfe_efficiency = 0 → score = 0."""
    trades = [_make_trade("2025-03-01T10:00:00Z", 0.0, 0.0, "MaxHoldBars") for _ in range(10)]
    score = _base_score(trades, min_trades=10)
    assert score == pytest.approx(0.0)


def test_base_score_confidence_scales_with_trade_count():
    trades_few = [_make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "TrailingStop") for _ in range(25)]
    trades_many = [_make_trade("2025-03-01T10:00:00Z", 5.0, 10.0, "TrailingStop") for _ in range(100)]
    score_few = _base_score(trades_few, min_trades=100)
    score_many = _base_score(trades_many, min_trades=100)
    assert score_many > score_few


# ---------------------------------------------------------------------------
# Quarterly bucketing (used by summary display)
# ---------------------------------------------------------------------------

def test_bucket_by_quarter():
    trades = [
        _make_trade("2025-01-15T10:00:00Z", 1.0, 2.0),
        _make_trade("2025-04-15T10:00:00Z", 1.0, 2.0),
        _make_trade("2025-07-15T10:00:00Z", 1.0, 2.0),
        _make_trade("2025-10-15T10:00:00Z", 1.0, 2.0),
    ]
    q = _bucket_by_quarter(trades)
    assert set(q.keys()) == {"2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"}
    assert len(q["2025-Q1"]) == 1
    assert len(q["2025-Q4"]) == 1


def test_bucket_ignores_invalid_timestamps():
    trades = [
        _make_trade("", 1.0, 2.0),
        _make_trade("0001-01-01T00:00:00Z", 1.0, 2.0),
        _make_trade("garbage", 1.0, 2.0),
    ]
    q = _bucket_by_quarter(trades)
    assert len(q) == 0


# ---------------------------------------------------------------------------
# score() integration (base class wrapper)
# ---------------------------------------------------------------------------

def test_score_returns_neginf_when_invalid():
    f = TrailingStopFitness(min_trades=200)
    m = _metrics(trade_count=50)
    assert f.score(m) == float("-inf")


def test_score_returns_compute_when_valid():
    f = TrailingStopFitness(min_trades=50)
    m = _metrics()
    score = f.score(m)
    assert score == pytest.approx(f.compute(m))
    assert score > 0
