import math

import pytest

from optimizer.results.extractor import ExtractedMetrics, extract


def _lean_result(stats_overrides: dict | None = None) -> dict:
    """Full-featured LEAN result dict with sensible defaults."""
    stats = {
        "Net Profit": "23.4%",
        "Drawdown": "5.0%",
        "Total Orders": "42",
        "Win Rate": "55%",
        "Sharpe Ratio": "1.45",
        "Average Win": "1.47%",
        "Average Loss": "-0.95%",
    }
    if stats_overrides:
        stats.update(stats_overrides)
    return {"runtimeStatistics": {}, "statistics": stats}


# ---------------------------------------------------------------------------
# basic extraction
# ---------------------------------------------------------------------------

def test_extract_returns_metrics():
    result = extract(_lean_result())
    assert isinstance(result, ExtractedMetrics)


def test_net_pnl_parsed():
    result = extract(_lean_result({"Net Profit": "184.318%"}))
    assert result.net_pnl == pytest.approx(184.318)


def test_max_drawdown_parsed():
    result = extract(_lean_result({"Drawdown": "11.500%"}))
    assert result.max_drawdown == pytest.approx(11.5)


def test_trade_count_from_total_orders():
    result = extract(_lean_result({"Total Orders": "580"}))
    assert result.trade_count == 580


def test_trade_count_fallback_to_total_trades():
    data = _lean_result()
    del data["statistics"]["Total Orders"]
    data["statistics"]["Total Trades"] = "99"
    result = extract(data)
    assert result.trade_count == 99


def test_win_rate_stored_as_fraction():
    result = extract(_lean_result({"Win Rate": "55%"}))
    assert result.win_rate == pytest.approx(0.55)


def test_sharpe_ratio_parsed():
    result = extract(_lean_result({"Sharpe Ratio": "3.822"}))
    assert result.sharpe_ratio == pytest.approx(3.822)


# ---------------------------------------------------------------------------
# derived fields
# ---------------------------------------------------------------------------

def test_profit_factor_computed():
    # win_rate=0.55, avg_win=1.47, avg_loss=|-0.95|=0.95
    # pf = (0.55 × 1.47) / (0.45 × 0.95)
    expected = (0.55 * 1.47) / (0.45 * 0.95)
    result = extract(_lean_result())
    assert result.profit_factor == pytest.approx(expected, rel=1e-4)


def test_profit_factor_100pct_win_rate_is_inf():
    result = extract(_lean_result({"Win Rate": "100%", "Average Loss": "0.00%"}))
    assert math.isinf(result.profit_factor)


def test_avg_trade_is_net_pnl_over_count():
    # net_pnl=23.4, trade_count=42
    result = extract(_lean_result())
    assert result.avg_trade == pytest.approx(23.4 / 42, rel=1e-4)


def test_avg_trade_zero_when_no_trades():
    result = extract(_lean_result({"Total Orders": "0"}))
    assert result.avg_trade == 0.0


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

def test_to_dict_has_expected_keys():
    result = extract(_lean_result())
    d = result.to_dict()
    assert set(d.keys()) == {
        "net_pnl", "max_drawdown", "trade_count",
        "profit_factor", "win_rate", "sharpe_ratio", "avg_trade",
    }


def test_to_dict_values_match_fields():
    result = extract(_lean_result())
    d = result.to_dict()
    assert d["net_pnl"] == result.net_pnl
    assert d["trade_count"] == result.trade_count


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

def test_missing_required_field_returns_none():
    data = _lean_result()
    del data["statistics"]["Win Rate"]
    assert extract(data) is None


def test_unparseable_float_returns_none():
    assert extract(_lean_result({"Sharpe Ratio": "N/A"})) is None


def test_raw_field_contains_original_stats():
    data = _lean_result()
    result = extract(data)
    assert result.raw is data["statistics"]
