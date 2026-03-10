import json
from pathlib import Path
from uuid import uuid4

import pytest

from optimizer.runner.evaluation import Evaluation, EvaluationStatus
from optimizer.results.collector import verify_result
from optimizer.results.extractor import ExtractedMetrics


def _make_eval(results_root: Path) -> Evaluation:
    ev = Evaluation(id=uuid4().hex, parameters={"ticker": "SPY"})
    ev.status = EvaluationStatus.SUCCESS
    return ev


def _valid_result_json() -> dict:
    return {
        "runtimeStatistics": {
            "Equity": "1234.56",
            "Return": "0.23",
        },
        "statistics": {
            "Sharpe Ratio": "1.45",
            "Total Trades": "42",
            "Net Profit": "23.4%",
            "Drawdown": "5.0%",
            "Win Rate": "55%",
            "Average Win": "1.47%",
            "Average Loss": "-0.95%",
        },
        "alphaRuntimeStatistics": {},
        "charts": {},
        "orders": {},
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_valid_result_returns_metrics(tmp_path):
    """A properly structured result JSON returns an ExtractedMetrics instance."""
    ev = _make_eval(tmp_path)
    run_dir = tmp_path / ev.id
    run_dir.mkdir(parents=True)
    result_file = run_dir / f"{ev.id}.json"
    result_file.write_text(json.dumps(_valid_result_json()))

    result = verify_result(ev, tmp_path)

    assert result is not None
    assert isinstance(result, ExtractedMetrics)
    assert result.sharpe_ratio == pytest.approx(1.45)
    assert result.net_pnl == pytest.approx(23.4)
    assert result.trade_count == 42
    assert result.win_rate == pytest.approx(0.55)


def test_missing_file_returns_none(tmp_path):
    """Non-existent result file → None returned, no exception."""
    ev = _make_eval(tmp_path)
    # Do NOT create the file

    result = verify_result(ev, tmp_path)

    assert result is None


def test_empty_file_returns_none(tmp_path):
    """Zero-byte result file → None returned."""
    ev = _make_eval(tmp_path)
    run_dir = tmp_path / ev.id
    run_dir.mkdir(parents=True)
    (run_dir / f"{ev.id}.json").write_bytes(b"")

    result = verify_result(ev, tmp_path)

    assert result is None


def test_missing_statistics_key_returns_none(tmp_path):
    """JSON that lacks 'statistics' key → None returned."""
    ev = _make_eval(tmp_path)
    run_dir = tmp_path / ev.id
    run_dir.mkdir(parents=True)
    bad_json = {"runtimeStatistics": {"Equity": "1000"}}  # no 'statistics'
    (run_dir / f"{ev.id}.json").write_text(json.dumps(bad_json))

    result = verify_result(ev, tmp_path)

    assert result is None


def test_result_dir_deleted_on_success(tmp_path):
    """Result directory is removed after successful verification (scratch cleanup)."""
    ev = _make_eval(tmp_path)
    run_dir = tmp_path / ev.id
    run_dir.mkdir(parents=True)
    (run_dir / f"{ev.id}.json").write_text(json.dumps(_valid_result_json()))

    verify_result(ev, tmp_path)

    assert not run_dir.exists()


def test_result_dir_kept_on_failure(tmp_path):
    """Result directory is NOT deleted when verification fails (leave for inspection)."""
    ev = _make_eval(tmp_path)
    run_dir = tmp_path / ev.id
    run_dir.mkdir(parents=True)
    bad_json = {"runtimeStatistics": {"Equity": "1000"}}  # missing 'statistics'
    (run_dir / f"{ev.id}.json").write_text(json.dumps(bad_json))

    verify_result(ev, tmp_path)

    assert run_dir.exists()


def test_missing_metric_field_returns_none(tmp_path):
    """Statistics dict missing a required field (e.g. Win Rate) → None returned."""
    ev = _make_eval(tmp_path)
    run_dir = tmp_path / ev.id
    run_dir.mkdir(parents=True)
    incomplete = _valid_result_json()
    del incomplete["statistics"]["Win Rate"]
    (run_dir / f"{ev.id}.json").write_text(json.dumps(incomplete))

    result = verify_result(ev, tmp_path)

    assert result is None
