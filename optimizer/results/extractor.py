import math
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


def _parse_pct(value: str) -> float:
    """Strip trailing '%' and surrounding whitespace, return float."""
    return float(value.strip().rstrip("%").strip())


@dataclass
class ExtractedMetrics:
    """Structured backtest metrics extracted from a LEAN result JSON."""

    net_pnl: float         # percentage (e.g. 23.4 means 23.4%)
    max_drawdown: float    # percentage (e.g. 5.0 means 5.0%)
    trade_count: int
    profit_factor: float   # (win_rate × avg_win) / ((1 − win_rate) × |avg_loss|)
    win_rate: float        # fraction 0–1 (e.g. 0.55 means 55%)
    sharpe_ratio: float
    avg_trade: float       # net_pnl / trade_count; 0.0 when trade_count == 0
    raw: dict = field(default_factory=dict, repr=False)
    trades: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for Evaluation.metrics storage)."""
        d = {
            "net_pnl": self.net_pnl,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_trade": self.avg_trade,
        }
        if self.trades:
            d["trades"] = self.trades
        return d


def _parse_order_tag(tag: str) -> dict[str, str]:
    """Parse pipe-delimited key=value order tag into a dict."""
    result = {}
    if not tag:
        return result
    for part in tag.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def _extract_trades(data: dict) -> list[dict]:
    """Extract per-trade records from closedTrades + orders.

    Each trade dict has: exitTime, profit, mfe, exit_reason.
    """
    closed_trades = data.get("totalPerformance", {}).get("closedTrades", [])
    orders = data.get("orders", {})
    if not closed_trades:
        return []

    trades = []
    for ct in closed_trades:
        order_ids = ct.get("orderIds", [])
        exit_reason = "Unknown"
        tag_mfe = None

        if len(order_ids) >= 2:
            exit_order = orders.get(str(order_ids[1]), {})
            tag = _parse_order_tag(exit_order.get("tag", ""))
            exit_reason = tag.get("reason", "Unknown")
            if "mfe" in tag:
                try:
                    tag_mfe = abs(float(tag["mfe"]))
                except (ValueError, TypeError):
                    pass

        # MFE: prefer order tag (authoritative), fallback to closedTrades
        if tag_mfe is not None:
            mfe = tag_mfe
        else:
            mfe = abs(float(ct.get("mfe", 0)))

        trades.append({
            "exitTime": ct.get("exitTime", ""),
            "profit": float(ct.get("profitLoss", 0)),
            "mfe": mfe,
            "exit_reason": exit_reason,
        })
    return trades


def extract(data: dict) -> Optional[ExtractedMetrics]:
    """Extract structured metrics from a parsed LEAN result dict.

    Args:
        data: Full parsed LEAN result JSON (must contain 'statistics' key).

    Returns:
        ExtractedMetrics on success, None on any missing field or parse error.
    """
    stats = data.get("statistics", {})
    try:
        net_pnl = _parse_pct(stats["Net Profit"])
        max_drawdown = _parse_pct(stats["Drawdown"])
        trade_count = int(
            stats.get("Total Orders", stats.get("Total Trades", "0"))
        )
        win_rate = _parse_pct(stats["Win Rate"]) / 100.0
        sharpe_ratio = float(stats["Sharpe Ratio"])
        avg_win = _parse_pct(stats["Average Win"])
        avg_loss = abs(_parse_pct(stats["Average Loss"]))

        # profit_factor = (win_rate × avg_win) / ((1 − win_rate) × |avg_loss|)
        loss_rate = 1.0 - win_rate
        if loss_rate == 0.0 or avg_loss == 0.0:
            profit_factor = float("inf") if win_rate > 0.0 else 0.0
        else:
            profit_factor = (win_rate * avg_win) / (loss_rate * avg_loss)

        avg_trade = net_pnl / trade_count if trade_count > 0 else 0.0

        trades = _extract_trades(data)

        return ExtractedMetrics(
            net_pnl=net_pnl,
            max_drawdown=max_drawdown,
            trade_count=trade_count,
            profit_factor=profit_factor,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            avg_trade=avg_trade,
            raw=stats,
            trades=trades,
        )

    except (KeyError, ValueError, ZeroDivisionError) as e:
        logger.warning(f"Failed to extract metrics: {e}")
        return None
