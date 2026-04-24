from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .data import BacktestConfig, spread_data_missing


def max_losing_streak(pnls: pd.Series) -> int:
    max_streak = cur = 0
    for x in pnls.fillna(0):
        if x < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def drawdown_stats(equity_df: pd.DataFrame) -> Tuple[float, float]:
    if equity_df.empty:
        return 0.0, 0.0
    eq = equity_df["marked_equity"].astype(float)
    peak = eq.cummax()
    dd_abs = eq - peak
    dd_pct = (eq / peak - 1.0) * 100.0
    return float(dd_pct.min()), float(dd_abs.min())


def monthly_returns(equity_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if equity_df.empty:
        return []
    e = equity_df.copy()
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
    e = e.set_index("timestamp")
    rows = []
    for month, g in e.groupby(pd.Grouper(freq="ME")):
        if g.empty:
            continue
        start = float(g["marked_equity"].iloc[0])
        end = float(g["marked_equity"].iloc[-1])
        rows.append(
            {
                "month": month.strftime("%Y-%m"),
                "starting_equity": start,
                "ending_equity": end,
                "monthly_return_pct": (
                    (end / start - 1.0) * 100.0 if start > 0 else np.nan
                ),
            }
        )
    return rows


def side_breakdown(trades: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if trades.empty:
        return out
    for side, g in trades.groupby("side"):
        gp = g.loc[g["net_pnl"] > 0, "net_pnl"].sum()
        gl = abs(g.loc[g["net_pnl"] < 0, "net_pnl"].sum())
        out[side] = {
            "trades": int(len(g)),
            "win_rate": float((g["net_pnl"] > 0).mean() * 100.0),
            "profit_factor": None if gl == 0 else float(gp / gl),
            "expectancy_r": float(g["r_multiple"].mean()),
            "net_pnl": float(g["net_pnl"].sum()),
        }
    return out


def collect_warnings(df: pd.DataFrame, cfg: BacktestConfig) -> List[str]:
    warnings: List[str] = []
    if cfg.allow_shorts and cfg.market_type not in {"margin", "futures"}:
        warnings.append(
            "Shorts enabled while market_type is not margin/futures; borrow, funding, margin, and liquidation are not modeled."
        )
    if spread_data_missing(df, cfg):
        warnings.append(
            f"spread_pct missing/partial in last_trade_ohlc mode; using default_spread_pct={cfg.default_spread_pct}."
        )
    missing = int(df["missing_before"].sum()) if "missing_before" in df else 0
    if missing > cfg.missing_candle_warning_threshold:
        warnings.append(
            f"Missing candles detected: {missing}, threshold={cfg.missing_candle_warning_threshold}; gaps cancel pending entries and close open positions."
        )
    if cfg.execution_mode == "bid_ask":
        warnings.append(
            "bid_ask mode assumes bid/ask OHLC columns are executable quote extremes and synchronized with trade candles."
        )
    return warnings


def build_report(
    df: pd.DataFrame, trades: pd.DataFrame, equity_df: pd.DataFrame, cfg: BacktestConfig
) -> Dict[str, Any]:
    total_trades = int(len(trades))
    final_equity = (
        float(equity_df["closed_equity"].iloc[-1])
        if not equity_df.empty
        else cfg.initial_equity
    )
    valid_candles = (
        int(df["valid_candle"].sum()) if "valid_candle" in df else int(len(df))
    )
    missing_candles = int(df["missing_before"].sum()) if "missing_before" in df else 0

    if total_trades > 0:
        wins = trades["net_pnl"] > 0
        gross_profit = float(trades.loc[wins, "net_pnl"].sum())
        gross_loss = float(abs(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()))
        profit_factor = None if gross_loss == 0 else gross_profit / gross_loss
        win_rate = float(wins.mean() * 100.0)
        expectancy_usd = float(trades["net_pnl"].mean())
        expectancy_r = float(trades["r_multiple"].mean())
        median_r = float(trades["r_multiple"].median())
        max_ls = max_losing_streak(trades["net_pnl"])
        avg_dur = float(trades["hold_candles"].mean())
        exit_dist = {
            str(k): int(v) for k, v in trades["exit_reason"].value_counts().items()
        }
    else:
        profit_factor = None
        win_rate = 0.0
        expectancy_usd = 0.0
        expectancy_r = 0.0
        median_r = 0.0
        max_ls = 0
        avg_dur = 0.0
        exit_dist = {}

    mdd_pct, mdd_usd = drawdown_stats(equity_df)
    risk_summary = {
        "gross_risk_usd": (
            float(trades["gross_risk_usd"].mean())
            if total_trades > 0 and "gross_risk_usd" in trades
            else 0.0
        ),
        "estimated_costs_usd": (
            float(trades["estimated_costs_usd"].mean())
            if total_trades > 0 and "estimated_costs_usd" in trades
            else 0.0
        ),
        "net_risk_usd": (
            float(trades["net_risk_usd"].mean())
            if total_trades > 0 and "net_risk_usd" in trades
            else 0.0
        ),
        "risk_target_usd": (
            float(trades["risk_target_usd"].mean())
            if total_trades > 0 and "risk_target_usd" in trades
            else 0.0
        ),
        "risk_overshoot_flag": (
            bool(trades["risk_overshoot_flag"].any())
            if total_trades > 0 and "risk_overshoot_flag" in trades
            else False
        ),
    }
    time_in_market = (
        float(equity_df["open_position_side"].notna().mean() * 100.0)
        if not equity_df.empty
        else 0.0
    )
    months = monthly_returns(equity_df)
    positive_months = [
        m["monthly_return_pct"]
        for m in months
        if pd.notna(m["monthly_return_pct"]) and m["monthly_return_pct"] > 0
    ]
    one_month_dependency = bool(
        positive_months
        and sum(positive_months) > 0
        and max(positive_months) / sum(positive_months) > 0.60
    )

    conclusion = "FAIL / DO NOT AUTOMATE LIVE"
    reasons: List[str] = []
    if total_trades < 300:
        conclusion = "INCONCLUSIVE"
        reasons.append("total_trades < 300")
    else:
        if profit_factor is None or profit_factor <= 1.2:
            reasons.append("profit_factor <= 1.2 or undefined")
        if expectancy_r <= 0:
            reasons.append("expectancy_r <= 0")
        if abs(mdd_pct) > 10:
            reasons.append("max_drawdown_pct > 10")
        if one_month_dependency:
            reasons.append("returns appear dependent on one month")
        if not reasons:
            conclusion = "BASE PASS ONLY - REQUIRES 2-5 BPS SLIPPAGE AND PARAMETER ROBUSTNESS REVIEW"

    return {
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "date_start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "date_end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "initial_equity": cfg.initial_equity,
        "final_equity": final_equity,
        "fee_rate": cfg.fee_rate,
        "slippage_bps": cfg.slippage_bps,
        "execution_mode": cfg.execution_mode,
        "allow_shorts": cfg.allow_shorts,
        "market_type": cfg.market_type,
        "total_candles": int(len(df)),
        "valid_candles": valid_candles,
        "missing_candles": missing_candles,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        **risk_summary,
        "expectancy_usd": expectancy_usd,
        "expectancy_r": expectancy_r,
        "average_r": expectancy_r,
        "median_r": median_r,
        "max_drawdown_pct": mdd_pct,
        "max_drawdown_usd": mdd_usd,
        "max_losing_streak": int(max_ls),
        "average_trade_duration": avg_dur,
        "time_in_market": time_in_market,
        "monthly_return_table": months,
        "exit_reason_distribution": exit_dist,
        "long_short_breakdown": side_breakdown(trades),
        "one_month_dependency": one_month_dependency,
        "parameter_robustness": "not_run_in_single_backtest; inspect *_robustness.csv if --robustness used",
        "warnings": collect_warnings(df, cfg),
        "conclusion": conclusion,
        "conclusion_reasons": reasons,
    }
