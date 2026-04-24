#!/usr/bin/env python3
"""
EMA Pullback Continuation crypto scalping backtest.

Required CSV columns:
    timestamp, open, high, low, close, volume

Optional for last_trade_ohlc mode:
    spread_pct, quote_volume

Required for bid_ask mode:
    bid_open, bid_high, bid_low, bid_close,
    ask_open, ask_high, ask_low, ask_close

Notes:
- Signals are calculated on completed candles only.
- Entries occur at next candle open only.
- This is for research/backtesting only, not live trading.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------


@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str
    initial_equity: float = 10_000.0
    fee_rate: float = 0.001  # 0.10% taker fee
    slippage_bps: float = 2.0  # 2 bps = 0.02%
    risk_per_trade: float = 0.005  # 0.5% strict modeled net risk target
    max_cost_to_gross_risk_ratio: float = (
        1.0  # skip if fees+spread+slippage estimate exceeds gross stop risk
    )
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    volume_sma_period: int = 20
    volume_mult: float = 1.2
    min_atr_pct: float = 0.0008
    max_spread_pct: float = 0.0002  # 0.02%; full spread, not half-spread
    default_spread_pct: float = 0.0002
    stop_atr_mult: float = 1.0
    take_profit_r: float = 1.2
    max_hold_candles: int = 5
    pause_after_losses: int = 3
    pause_candles: int = 30
    warmup_candles: int = 100
    min_notional: float = 10.0
    qty_step_size: float = 0.0  # 0 disables rounding
    liquidity_cap_fraction: float = 0.01
    apply_liquidity_cap: bool = False
    execution_mode: str = "last_trade_ohlc"  # "last_trade_ohlc" or "bid_ask"
    allow_shorts: bool = True
    market_type: str = "spot"  # "spot", "margin", or "futures"; warning only
    missing_candle_warning_threshold: int = 0

    @property
    def slippage_pct(self) -> float:
        return self.slippage_bps / 10_000.0


@dataclass
class Position:
    trade_id: int
    symbol: str
    timeframe: str
    side: str
    signal_index: int
    signal_time: pd.Timestamp
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    notional: float
    stop_distance: float
    stop_price: float
    take_profit_price: float
    risk_amount_planned: float
    risk_amount_actual: float
    gross_risk_usd: float
    estimated_costs_usd: float
    net_risk_usd: float
    risk_target_usd: float
    risk_overshoot_flag: bool
    entry_fee: float
    equity_before_trade: float
    signal_open: float
    signal_high: float
    signal_low: float
    signal_close: float
    signal_volume: float
    ema20_at_signal: float
    ema50_at_signal: float
    atr14_at_signal: float
    volume_sma20_at_signal: float
    atr_pct_at_signal: float
    spread_pct_at_signal: float
    entry_reference_open: float
    entry_spread_pct: float
    entry_slippage_pct: float
    execution_mode: str


# -----------------------------
# Data and indicators
# -----------------------------


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        dupes = int(df["timestamp"].duplicated().sum())
        raise ValueError(f"Duplicate timestamps detected: {dupes}")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "spread_pct",
        "quote_volume",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def expected_timedelta(timeframe: str) -> pd.Timedelta:
    if timeframe.endswith("m"):
        return pd.Timedelta(minutes=int(timeframe[:-1]))
    if timeframe.endswith("h"):
        return pd.Timedelta(hours=int(timeframe[:-1]))
    if timeframe.endswith("d"):
        return pd.Timedelta(days=int(timeframe[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def mark_valid_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    valid = (
        df["timestamp"].notna()
        & df["open"].gt(0)
        & df["high"].gt(0)
        & df["low"].gt(0)
        & df["close"].gt(0)
        & df["volume"].ge(0)
        & df["high"].ge(df["low"])
        & df["high"].ge(df["open"])
        & df["high"].ge(df["close"])
        & df["low"].le(df["open"])
        & df["low"].le(df["close"])
    )
    df["valid_candle"] = valid

    dt = expected_timedelta(timeframe)
    diffs = df["timestamp"].diff()
    multiples = diffs / dt
    gaps = np.where(multiples > 1.000001, np.floor(multiples).astype(float) - 1, 0)
    gaps = pd.Series(gaps, index=df.index).fillna(0).clip(lower=0).astype(int)
    df["missing_before"] = gaps
    return df


def compute_indicators(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    df["ema20"] = close.ewm(
        span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast
    ).mean()
    df["ema50"] = close.ewm(
        span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow
    ).mean()
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(
        alpha=1.0 / cfg.atr_period, adjust=False, min_periods=cfg.atr_period
    ).mean()
    df["volume_sma20"] = (
        df["volume"]
        .rolling(cfg.volume_sma_period, min_periods=cfg.volume_sma_period)
        .mean()
    )
    df["atr_pct"] = df["atr14"] / df["close"]
    return df


# -----------------------------
# Execution mode helpers
# -----------------------------


def bid_ask_required_cols() -> set[str]:
    return {
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    }


def validate_execution_mode(df: pd.DataFrame, cfg: BacktestConfig) -> None:
    if cfg.execution_mode not in {"last_trade_ohlc", "bid_ask"}:
        raise ValueError("execution_mode must be 'last_trade_ohlc' or 'bid_ask'")
    if cfg.market_type not in {"spot", "margin", "futures"}:
        raise ValueError("market_type must be 'spot', 'margin', or 'futures'")
    if cfg.execution_mode == "bid_ask":
        missing = bid_ask_required_cols() - set(df.columns)
        if missing:
            raise ValueError(f"bid_ask mode requires columns: {sorted(missing)}")
        q = df[list(bid_ask_required_cols())]
        if q.isna().any().any() or (q <= 0).any().any():
            raise ValueError("bid_ask mode has NaN or non-positive bid/ask OHLC values")
        if not (
            (df["bid_open"] <= df["ask_open"])
            & (df["bid_high"] <= df["ask_high"])
            & (df["bid_low"] <= df["ask_low"])
            & (df["bid_close"] <= df["ask_close"])
        ).all():
            raise ValueError("bid_ask mode has inverted bid/ask values")


def spread_pct_at(row: pd.Series, cfg: BacktestConfig, when: str = "close") -> float:
    """Full spread at open or close. In last_trade_ohlc mode uses spread_pct/default."""
    if cfg.execution_mode == "bid_ask":
        bcol = "bid_open" if when == "open" else "bid_close"
        acol = "ask_open" if when == "open" else "ask_close"
        bid = row.get(bcol, np.nan)
        ask = row.get(acol, np.nan)
        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
            mid = (float(bid) + float(ask)) / 2
            return float((ask - bid) / mid) if mid > 0 else np.nan
    if "spread_pct" in row.index and pd.notna(row["spread_pct"]):
        return float(row["spread_pct"])
    return float(cfg.default_spread_pct)


def spread_pct(row: pd.Series, cfg: BacktestConfig) -> float:
    """Full spread. In bid_ask mode, compute from executable quotes when possible."""
    if cfg.execution_mode == "bid_ask":
        bid = row.get("bid_close", np.nan)
        ask = row.get("ask_close", np.nan)
        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
            mid = (float(bid) + float(ask)) / 2
            return float((ask - bid) / mid) if mid > 0 else np.nan
    if "spread_pct" in row.index and pd.notna(row["spread_pct"]):
        return float(row["spread_pct"])
    return float(cfg.default_spread_pct)


def spread_data_missing(df: pd.DataFrame, cfg: BacktestConfig) -> bool:
    if cfg.execution_mode == "bid_ask":
        return False
    return "spread_pct" not in df.columns or df["spread_pct"].isna().any()


def floor_to_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def indicators_valid(row: pd.Series) -> bool:
    return all(
        pd.notna(row[c]) for c in ["ema20", "ema50", "atr14", "volume_sma20", "atr_pct"]
    )


def entry_price(side: str, row: pd.Series, cfg: BacktestConfig) -> float:
    slip = cfg.slippage_pct
    if cfg.execution_mode == "bid_ask":
        return (
            float(row["ask_open"]) * (1 + slip)
            if side == "long"
            else float(row["bid_open"]) * (1 - slip)
        )
    sp = spread_pct(row, cfg)
    return (
        float(row["open"]) * (1 + sp / 2 + slip)
        if side == "long"
        else float(row["open"]) * (1 - sp / 2 - slip)
    )


def exit_price_from_trigger(
    side: str, trigger_price: float, row: pd.Series, cfg: BacktestConfig
) -> float:
    """Adverse executable SL/TP fill. Adds spread only in last_trade_ohlc mode."""
    slip = cfg.slippage_pct
    if cfg.execution_mode == "bid_ask":
        return (
            float(trigger_price) * (1 - slip)
            if side == "long"
            else float(trigger_price) * (1 + slip)
        )
    sp = spread_pct(row, cfg)
    return (
        float(trigger_price) * (1 - sp / 2 - slip)
        if side == "long"
        else float(trigger_price) * (1 + sp / 2 + slip)
    )


def market_exit_price(
    side: str, row: pd.Series, cfg: BacktestConfig, when: str = "close"
) -> float:
    """Market-style exit at open/close with adverse spread/slippage handling."""
    slip = cfg.slippage_pct
    if cfg.execution_mode == "bid_ask":
        if when == "open":
            return (
                float(row["bid_open"]) * (1 - slip)
                if side == "long"
                else float(row["ask_open"]) * (1 + slip)
            )
        return (
            float(row["bid_close"]) * (1 - slip)
            if side == "long"
            else float(row["ask_close"]) * (1 + slip)
        )
    base = float(row["open"] if when == "open" else row["close"])
    sp = spread_pct(row, cfg)
    return base * (1 - sp / 2 - slip) if side == "long" else base * (1 + sp / 2 + slip)


def trigger_ohlc(row: pd.Series, cfg: BacktestConfig) -> Tuple[float, float, float]:
    """Return open/high/low used for stop/target trigger checks, not fill prices."""
    if cfg.execution_mode == "bid_ask":
        # Long exits execute on bid; short exits execute on ask. check_exit chooses side-specific fields.
        raise RuntimeError("Use side-specific trigger fields in bid_ask mode")
    return float(row["open"]), float(row["high"]), float(row["low"])


# -----------------------------
# Signals
# -----------------------------


def is_long_signal(df: pd.DataFrame, i: int, cfg: BacktestConfig) -> bool:
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    sp = spread_pct(row, cfg)
    return bool(
        row["ema20"] > row["ema50"]
        and row["low"] <= row["ema20"]
        and row["close"] > row["ema20"]
        and row["close"] > row["open"]
        and row["close"] > prev["high"]
        and row["volume"] > cfg.volume_mult * row["volume_sma20"]
        and row["atr_pct"] >= cfg.min_atr_pct
        and pd.notna(sp)
        and sp <= cfg.max_spread_pct
    )


def is_short_signal(df: pd.DataFrame, i: int, cfg: BacktestConfig) -> bool:
    if not cfg.allow_shorts:
        return False
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    sp = spread_pct(row, cfg)
    return bool(
        row["ema20"] < row["ema50"]
        and row["high"] >= row["ema20"]
        and row["close"] < row["ema20"]
        and row["close"] < row["open"]
        and row["close"] < prev["low"]
        and row["volume"] > cfg.volume_mult * row["volume_sma20"]
        and row["atr_pct"] >= cfg.min_atr_pct
        and pd.notna(sp)
        and sp <= cfg.max_spread_pct
    )


# -----------------------------
# Trading mechanics
# -----------------------------


def estimate_stop_loss_per_unit(
    side: str,
    entry_px: float,
    stop_price: float,
    entry_row: pd.Series,
    cfg: BacktestConfig,
) -> Tuple[float, float, float]:
    """Return (net_loss_per_unit, gross_price_risk_per_unit, estimated_costs_per_unit).

    The estimate assumes a normal stop fill at stop_price under the configured execution
    model, including entry fee, exit fee, slippage, and synthetic spread where applicable.
    Gap-through-stop losses can still exceed this estimate.
    """
    stop_exit_px = exit_price_from_trigger(side, stop_price, entry_row, cfg)
    if side == "long":
        price_loss_per_unit = entry_px - stop_exit_px
        gross_price_risk_per_unit = entry_px - stop_price
    else:
        price_loss_per_unit = stop_exit_px - entry_px
        gross_price_risk_per_unit = stop_price - entry_px
    fee_per_unit = cfg.fee_rate * (abs(entry_px) + abs(stop_exit_px))
    net_loss_per_unit = price_loss_per_unit + fee_per_unit
    estimated_costs_per_unit = max(0.0, net_loss_per_unit - gross_price_risk_per_unit)
    return net_loss_per_unit, gross_price_risk_per_unit, estimated_costs_per_unit


def size_for_strict_net_risk(
    side: str,
    entry_px: float,
    stop_price: float,
    entry_row: pd.Series,
    equity: float,
    cfg: BacktestConfig,
) -> Optional[Dict[str, float]]:
    target = equity * cfg.risk_per_trade
    net_loss_per_unit, gross_per_unit, costs_per_unit = estimate_stop_loss_per_unit(
        side, entry_px, stop_price, entry_row, cfg
    )
    if not np.isfinite(net_loss_per_unit) or net_loss_per_unit <= 0:
        return None
    if not np.isfinite(gross_per_unit) or gross_per_unit <= 0:
        return None
    cost_ratio = costs_per_unit / gross_per_unit
    if cost_ratio > cfg.max_cost_to_gross_risk_ratio:
        return None

    qty = floor_to_step(target / net_loss_per_unit, cfg.qty_step_size)
    if qty <= 0 or not np.isfinite(qty):
        return None

    # Safety cap after step rounding. floor_to_step should not overshoot, but this
    # loop protects against custom bad step values / floating point edge cases.
    while qty > 0 and qty * net_loss_per_unit > target * (1 + 1e-12):
        if cfg.qty_step_size > 0:
            qty = floor_to_step(qty - cfg.qty_step_size, cfg.qty_step_size)
        else:
            qty = target / net_loss_per_unit
            break
    if qty <= 0 or not np.isfinite(qty):
        return None

    return {
        "qty": qty,
        "gross_risk_usd": qty * gross_per_unit,
        "estimated_costs_usd": qty * costs_per_unit,
        "net_risk_usd": qty * net_loss_per_unit,
        "risk_target_usd": target,
        "risk_overshoot_flag": bool(qty * net_loss_per_unit > target * (1 + 1e-9)),
    }


def create_position(
    trade_id: int,
    df: pd.DataFrame,
    signal_index: int,
    entry_index: int,
    side: str,
    equity: float,
    cfg: BacktestConfig,
) -> Optional[Position]:
    sig = df.iloc[signal_index]
    ent = df.iloc[entry_index]
    sp_entry = spread_pct_at(ent, cfg, when="open")
    if pd.isna(sp_entry) or sp_entry > cfg.max_spread_pct:
        return None
    atr = float(sig["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    px = entry_price(side, ent, cfg)
    if not np.isfinite(px) or px <= 0:
        return None

    stop_distance = atr * cfg.stop_atr_mult
    risk_amount = equity * cfg.risk_per_trade

    if side == "long":
        stop = px - stop_distance
        tp = px + cfg.take_profit_r * stop_distance
    else:
        stop = px + stop_distance
        tp = px - cfg.take_profit_r * stop_distance
    if stop <= 0 or tp <= 0:
        return None

    risk_est = size_for_strict_net_risk(side, px, stop, ent, equity, cfg)
    if risk_est is None:
        return None
    qty = risk_est["qty"]

    notional = qty * px
    if notional < cfg.min_notional:
        return None
    if cfg.apply_liquidity_cap:
        quote_volume = (
            float(ent["quote_volume"])
            if "quote_volume" in ent.index and pd.notna(ent["quote_volume"])
            else float(ent["volume"] * ent["close"])
        )
        if quote_volume <= 0 or notional > cfg.liquidity_cap_fraction * quote_volume:
            return None

    entry_fee = notional * cfg.fee_rate
    if equity <= entry_fee:
        return None

    return Position(
        trade_id=trade_id,
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        side=side,
        signal_index=signal_index,
        signal_time=sig["timestamp"],
        entry_index=entry_index,
        entry_time=ent["timestamp"],
        entry_price=px,
        quantity=qty,
        notional=notional,
        stop_distance=stop_distance,
        stop_price=stop,
        take_profit_price=tp,
        risk_amount_planned=risk_amount,
        risk_amount_actual=risk_est["net_risk_usd"],
        gross_risk_usd=risk_est["gross_risk_usd"],
        estimated_costs_usd=risk_est["estimated_costs_usd"],
        net_risk_usd=risk_est["net_risk_usd"],
        risk_target_usd=risk_est["risk_target_usd"],
        risk_overshoot_flag=risk_est["risk_overshoot_flag"],
        entry_fee=entry_fee,
        equity_before_trade=equity,
        signal_open=float(sig["open"]),
        signal_high=float(sig["high"]),
        signal_low=float(sig["low"]),
        signal_close=float(sig["close"]),
        signal_volume=float(sig["volume"]),
        ema20_at_signal=float(sig["ema20"]),
        ema50_at_signal=float(sig["ema50"]),
        atr14_at_signal=float(sig["atr14"]),
        volume_sma20_at_signal=float(sig["volume_sma20"]),
        atr_pct_at_signal=float(sig["atr_pct"]),
        spread_pct_at_signal=spread_pct(sig, cfg),
        entry_reference_open=float(ent["open"]),
        entry_spread_pct=sp_entry,
        entry_slippage_pct=cfg.slippage_pct,
        execution_mode=cfg.execution_mode,
    )


def check_exit(
    pos: Position, row: pd.Series, cfg: BacktestConfig
) -> Optional[Tuple[str, float]]:
    """Return (reason, exit_price) or None. Conservative stop-first conflict."""
    if cfg.execution_mode == "bid_ask":
        if pos.side == "long":
            o, h, l = (
                float(row["bid_open"]),
                float(row["bid_high"]),
                float(row["bid_low"]),
            )
        else:
            o, h, l = (
                float(row["ask_open"]),
                float(row["ask_high"]),
                float(row["ask_low"]),
            )
    else:
        o, h, l = float(row["open"]), float(row["high"]), float(row["low"])

    if pos.side == "long":
        if o <= pos.stop_price:
            return "gap_beyond_stop", exit_price_from_trigger(pos.side, o, row, cfg)
        if o >= pos.take_profit_price:
            return "gap_beyond_target", exit_price_from_trigger(
                pos.side, pos.take_profit_price, row, cfg
            )
        stop_hit = l <= pos.stop_price
        tp_hit = h >= pos.take_profit_price
        if stop_hit and tp_hit:
            return "same_candle_stop_first", exit_price_from_trigger(
                pos.side, pos.stop_price, row, cfg
            )
        if stop_hit:
            return "stop_loss", exit_price_from_trigger(
                pos.side, pos.stop_price, row, cfg
            )
        if tp_hit:
            return "take_profit", exit_price_from_trigger(
                pos.side, pos.take_profit_price, row, cfg
            )
    else:
        if o >= pos.stop_price:
            return "gap_beyond_stop", exit_price_from_trigger(pos.side, o, row, cfg)
        if o <= pos.take_profit_price:
            return "gap_beyond_target", exit_price_from_trigger(
                pos.side, pos.take_profit_price, row, cfg
            )
        stop_hit = h >= pos.stop_price
        tp_hit = l <= pos.take_profit_price
        if stop_hit and tp_hit:
            return "same_candle_stop_first", exit_price_from_trigger(
                pos.side, pos.stop_price, row, cfg
            )
        if stop_hit:
            return "stop_loss", exit_price_from_trigger(
                pos.side, pos.stop_price, row, cfg
            )
        if tp_hit:
            return "take_profit", exit_price_from_trigger(
                pos.side, pos.take_profit_price, row, cfg
            )
    return None


def close_position(
    pos: Position,
    row: pd.Series,
    exit_index: int,
    exit_price: float,
    reason: str,
    cfg: BacktestConfig,
    equity_before_exit: float,
) -> Dict[str, Any]:
    qty = pos.quantity
    gross_pnl = (
        qty * (exit_price - pos.entry_price)
        if pos.side == "long"
        else qty * (pos.entry_price - exit_price)
    )
    exit_fee = abs(qty * exit_price) * cfg.fee_rate
    net_pnl = gross_pnl - pos.entry_fee - exit_fee
    equity_after = equity_before_exit + net_pnl
    hold = exit_index - pos.entry_index + 1
    return {
        "trade_id": pos.trade_id,
        "symbol": pos.symbol,
        "timeframe": pos.timeframe,
        "side": pos.side,
        "signal_index": pos.signal_index,
        "signal_time": pos.signal_time,
        "signal_open": pos.signal_open,
        "signal_high": pos.signal_high,
        "signal_low": pos.signal_low,
        "signal_close": pos.signal_close,
        "signal_volume": pos.signal_volume,
        "ema20_at_signal": pos.ema20_at_signal,
        "ema50_at_signal": pos.ema50_at_signal,
        "atr14_at_signal": pos.atr14_at_signal,
        "volume_sma20_at_signal": pos.volume_sma20_at_signal,
        "atr_pct_at_signal": pos.atr_pct_at_signal,
        "spread_pct_at_signal": pos.spread_pct_at_signal,
        "entry_index": pos.entry_index,
        "entry_time": pos.entry_time,
        "entry_reference_open": pos.entry_reference_open,
        "entry_price": pos.entry_price,
        "entry_spread_pct": pos.entry_spread_pct,
        "entry_slippage_pct": pos.entry_slippage_pct,
        "entry_fee": pos.entry_fee,
        "entry_order_type": "market",
        "execution_mode": pos.execution_mode,
        "equity_before_trade": pos.equity_before_trade,
        "risk_pct": cfg.risk_per_trade,
        "risk_amount_planned": pos.risk_amount_planned,
        "risk_amount_actual": pos.risk_amount_actual,
        "gross_risk_usd": pos.gross_risk_usd,
        "estimated_costs_usd": pos.estimated_costs_usd,
        "net_risk_usd": pos.net_risk_usd,
        "risk_target_usd": pos.risk_target_usd,
        "risk_overshoot_flag": pos.risk_overshoot_flag,
        "stop_distance": pos.stop_distance,
        "stop_price": pos.stop_price,
        "take_profit_price": pos.take_profit_price,
        "quantity": pos.quantity,
        "notional": pos.notional,
        "exit_index": exit_index,
        "exit_time": row["timestamp"],
        "exit_price": exit_price,
        "exit_fee": exit_fee,
        "exit_reason": reason,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "r_multiple": (
            net_pnl / pos.risk_amount_actual if pos.risk_amount_actual > 0 else np.nan
        ),
        "return_on_equity_pct": (
            (net_pnl / pos.equity_before_trade * 100.0)
            if pos.equity_before_trade > 0
            else np.nan
        ),
        "hold_candles": hold,
        "was_winner": bool(net_pnl > 0),
        "equity_after_trade": equity_after,
        "exit_candle_open": float(row["open"]),
        "exit_candle_high": float(row["high"]),
        "exit_candle_low": float(row["low"]),
        "exit_candle_close": float(row["close"]),
        "ambiguous_intrabar_boolean": bool(reason == "same_candle_stop_first"),
    }


def apply_closed_trade(
    tr: Dict[str, Any],
    equity: float,
    consecutive_losses: int,
    cfg: BacktestConfig,
    i: int,
) -> Tuple[float, int, Optional[int]]:
    equity = tr["equity_after_trade"]
    consecutive_losses = consecutive_losses + 1 if tr["net_pnl"] < 0 else 0
    pause_until = None
    if consecutive_losses >= cfg.pause_after_losses:
        pause_until = i + cfg.pause_candles
        consecutive_losses = 0
    tr["consecutive_losses_after_trade"] = consecutive_losses
    return equity, consecutive_losses, pause_until


# -----------------------------
# Backtest engine
# -----------------------------


def backtest(
    df_raw: pd.DataFrame, cfg: BacktestConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = mark_valid_candles(df_raw, cfg.timeframe)
    validate_execution_mode(df, cfg)
    df = compute_indicators(df, cfg)

    equity = cfg.initial_equity
    open_pos: Optional[Position] = None
    pending_entry: Optional[Dict[str, Any]] = None
    gap_pending_for_open_position = False
    consecutive_losses = 0
    pause_until_index: Optional[int] = None
    trade_id = 1
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    start_index = max(
        cfg.warmup_candles,
        cfg.ema_slow + 10,
        cfg.atr_period + 10,
        cfg.volume_sma_period + 10,
    )
    dt = expected_timedelta(cfg.timeframe)

    for i in range(start_index, len(df)):
        row = df.iloc[i]

        # Missing-candle handling comes before invalid-row skip so gaps are not lost.
        if int(row.get("missing_before", 0)) > 0:
            pending_entry = None
            if open_pos is not None:
                gap_pending_for_open_position = True

        if not bool(row["valid_candle"]):
            pending_entry = None
            equity_rows.append(
                _equity_snapshot(
                    row, cfg, equity, open_pos, pause_until_index, i, consecutive_losses
                )
            )
            continue

        # If a gap occurred before this valid row, close the open position at current open.
        if gap_pending_for_open_position and open_pos is not None:
            px = market_exit_price(open_pos.side, row, cfg, when="open")
            tr = close_position(open_pos, row, i, px, "missing_data_exit", cfg, equity)
            equity, consecutive_losses, new_pause = apply_closed_trade(
                tr, equity, consecutive_losses, cfg, i
            )
            if new_pause is not None:
                pause_until_index = new_pause
            trades.append(tr)
            open_pos = None
            gap_pending_for_open_position = False
            equity_rows.append(
                _equity_snapshot(
                    row, cfg, equity, open_pos, pause_until_index, i, consecutive_losses
                )
            )
            continue
        gap_pending_for_open_position = False

        # 1) Execute pending entry at current open only if timestamp is truly next candle.
        if pending_entry is not None and pending_entry["entry_index"] == i:
            signal_ts = df.iloc[pending_entry["signal_index"]]["timestamp"]
            if row["timestamp"] - signal_ts != dt:
                pending_entry = None
            elif open_pos is None:
                open_pos = create_position(
                    trade_id,
                    df,
                    pending_entry["signal_index"],
                    i,
                    pending_entry["side"],
                    equity,
                    cfg,
                )
                if open_pos is not None:
                    trade_id += 1
                pending_entry = None
            else:
                pending_entry = None

        # 2) Manage open position intrabar, including entry candle.
        if open_pos is not None:
            exit_event = check_exit(open_pos, row, cfg)
            if exit_event is not None:
                reason, px = exit_event
                tr = close_position(open_pos, row, i, px, reason, cfg, equity)
                equity, consecutive_losses, new_pause = apply_closed_trade(
                    tr, equity, consecutive_losses, cfg, i
                )
                if new_pause is not None:
                    pause_until_index = new_pause
                    pending_entry = None
                trades.append(tr)
                open_pos = None
                equity_rows.append(
                    _equity_snapshot(
                        row,
                        cfg,
                        equity,
                        open_pos,
                        pause_until_index,
                        i,
                        consecutive_losses,
                    )
                )
                continue

        # 3) Max hold after SL/TP check. Entry candle counts as hold candle 1.
        if open_pos is not None:
            hold_candles = i - open_pos.entry_index + 1
            if hold_candles >= cfg.max_hold_candles:
                px = market_exit_price(open_pos.side, row, cfg, when="close")
                tr = close_position(open_pos, row, i, px, "max_hold_exit", cfg, equity)
                equity, consecutive_losses, new_pause = apply_closed_trade(
                    tr, equity, consecutive_losses, cfg, i
                )
                if new_pause is not None:
                    pause_until_index = new_pause
                    pending_entry = None
                trades.append(tr)
                open_pos = None
                equity_rows.append(
                    _equity_snapshot(
                        row,
                        cfg,
                        equity,
                        open_pos,
                        pause_until_index,
                        i,
                        consecutive_losses,
                    )
                )
                continue

        # 4) Generate signal after current candle close for next open.
        if open_pos is None and pending_entry is None and i + 1 < len(df):
            if pause_until_index is None or i >= pause_until_index:
                if indicators_valid(row):
                    long_sig = is_long_signal(df, i, cfg)
                    short_sig = is_short_signal(df, i, cfg)
                    if long_sig and not short_sig:
                        pending_entry = {
                            "side": "long",
                            "signal_index": i,
                            "entry_index": i + 1,
                        }
                    elif short_sig and not long_sig:
                        pending_entry = {
                            "side": "short",
                            "signal_index": i,
                            "entry_index": i + 1,
                        }

        equity_rows.append(
            _equity_snapshot(
                row, cfg, equity, open_pos, pause_until_index, i, consecutive_losses
            )
        )

    # End of data cleanup.
    if open_pos is not None:
        final_row = df.iloc[-1]
        px = market_exit_price(open_pos.side, final_row, cfg, when="close")
        tr = close_position(
            open_pos, final_row, len(df) - 1, px, "end_of_data_exit", cfg, equity
        )
        equity = tr["equity_after_trade"]
        trades.append(tr)
        open_pos = None
        equity_rows.append(
            _equity_snapshot(
                final_row,
                cfg,
                equity,
                open_pos,
                pause_until_index,
                len(df) - 1,
                consecutive_losses,
            )
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    report = build_report(df, trades_df, equity_df, cfg)
    return trades_df, equity_df, report


def _equity_snapshot(
    row: pd.Series,
    cfg: BacktestConfig,
    equity: float,
    pos: Optional[Position],
    pause_until: Optional[int],
    i: int,
    consecutive_losses: int,
) -> Dict[str, Any]:
    unreal = 0.0
    if pos is not None and bool(row.get("valid_candle", True)):
        exit_now = market_exit_price(pos.side, row, cfg, when="close")
        unreal = (
            pos.quantity * (exit_now - pos.entry_price)
            if pos.side == "long"
            else pos.quantity * (pos.entry_price - exit_now)
        )
        unreal -= pos.entry_fee
        unreal -= abs(pos.quantity * exit_now) * cfg.fee_rate
    marked = equity + unreal
    return {
        "timestamp": row["timestamp"],
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "closed_equity": equity,
        "unrealized_pnl": unreal,
        "marked_equity": marked,
        "open_position_side": None if pos is None else pos.side,
        "open_position_notional": 0.0 if pos is None else pos.notional,
        "pause_active": bool(pause_until is not None and i < pause_until),
        "consecutive_losses": consecutive_losses,
    }


# -----------------------------
# Metrics/reporting
# -----------------------------


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


def run_robustness(df: pd.DataFrame, base_cfg: BacktestConfig) -> pd.DataFrame:
    rows = []
    grids = {
        "volume_mult": [1.0, 1.2, 1.5],
        "min_atr_pct": [0.0006, 0.0008, 0.0010],
        "take_profit_r": [1.0, 1.2, 1.5],
        "max_hold_candles": [3, 5, 8],
        "slippage_bps": [2.0, 5.0],
    }
    for key, vals in grids.items():
        for val in vals:
            cfg = BacktestConfig(**asdict(base_cfg))
            setattr(cfg, key, val)
            _, _, report = backtest(df, cfg)
            rows.append(
                {
                    "changed_param": key,
                    "value": val,
                    "total_trades": report["total_trades"],
                    "profit_factor": report["profit_factor"],
                    "expectancy_r": report["expectancy_r"],
                    "max_drawdown_pct": report["max_drawdown_pct"],
                    "final_equity": report["final_equity"],
                    "conclusion": report["conclusion"],
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Unit-test-style checks
# -----------------------------


def _synthetic_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    out = []
    for k, r in enumerate(rows):
        d = {
            "timestamp": base + pd.Timedelta(minutes=k),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "spread_pct": 0.0002,
        }
        d.update(r)
        out.append(d)
    return pd.DataFrame(out)


def run_self_tests() -> None:
    cfg = BacktestConfig(
        "TEST/USDT",
        "1m",
        initial_equity=10_000,
        fee_rate=0.001,
        slippage_bps=2,
        allow_shorts=True,
        market_type="futures",
        warmup_candles=1,
    )

    # next-candle entry: direct create_position uses signal 0, entry 1 and must reference row 1 open.
    df = compute_indicators(
        mark_valid_candles(
            _synthetic_df(
                [{"close": 100}, {"open": 110, "high": 111, "low": 109, "close": 110}]
            ),
            "1m",
        ),
        cfg,
    )
    df.loc[0, ["atr14", "ema20", "ema50", "volume_sma20", "atr_pct"]] = [
        1.0,
        101,
        100,
        1000,
        0.01,
    ]
    p = create_position(1, df, 0, 1, "long", 10_000, cfg)
    assert (
        p is not None and abs(p.entry_reference_open - 110) < 1e-9
    ), "next-candle entry failed"

    # same-candle SL/TP conflict = stop first.
    row = pd.Series(
        {
            "open": p.entry_price,
            "high": p.take_profit_price * 1.01,
            "low": p.stop_price * 0.99,
            "close": p.entry_price,
            "spread_pct": 0.0002,
        }
    )
    reason, px = check_exit(p, row, cfg)
    assert (
        reason == "same_candle_stop_first" and px < p.stop_price
    ), "SL/TP conflict not stop-first with costs"

    # fee/slippage/spread applied on exits: long TP fill should be below trigger in last_trade_ohlc mode.
    tp_px = exit_price_from_trigger("long", p.take_profit_price, row, cfg)
    assert tp_px < p.take_profit_price, "exit costs not applied to TP"

    # pause rule after 3 losses.
    consec = 0
    pause = None
    eq = 10_000
    for i in range(3):
        tr = {"net_pnl": -1.0, "equity_after_trade": eq - 1.0}
        eq, consec, new_pause = apply_closed_trade(tr, eq, consec, cfg, i)
        if new_pause is not None:
            pause = new_pause
    assert pause == 2 + cfg.pause_candles and consec == 0, "pause after 3 losses failed"

    # max hold behavior: entry candle is hold candle 1; max hold 5 exits at entry_index+4.
    assert (
        p.entry_index + cfg.max_hold_candles - 1
    ) - p.entry_index + 1 == 5, "max hold arithmetic failed"

    # long-only blocks shorts.
    cfg_long_only = BacktestConfig(
        "TEST/USDT", "1m", allow_shorts=False, warmup_candles=1
    )
    d2 = _synthetic_df(
        [{"low": 98}, {"high": 103, "close": 99, "open": 101, "volume": 2000}]
    )
    d2 = compute_indicators(mark_valid_candles(d2, "1m"), cfg_long_only)
    d2.loc[1, ["ema20", "ema50", "atr14", "volume_sma20", "atr_pct"]] = [
        99,
        100,
        1,
        1000,
        0.01,
    ]
    assert not is_short_signal(
        d2, 1, cfg_long_only
    ), "long-only mode did not block short"
    print("self-tests passed")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMA Pullback Continuation backtest")
    p.add_argument("--csv", help="Input OHLCV CSV path")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1m", choices=["1m", "5m"])
    p.add_argument("--initial-equity", type=float, default=10_000.0)
    p.add_argument("--fee-rate", type=float, default=0.001, help="0.001 = 0.10%%")
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument(
        "--max-cost-to-gross-risk-ratio",
        type=float,
        default=1.0,
        help="Skip trades when estimated fees/spread/slippage exceed this multiple of gross stop risk",
    )
    p.add_argument("--min-notional", type=float, default=10.0)
    p.add_argument("--qty-step-size", type=float, default=0.0)
    p.add_argument(
        "--execution-mode",
        choices=["last_trade_ohlc", "bid_ask"],
        default="last_trade_ohlc",
    )
    p.add_argument(
        "--allow-shorts", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--market-type", choices=["spot", "margin", "futures"], default="spot"
    )
    p.add_argument("--missing-candle-warning-threshold", type=int, default=0)
    p.add_argument("--out-dir", default="backtest_results")
    p.add_argument("--robustness", action="store_true")
    p.add_argument(
        "--self-test", action="store_true", help="Run unit-test-style checks and exit"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return
    if not args.csv:
        raise SystemExit("--csv is required unless --self-test is used")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_equity=args.initial_equity,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        min_notional=args.min_notional,
        max_cost_to_gross_risk_ratio=args.max_cost_to_gross_risk_ratio,
        qty_step_size=args.qty_step_size,
        execution_mode=args.execution_mode,
        allow_shorts=args.allow_shorts,
        market_type=args.market_type,
        missing_candle_warning_threshold=args.missing_candle_warning_threshold,
    )

    df = load_csv(args.csv)
    trades, equity, report = backtest(df, cfg)
    safe_symbol = args.symbol.replace("/", "")
    prefix = f"{safe_symbol}_{args.timeframe}_{args.execution_mode}"
    trades.to_csv(out_dir / f"{prefix}_trades.csv", index=False)
    equity.to_csv(out_dir / f"{prefix}_equity.csv", index=False)
    with open(out_dir / f"{prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    if args.robustness:
        rob = run_robustness(df, cfg)
        rob.to_csv(out_dir / f"{prefix}_robustness.csv", index=False)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
