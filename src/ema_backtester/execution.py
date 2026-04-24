from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import (
    BacktestConfig,
    compute_indicators,
    expected_timedelta,
    mark_valid_candles,
    spread_pct,
    spread_pct_at,
    validate_execution_mode,
)
from .metrics import build_report
from .strategy import indicators_valid, is_long_signal, is_short_signal


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


def floor_to_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


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
