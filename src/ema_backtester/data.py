from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
