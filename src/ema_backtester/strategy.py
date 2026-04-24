from __future__ import annotations

import pandas as pd

from .data import BacktestConfig, spread_pct


def indicators_valid(row: pd.Series) -> bool:
    return all(
        pd.notna(row[c]) for c in ["ema20", "ema50", "atr14", "volume_sma20", "atr_pct"]
    )


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
