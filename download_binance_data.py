#!/usr/bin/env python3
"""
Download Binance public historical kline data and build CSVs for ema_pullback_backtest.py.

Examples:
    python download_binance_data.py --symbol BTCUSDT --interval 1m --start 2024-01 --end 2024-03
    python download_binance_data.py --all --start 2024-01 --end 2024-03

Creates files like:
    data/BTCUSDT_1m.csv
    data/BTCUSDT_5m.csv
    data/ETHUSDT_1m.csv
    data/ETHUSDT_5m.csv
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
RAW_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
OUT_COLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "number_of_trades",
]


def parse_ym(s: str) -> tuple[int, int]:
    try:
        d = datetime.strptime(s, "%Y-%m")
        return d.year, d.month
    except ValueError as e:
        raise argparse.ArgumentTypeError("Use YYYY-MM format, e.g. 2024-01") from e


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    sy, sm = parse_ym(start)
    ey, em = parse_ym(end)
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m == 13:
            y += 1
            m = 1
    return months


def download_zip(symbol: str, interval: str, year: int, month: int) -> bytes | None:
    name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{name}"
    print(f"Downloading {url}")
    try:
        with urlopen(url, timeout=60) as r:
            return r.read()
    except HTTPError as e:
        if e.code == 404:
            print(f"  missing: {name}")
            return None
        raise
    except URLError as e:
        print(f"  network error: {e}")
        return None


def zip_to_frame(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError("zip contains no csv")
        with z.open(csv_names[0]) as f:
            df = pd.read_csv(f, header=None, names=RAW_COLS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["number_of_trades"] = (
        pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype(int)
    )
    return df[OUT_COLS]


def build_csv(symbol: str, interval: str, start: str, end: str, out_dir: Path) -> Path:
    frames = []
    for year, month in month_range(start, end):
        blob = download_zip(symbol, interval, year, month)
        if blob is None:
            continue
        frames.append(zip_to_frame(blob))
    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {interval} {start}..{end}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{interval}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows -> {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Binance monthly kline CSVs")
    p.add_argument(
        "--symbol", choices=["BTCUSDT", "ETHUSDT"], help="Symbol to download"
    )
    p.add_argument("--interval", choices=["1m", "5m"], help="Interval to download")
    p.add_argument("--start", required=True, help="Start month YYYY-MM")
    p.add_argument("--end", required=True, help="End month YYYY-MM, inclusive")
    p.add_argument("--out-dir", default="data")
    p.add_argument(
        "--all", action="store_true", help="Download BTCUSDT/ETHUSDT for 1m/5m"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.all:
        jobs = [(s, i) for s in ["BTCUSDT", "ETHUSDT"] for i in ["1m", "5m"]]
    else:
        if not args.symbol or not args.interval:
            raise SystemExit("Use --symbol and --interval, or use --all")
        jobs = [(args.symbol, args.interval)]

    for symbol, interval in jobs:
        try:
            build_csv(symbol, interval, args.start, args.end, out_dir)
        except Exception as e:
            print(f"FAILED {symbol} {interval}: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
