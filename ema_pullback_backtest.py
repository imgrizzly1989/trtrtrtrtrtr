#!/usr/bin/env python3
"""CLI wrapper for the EMA Pullback Continuation crypto scalping backtest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ema_backtester.data import BacktestConfig, load_csv
from ema_backtester.execution import backtest, run_robustness, run_self_tests


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
