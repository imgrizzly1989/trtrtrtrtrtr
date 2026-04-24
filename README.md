# EMA Pullback Continuation Backtester

Conservative Python backtester for a BTC/ETH EMA pullback continuation scalping strategy.

This is research/backtesting code only. Do not use it for live trading.

## Strategy

- Markets: BTC/USDT, ETH/USDT
- Timeframes: 1m and 5m, tested separately
- Trend:
  - Long: EMA20 > EMA50
  - Short: EMA20 < EMA50
- Entry:
  - Pullback to EMA20
  - Confirmation close in trend direction
  - Entry at next candle open only
- Stop loss: 1 x ATR(14)
- Take profit: 1.2R
- Max hold: 5 candles
- Pause: after 3 net losing trades, pause new entries for 30 candles

## Risk model

Position sizing targets strict modeled net risk, not gross ATR risk.

The estimated stop-loss loss includes:

- gross stop distance
- entry fee
- estimated exit fee
- entry slippage
- stop-exit slippage
- spread cost in `last_trade_ohlc` mode

Default risk:

- 0.5% of current equity per trade

Important: real loss can still exceed 0.5% during gaps, missing data, spread widening, slippage worse than configured, exchange outages, or live execution failures.

## Install

```bash
pip install -r requirements.txt
```

## Run self-tests

```bash
python ema_pullback_backtest.py --self-test
```

Expected:

```text
self-tests passed
```

## CSV format

Required columns:

```csv
timestamp,open,high,low,close,volume
```

Optional for `last_trade_ohlc` mode:

```csv
spread_pct,quote_volume
```

Required for `bid_ask` mode:

```csv
bid_open,bid_high,bid_low,bid_close,ask_open,ask_high,ask_low,ask_close
```

Put your CSVs in `data/`, for example:

```text
data/BTCUSDT_1m.csv
data/BTCUSDT_5m.csv
data/ETHUSDT_1m.csv
data/ETHUSDT_5m.csv
```

CSV files are gitignored so large market data does not get committed.

## Download market data

You do not manually create the CSV. Use the downloader.

Download BTC 1m data for Jan-Mar 2024:

```bash
python download_binance_data.py --symbol BTCUSDT --interval 1m --start 2024-01 --end 2024-03
```

This creates:

```text
data/BTCUSDT_1m.csv
```

Download all four files, BTC/ETH 1m/5m:

```bash
python download_binance_data.py --all --start 2024-01 --end 2024-03
```

This creates:

```text
data/BTCUSDT_1m.csv
data/BTCUSDT_5m.csv
data/ETHUSDT_1m.csv
data/ETHUSDT_5m.csv
```

Use more months for a more meaningful backtest. Example full year:

```bash
python download_binance_data.py --all --start 2024-01 --end 2024-12
```

## Example commands

BTC 1m:

```bash
python ema_pullback_backtest.py \
  --csv data/BTCUSDT_1m.csv \
  --symbol BTC/USDT \
  --timeframe 1m \
  --execution-mode last_trade_ohlc \
  --slippage-bps 2 \
  --fee-rate 0.001 \
  --out-dir results \
  --robustness
```

BTC 5m:

```bash
python ema_pullback_backtest.py \
  --csv data/BTCUSDT_5m.csv \
  --symbol BTC/USDT \
  --timeframe 5m \
  --execution-mode last_trade_ohlc \
  --slippage-bps 2 \
  --fee-rate 0.001 \
  --out-dir results \
  --robustness
```

ETH 1m:

```bash
python ema_pullback_backtest.py \
  --csv data/ETHUSDT_1m.csv \
  --symbol ETH/USDT \
  --timeframe 1m \
  --execution-mode last_trade_ohlc \
  --slippage-bps 2 \
  --fee-rate 0.001 \
  --out-dir results \
  --robustness
```

ETH 5m:

```bash
python ema_pullback_backtest.py \
  --csv data/ETHUSDT_5m.csv \
  --symbol ETH/USDT \
  --timeframe 5m \
  --execution-mode last_trade_ohlc \
  --slippage-bps 2 \
  --fee-rate 0.001 \
  --out-dir results \
  --robustness
```

## Outputs

The script writes:

- `*_trades.csv` full trade log
- `*_equity.csv` equity curve
- `*_report.json` summary report
- `*_robustness.csv` if `--robustness` is used

## Execution modes

### last_trade_ohlc

Uses standard OHLCV candles. If `spread_pct` exists, it is used. Otherwise the script uses `default_spread_pct`.

This mode approximates execution and is less precise than quote/order-book data.

### bid_ask

Requires bid/ask OHLC columns and avoids synthetic spread double-counting.

Longs buy at ask and sell at bid.
Shorts sell at bid and buy at ask.

## Safety notes

This code is not safe for live trading.

Missing live-trading requirements include:

- broker/order state reconciliation
- partial fills
- latency and order queue modeling
- funding/borrow/liquidation logic
- exchange API error handling
- kill switch
- production monitoring
