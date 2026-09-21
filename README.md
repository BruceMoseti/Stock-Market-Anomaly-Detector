# Stock Market Anomaly Detector

Python pipeline that downloads daily market data, engineers return/volume/volatility features, and flags unusual moves with two sliding-window detectors. A bounded min-heap keeps the top-K scores without sorting the full panel.

## What it does

1. Loads 50+ tickers of daily OHLCV (Yahoo Finance, with a synthetic fallback).
2. Cleans missing/invalid prices.
3. Builds daily returns, log returns, rolling volatility, volume ratio, and overnight gaps.
4. Scores each ticker-day with a **rolling Z-score** and a **rolling MAD** (robust) detector. Statistics use the previous N days so the current bar is not in its own window.
5. Combines price, volume, and volatility scores:
   `0.60 * |return_z| + 0.25 * |volume_z| + 0.15 * |volatility_z|`
6. Keeps the strongest events in **O(n log K)** with a min-heap of size K.
7. Benchmarks detectors on **injected synthetic anomalies** (precision, recall, F1, latency, runtime).
8. Sweeps window sizes `{5, 10, 20, 30, 60, 90}` and thresholds `{2.0, 2.5, 3.0, 3.5}`.
9. Writes charts plus `reports/summaries/run_summary.txt`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --config config.yaml
```

Yahoo Finance is the default source. If the download fails, the CLI generates a matching synthetic panel so the rest of the pipeline still runs.

Offline / deterministic demo:

```bash
python main.py --synthetic --config config.yaml
```

Subset of tickers:

```bash
python main.py --tickers AAPL MSFT NVDA TSLA --start 2015-01-01 --window 30 --threshold 3.0 --top-k 100
```

## Tests

```bash
pytest -q
```

## Layout

```
config.yaml
main.py
src/
  data_loader.py
  preprocessing.py
  features.py
  detectors.py
  heap_filter.py
  benchmark.py
  visualization.py
  reporting.py
tests/
reports/
  figures/
  summaries/
```

## Interpreting results

Shorter windows react faster to regime changes and usually raise recall at the cost of more false positives. Longer windows are stabler and miss some short spikes. MAD is less distorted by fat tails than mean/std Z-scores; the summary file records the measured relative recall difference at the configured window and threshold — use that number, do not assume 30%.

Top-K ranking uses a bounded min-heap: only K entries stay in memory, so selecting the 100 strongest scores from ~100k+ rows is `O(n log K)` rather than a full `O(n log n)` sort.
