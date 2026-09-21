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

Shorter windows react faster to regime changes and usually raise recall at the cost of more false positives. Longer windows are stabler and miss some short spikes. MAD is less distorted by fat tails than mean/std Z-scores; `reports/summaries/run_summary.txt` records the measured relative recall difference — use that number, do not assume 30%.

Top-K ranking uses a bounded min-heap (`O(n log K)`, `O(K)` extra memory) instead of sorting every score (`O(n log n)`). At this panel size CPython's Timsort can still win on wall-clock time; the heap's advantage is the bounded working set.

## Sample synthetic run

`python main.py --synthetic` on the default 55 tickers (2015-01-01 to 2025-08-01):

- 151,910 ticker-day observations
- 126 composite Z-score anomalies at window=30, threshold=3.0
- Injected-anomaly benchmark at the same settings: Z-score precision 1.00 / recall 0.67 / F1 0.80; MAD precision 0.90 / recall 1.00 / F1 0.95
- Relative recall difference (MAD vs Z-score): **50.4%**
- Top-K heap vs full sort on 150,205 scores (K=100): heap 0.003 s / ~0 MB extra vs sort 0.019 s / 1.7 MB
- Charts under `reports/figures/` (price markers, return histogram, volatility, volume, Z/MAD paths, precision/recall vs threshold, F1 vs window, runtime, counts by ticker, algorithm comparison)
