"""CLI for the stock market anomaly detection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.benchmark import (
    benchmark_heap_vs_sort,
    parameter_sweep,
    relative_recall_difference,
)
from src.data_loader import default_tickers, download_prices, generate_synthetic_prices, save_prices
from src.detectors import add_detector_scores, flag_anomalies
from src.features import add_features
from src.heap_filter import events_from_frame, events_to_frame, top_k_anomalies
from src.preprocessing import clean_prices
from src.reporting import write_csv, write_run_summary
from src.visualization import generate_all_figures


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock market anomaly detector")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols (default: config list)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--window", type=int, help="Rolling window in trading days")
    parser.add_argument("--threshold", type=float, help="Anomaly score threshold")
    parser.add_argument("--top-k", type=int, dest="top_k", help="Number of top anomalies to keep")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated prices instead of Yahoo Finance",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    det_cfg = cfg["detector"]
    feat_cfg = cfg["features"]
    bench_cfg = cfg["benchmark"]
    rep_cfg = cfg["reporting"]

    tickers = args.tickers or data_cfg.get("tickers") or default_tickers()
    start = args.start or data_cfg["start_date"]
    end = args.end or data_cfg["end_date"]
    window = args.window or det_cfg["window"]
    threshold = args.threshold if args.threshold is not None else det_cfg["z_threshold"]
    top_k = args.top_k or det_cfg["top_k"]
    weights = det_cfg.get("score_weights")

    if args.synthetic:
        prices = generate_synthetic_prices(tickers, start, end)
        save_prices(prices, data_cfg["cache_path"])
        source = "synthetic"
    else:
        try:
            prices = download_prices(tickers, start, end, cache_path=data_cfg["cache_path"])
            source = "yahoo_finance"
        except Exception as exc:
            print(f"Download failed ({exc}); falling back to synthetic prices.")
            prices = generate_synthetic_prices(tickers, start, end)
            save_prices(prices, data_cfg["cache_path"])
            source = "synthetic_fallback"

    clean = clean_prices(prices)
    featured = add_features(
        clean,
        volatility_window=feat_cfg["volatility_window"],
        volume_window=feat_cfg["volume_window"],
    )
    featured = add_detector_scores(featured, window=window, method="zscore", weights=weights)
    featured = add_detector_scores(featured, window=window, method="mad", weights=weights)
    featured = flag_anomalies(featured, "z_score", threshold)

    processed_path = Path(data_cfg["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(processed_path, index=False)

    events = events_from_frame(featured, "z_score", method="zscore", min_score=threshold)
    top_events = top_k_anomalies(events, top_k)
    top_df = events_to_frame(top_events)

    sweep = parameter_sweep(
        n=int(bench_cfg["series_length"]),
        n_anomalies=int(bench_cfg["synthetic_anomalies"]),
        windows=list(bench_cfg["windows"]),
        thresholds=list(bench_cfg["thresholds"]),
        latency_days=int(bench_cfg["latency_days"]),
    )
    recall_diff = relative_recall_difference(sweep, window=window, threshold=threshold)
    heap_bench = benchmark_heap_vs_sort(featured["z_score"].dropna().to_numpy(), k=top_k)

    reports = Path("reports")
    if rep_cfg.get("save_csv", True):
        write_csv(top_df, reports / "anomaly_summary.csv")
        write_csv(sweep, reports / "benchmark_results.csv")

    if rep_cfg.get("save_plots", True) and not args.skip_plots:
        generate_all_figures(
            featured,
            sweep,
            Path(rep_cfg["figures_dir"]),
            list(rep_cfg.get("example_tickers") or tickers[:5]),
            threshold,
        )

    n_anomalies = int(featured["is_anomaly"].fillna(False).sum())
    date_start = pd.to_datetime(featured["date"]).min().date()
    date_end = pd.to_datetime(featured["date"]).max().date()
    write_run_summary(
        Path(rep_cfg["summaries_dir"]) / "run_summary.txt",
        tickers=tickers,
        n_obs=len(featured),
        date_start=str(date_start),
        date_end=str(date_end),
        n_anomalies=n_anomalies,
        top_events=top_events,
        sweep=sweep,
        recall_diff=recall_diff,
        heap_bench=heap_bench,
        window=window,
        threshold=threshold,
        data_source=source,
    )
    print(f"Wrote reports for {len(tickers)} tickers, {len(featured):,} observations ({source}).")


if __name__ == "__main__":
    run(parse_args())
