"""Write CSV tables and a text run summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.heap_filter import AnomalyEvent


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_run_summary(
    path: Path,
    tickers: list[str],
    n_obs: int,
    date_start: str,
    date_end: str,
    n_anomalies: int,
    top_events: list[AnomalyEvent],
    sweep: pd.DataFrame,
    recall_diff: dict[str, float],
    heap_bench: dict[str, float],
    window: int,
    threshold: float,
    data_source: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _best_row(method: str) -> pd.Series:
        subset = sweep[(sweep["method"] == method) & (sweep["window"] == window) & (sweep["threshold"] == threshold)]
        if subset.empty:
            subset = sweep[sweep["method"] == method]
        return subset.iloc[0]

    z = _best_row("zscore")
    m = _best_row("mad")

    lines = [
        "STOCK MARKET ANOMALY DETECTOR",
        "=============================",
        f"Data source: {data_source}",
        f"Tickers analyzed: {len(tickers)}",
        f"Market observations: {n_obs:,}",
        f"Date range: {date_start} — {date_end}",
        f"Detector window: {window}  threshold: {threshold}",
        f"Anomalies detected (composite Z-score >= {threshold}): {n_anomalies:,}",
        "",
        "Top anomalies:",
    ]
    for i, event in enumerate(top_events[:10], start=1):
        date_s = pd.Timestamp(event.date).date()
        lines.append(f"{i}. {event.ticker:5}  {date_s}  Score: {event.score:.2f}")

    lines.extend(
        [
            "",
            "Benchmark Results (synthetic injection, matched window/threshold)",
            "Rolling Z-Score:",
            f"  Precision: {z['precision']:.2f}",
            f"  Recall:    {z['recall']:.2f}",
            f"  F1:        {z['f1']:.2f}",
            f"  Runtime:   {z['runtime_seconds']:.3f} s",
            "Rolling MAD:",
            f"  Precision: {m['precision']:.2f}",
            f"  Recall:    {m['recall']:.2f}",
            f"  F1:        {m['f1']:.2f}",
            f"  Runtime:   {m['runtime_seconds']:.3f} s",
            "",
            "Recall comparison at configured window/threshold:",
            f"  Z-score recall: {recall_diff['zscore_recall']:.3f}",
            f"  MAD recall:     {recall_diff['mad_recall']:.3f}",
            f"  Relative difference (MAD vs Z-score): {recall_diff['relative_difference'] * 100:.1f}%",
            "",
            "Heap vs full sort (top-K):",
            f"  n={int(heap_bench['n'])}  K={int(heap_bench['k'])}",
            f"  Heap: {heap_bench['heap_seconds']:.4f} s  peak {heap_bench['heap_peak_mb']:.2f} MB",
            f"  Sort: {heap_bench['sort_seconds']:.4f} s  peak {heap_bench['sort_peak_mb']:.2f} MB",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
