"""Synthetic-anomaly benchmarking: precision/recall/F1, latency, runtime, sweeps."""

from __future__ import annotations

import heapq
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.detectors import rolling_mad_zscore, rolling_zscore


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    mean_latency: float
    n_flagged: int


def inject_synthetic_anomalies(
    returns: np.ndarray,
    volume: np.ndarray,
    n_anomalies: int,
    seed: int = 7,
    min_gap: int = 8,
    start_index: int = 100,
) -> np.ndarray:
    """Overwrite selected days with large return and volume shocks. Returns indices."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    candidates = np.arange(start_index, n - 2)
    rng.shuffle(candidates)
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= n_anomalies:
            break
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(chosen))
    magnitudes = rng.uniform(0.06, 0.12, size=len(chosen))
    for idx, sign, mag in zip(chosen, signs, magnitudes):
        returns[idx] = sign * mag
        volume[idx] = volume[idx] * rng.uniform(4.0, 7.0)
    return np.array(sorted(chosen), dtype=int)


def _series_from_close(close: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(close, index=dates)
    return s.pct_change()


def build_synthetic_series(n: int, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    returns = rng.normal(0.0004, 0.012, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    volume = rng.lognormal(16.0, 0.25, size=n)
    return pd.Series(close, index=dates, name="close"), pd.Series(volume, index=dates, name="volume")


def classify(
    scores: pd.Series,
    true_indices: np.ndarray,
    threshold: float,
    latency_days: int = 1,
) -> ClassificationMetrics:
    flagged = scores.abs() >= threshold
    flagged_pos = set(np.flatnonzero(flagged.fillna(False).to_numpy()))
    true_set = set(int(i) for i in true_indices)

    tp = 0
    latencies: list[int] = []
    detected = set()
    for idx in true_indices:
        hit = None
        for lag in range(0, latency_days + 1):
            if int(idx) + lag in flagged_pos:
                hit = lag
                break
        if hit is not None:
            tp += 1
            latencies.append(hit)
            detected.add(int(idx))
            for lag in range(0, latency_days + 1):
                detected.add(int(idx) + lag)

    fn = len(true_set) - tp
    fp = len(flagged_pos - detected)
    valid = int(scores.notna().sum())
    tn = max(valid - tp - fp - fn, 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else float("nan")
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        mean_latency=mean_latency,
        n_flagged=int(flagged.fillna(False).sum()),
    )


def run_detector_on_returns(returns: pd.Series, window: int, method: str) -> pd.Series:
    if method == "zscore":
        return rolling_zscore(returns, window)
    if method == "mad":
        return rolling_mad_zscore(returns, window)
    raise ValueError(f"Unknown method: {method}")


def time_and_memory(fn) -> tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / (1024 * 1024)


def benchmark_heap_vs_sort(scores: np.ndarray, k: int) -> dict[str, float]:
    values = [float(s) for s in scores]

    def _heap():
        heap: list[float] = []
        for score in values:
            if len(heap) < k:
                heapq.heappush(heap, score)
            elif score > heap[0]:
                heapq.heapreplace(heap, score)
        return heap

    def _sort():
        return sorted(values, reverse=True)[:k]

    heap_t, heap_m = time_and_memory(_heap)
    sort_t, sort_m = time_and_memory(_sort)
    return {
        "heap_seconds": heap_t,
        "sort_seconds": sort_t,
        "heap_peak_mb": heap_m,
        "sort_peak_mb": sort_m,
        "k": k,
        "n": float(len(values)),
    }


def parameter_sweep(
    n: int,
    n_anomalies: int,
    windows: list[int],
    thresholds: list[float],
    latency_days: int = 1,
    seed: int = 11,
) -> pd.DataFrame:
    close, volume = build_synthetic_series(n, seed=seed)
    returns = close.pct_change().to_numpy(copy=True)
    vol = volume.to_numpy(copy=True)
    true_idx = inject_synthetic_anomalies(returns, vol, n_anomalies=n_anomalies, seed=seed)
    ret_series = pd.Series(returns, index=close.index)

    rows = []
    for method in ("zscore", "mad"):
        for window in windows:
            t0 = time.perf_counter()
            scores = run_detector_on_returns(ret_series, window, method)
            runtime = time.perf_counter() - t0
            for threshold in thresholds:
                metrics = classify(scores, true_idx, threshold, latency_days=latency_days)
                rows.append(
                    {
                        "algorithm": "Rolling Z-Score" if method == "zscore" else "Rolling MAD",
                        "method": method,
                        "window": window,
                        "threshold": threshold,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                        "true_positive": metrics.true_positive,
                        "false_positive": metrics.false_positive,
                        "true_negative": metrics.true_negative,
                        "false_negative": metrics.false_negative,
                        "mean_latency": metrics.mean_latency,
                        "n_flagged": metrics.n_flagged,
                        "runtime_seconds": runtime,
                    }
                )
    return pd.DataFrame(rows)


def relative_recall_difference(sweep: pd.DataFrame, window: int, threshold: float) -> dict[str, float]:
    subset = sweep[(sweep["window"] == window) & (sweep["threshold"] == threshold)]
    z = subset.loc[subset["method"] == "zscore", "recall"]
    m = subset.loc[subset["method"] == "mad", "recall"]
    z_rec = float(z.iloc[0]) if len(z) else float("nan")
    m_rec = float(m.iloc[0]) if len(m) else float("nan")
    if z_rec and z_rec > 0:
        rel = (m_rec - z_rec) / z_rec
    else:
        rel = float("nan")
    return {"zscore_recall": z_rec, "mad_recall": m_rec, "relative_difference": rel}
