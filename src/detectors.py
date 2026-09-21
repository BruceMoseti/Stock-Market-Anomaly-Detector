"""Sliding-window Z-score and rolling MAD detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_CONSTANT = 0.6745

DEFAULT_WEIGHTS = {"return": 0.60, "volume": 0.25, "volatility": 0.15}


def _rolling_mean_std(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=window).mean()
    std = prior.rolling(window, min_periods=window).std(ddof=1)
    return mean, std


def _scale_against_center(series: pd.Series, center: pd.Series, scale: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Divide by scale; a zero scale is 0 if x equals the center, otherwise a huge score."""
    x = series.to_numpy(dtype=float)
    c = center.to_numpy(dtype=float)
    s = scale.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.isfinite(c) & np.isfinite(s)
    nonzero = valid & (s > 0)
    out[nonzero] = multiplier * (x[nonzero] - c[nonzero]) / s[nonzero]
    zero = valid & (s == 0)
    delta = x[zero] - c[zero]
    out[zero] = np.where(np.abs(delta) <= 1e-12, 0.0, np.sign(delta) * 1e6)
    return pd.Series(out, index=series.index)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Z-score of each point against the previous `window` observations."""
    mean, std = _rolling_mean_std(series, window)
    return _scale_against_center(series, mean, std)


def rolling_mad_zscore(series: pd.Series, window: int) -> pd.Series:
    """Robust Z-score using rolling median and MAD of the previous window."""
    prior = series.shift(1)
    median = prior.rolling(window, min_periods=window).median()

    def _mad(arr: np.ndarray) -> float:
        return float(np.median(np.abs(arr - np.median(arr))))

    mad = prior.rolling(window, min_periods=window).apply(_mad, raw=True)
    return _scale_against_center(series, median, mad, multiplier=MAD_CONSTANT)


def _grouped_score(df: pd.DataFrame, column: str, window: int, method: str) -> pd.Series:
    fn = rolling_zscore if method == "zscore" else rolling_mad_zscore
    return df.groupby("ticker", sort=False)[column].transform(lambda s: fn(s, window))


def add_detector_scores(
    df: pd.DataFrame,
    window: int = 30,
    method: str = "zscore",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach per-feature scores and a weighted composite anomaly score."""
    if method not in {"zscore", "mad"}:
        raise ValueError("method must be 'zscore' or 'mad'")
    weights = weights or DEFAULT_WEIGHTS
    out = df.copy()
    prefix = "z" if method == "zscore" else "mad"

    out[f"{prefix}_return"] = _grouped_score(out, "return", window, method)
    out[f"{prefix}_volume"] = _grouped_score(out, "volume_ratio", window, method)
    out[f"{prefix}_volatility"] = _grouped_score(out, "volatility", window, method)

    ret = out[f"{prefix}_return"].abs()
    out[f"{prefix}_score"] = (
        weights["return"] * ret
        + weights["volume"] * out[f"{prefix}_volume"].abs().fillna(0.0)
        + weights["volatility"] * out[f"{prefix}_volatility"].abs().fillna(0.0)
    )
    out.loc[ret.isna(), f"{prefix}_score"] = np.nan
    return out


def flag_anomalies(
    df: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["is_anomaly"] = out[score_column].abs() >= threshold
    return out
