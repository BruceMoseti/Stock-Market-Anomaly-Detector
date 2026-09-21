"""Detector and heap tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors import add_detector_scores, rolling_mad_zscore, rolling_zscore
from src.features import add_features
from src.heap_filter import AnomalyEvent, top_k_anomalies
from src.preprocessing import clean_prices


def _series_with_spike(n: int = 80, spike_at: int = 50, spike: float = 0.12) -> pd.Series:
    data = np.full(n, 0.001)
    data[spike_at] = spike
    return pd.Series(data)


def test_normal_data_not_flagged():
    s = pd.Series(np.full(80, 0.001))
    z = rolling_zscore(s, window=20)
    assert z.iloc[40:80].abs().max() < 2.0


def test_price_spike_detected():
    s = _series_with_spike(spike=0.15)
    z = rolling_zscore(s, window=20)
    assert abs(z.iloc[50]) >= 3.0


def test_negative_crash_detected():
    s = _series_with_spike(spike=-0.18)
    z = rolling_zscore(s, window=20)
    assert z.iloc[50] <= -3.0


def test_mad_detects_spike_when_history_is_noisy():
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.01, size=120)
    data[40] = 0.08
    data[90] = 0.20
    s = pd.Series(data)
    mad_z = rolling_mad_zscore(s, window=30)
    assert abs(mad_z.iloc[90]) >= 3.0


def test_heap_never_exceeds_k():
    events = [
        AnomalyEvent(ticker="T", date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), score=float(i))
        for i in range(500)
    ]
    top = top_k_anomalies(events, k=10)
    assert len(top) == 10


def test_heap_contains_highest_scores():
    scores = [1.0, 9.0, 3.0, 8.5, 2.0, 7.0, 0.5, 10.0]
    events = [
        AnomalyEvent(ticker="T", date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), score=s)
        for i, s in enumerate(scores)
    ]
    top = top_k_anomalies(events, k=3)
    assert [e.score for e in top] == [10.0, 9.0, 8.5]


def test_composite_score_ranks_joint_price_volume_shock():
    n = 60
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.cumprod(np.r_[100.0, np.full(n - 1, 1.001)])
    volume = np.full(n, 1_000_000.0)
    close[50] = close[49] * 0.90
    volume[50] = 5_000_000.0
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "TEST",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
        }
    )
    featured = add_features(df)
    scored = add_detector_scores(featured, window=20, method="zscore")
    assert scored.loc[50, "z_score"] == scored["z_score"].max()
