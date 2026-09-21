"""Bounded min-heap for retaining the K highest-scoring anomalies."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class AnomalyEvent:
    ticker: str
    date: pd.Timestamp
    score: float
    return_: float | None = None
    volume_ratio: float | None = None
    method: str = ""


@dataclass(order=True)
class _HeapItem:
    score: float
    seq: int
    event: AnomalyEvent = field(compare=False)


def top_k_anomalies(events: Iterable[AnomalyEvent], k: int) -> list[AnomalyEvent]:
    """Return the K highest-scoring events using a bounded min-heap.

    Complexity is O(n log K) rather than O(n log n) full sort.
    """
    if k <= 0:
        return []

    heap: list[_HeapItem] = []
    for seq, event in enumerate(events):
        item = _HeapItem(score=float(event.score), seq=seq, event=event)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item.score > heap[0].score:
            heapq.heapreplace(heap, item)

    return sorted((item.event for item in heap), key=lambda e: e.score, reverse=True)


def events_from_frame(
    df: pd.DataFrame,
    score_column: str,
    method: str,
    min_score: float = 0.0,
) -> list[AnomalyEvent]:
    rows = df.dropna(subset=[score_column])
    rows = rows.loc[rows[score_column].abs() >= min_score]
    work = rows.rename(columns={"return": "ret"})
    events: list[AnomalyEvent] = []
    for row in work.itertuples(index=False):
        ret = getattr(row, "ret", None)
        vol_ratio = getattr(row, "volume_ratio", None)
        events.append(
            AnomalyEvent(
                ticker=str(row.ticker),
                date=pd.Timestamp(row.date),
                score=float(getattr(row, score_column)),
                return_=None if ret is None or pd.isna(ret) else float(ret),
                volume_ratio=None if vol_ratio is None or pd.isna(vol_ratio) else float(vol_ratio),
                method=method,
            )
        )
    return events


def events_to_frame(events: Sequence[AnomalyEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=["rank", "ticker", "date", "score", "return", "volume_ratio", "method"]
        )
    return pd.DataFrame(
        [
            {
                "rank": i + 1,
                "ticker": e.ticker,
                "date": e.date,
                "score": e.score,
                "return": e.return_,
                "volume_ratio": e.volume_ratio,
                "method": e.method,
            }
            for i, e in enumerate(events)
        ]
    )
