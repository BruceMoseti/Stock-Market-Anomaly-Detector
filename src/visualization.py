"""Charts for prices, scores, and benchmark comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_price_with_anomalies(df: pd.DataFrame, ticker: str, score_col: str, threshold: float, path: Path) -> Path:
    g = df[df["ticker"] == ticker].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g["date"], g["close"], color="steelblue", linewidth=1, label="Close")
    hits = g[g[score_col].abs() >= threshold]
    ax.scatter(hits["date"], hits["close"], color="crimson", s=18, zorder=3, label="Anomaly")
    ax.set_title(f"{ticker} close price with anomaly markers")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_return_distribution(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["return"].dropna(), bins=80, color="slategray", edgecolor="white")
    ax.set_title("Daily return distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    return _save(fig, path)


def plot_rolling_volatility(df: pd.DataFrame, ticker: str, path: Path) -> Path:
    g = df[df["ticker"] == ticker].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g["date"], g["volatility"], color="darkorange", linewidth=1)
    ax.set_title(f"{ticker} rolling volatility")
    ax.set_ylabel("Std. of returns")
    return _save(fig, path)


def plot_volume_anomalies(df: pd.DataFrame, ticker: str, score_col: str, threshold: float, path: Path) -> Path:
    g = df[df["ticker"] == ticker].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(g["date"], g["volume"], color="lightsteelblue", width=1.5)
    hits = g[g[score_col].abs() >= threshold]
    ax.scatter(hits["date"], hits["volume"], color="crimson", s=18, zorder=3, label="Anomaly")
    ax.set_title(f"{ticker} trading volume")
    ax.set_ylabel("Volume")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_score_over_time(df: pd.DataFrame, ticker: str, score_col: str, title: str, path: Path) -> Path:
    g = df[df["ticker"] == ticker].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g["date"], g[score_col], color="purple", linewidth=0.8)
    ax.axhline(3.0, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-3.0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Score")
    return _save(fig, path)


def plot_metric_vs_threshold(sweep: pd.DataFrame, metric: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, g in sweep.groupby("algorithm"):
        agg = g.groupby("threshold")[metric].mean()
        ax.plot(agg.index, agg.values, marker="o", label=method)
    ax.set_title(f"{metric.capitalize()} vs threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    return _save(fig, path)


def plot_f1_vs_window(sweep: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, g in sweep.groupby("algorithm"):
        agg = g.groupby("window")["f1"].mean()
        ax.plot(agg.index, agg.values, marker="o", label=method)
    ax.set_title("F1 vs window size")
    ax.set_xlabel("Window (trading days)")
    ax.set_ylabel("F1")
    ax.legend()
    return _save(fig, path)


def plot_runtime(sweep: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    agg = sweep.groupby("algorithm")["runtime_seconds"].mean()
    ax.bar(agg.index, agg.values, color=["steelblue", "darkorange"])
    ax.set_title("Detector runtime")
    ax.set_ylabel("Seconds")
    return _save(fig, path)


def plot_anomalies_by_ticker(df: pd.DataFrame, score_col: str, threshold: float, path: Path) -> Path:
    flagged = df[df[score_col].abs() >= threshold]
    counts = flagged.groupby("ticker").size().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index.astype(str), counts.values, color="teal")
    ax.set_title("Anomalies by ticker (top 20)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def plot_algorithm_comparison(sweep: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    agg = sweep.groupby("algorithm")[["precision", "recall", "f1"]].mean()
    agg.plot(kind="bar", ax=ax)
    ax.set_title("Algorithm comparison (mean over sweep)")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    return _save(fig, path)


def generate_all_figures(
    featured: pd.DataFrame,
    sweep: pd.DataFrame,
    figures_dir: Path,
    example_tickers: list[str],
    threshold: float,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    available = set(featured["ticker"].unique())
    tickers = [t for t in example_tickers if t in available] or list(available)[:3]
    primary = tickers[0]

    score_col = "z_score" if "z_score" in featured.columns else "z_return"
    mad_col = "mad_return" if "mad_return" in featured.columns else score_col

    paths.append(plot_price_with_anomalies(featured, primary, score_col, threshold, figures_dir / f"{primary}_anomalies.png"))
    for t in tickers[1:5]:
        paths.append(plot_price_with_anomalies(featured, t, score_col, threshold, figures_dir / f"{t}_anomalies.png"))
    paths.append(plot_return_distribution(featured, figures_dir / "return_distribution.png"))
    paths.append(plot_rolling_volatility(featured, primary, figures_dir / f"{primary}_volatility.png"))
    paths.append(plot_volume_anomalies(featured, primary, score_col, threshold, figures_dir / f"{primary}_volume.png"))
    paths.append(plot_score_over_time(featured, primary, "z_return", f"{primary} rolling Z-score", figures_dir / f"{primary}_zscore.png"))
    paths.append(plot_score_over_time(featured, primary, mad_col, f"{primary} rolling MAD score", figures_dir / f"{primary}_mad.png"))
    paths.append(plot_metric_vs_threshold(sweep, "precision", figures_dir / "precision_vs_threshold.png"))
    paths.append(plot_metric_vs_threshold(sweep, "recall", figures_dir / "recall_vs_threshold.png"))
    paths.append(plot_f1_vs_window(sweep, figures_dir / "f1_vs_window.png"))
    paths.append(plot_runtime(sweep, figures_dir / "detector_runtime.png"))
    paths.append(plot_anomalies_by_ticker(featured, score_col, threshold, figures_dir / "anomalies_by_ticker.png"))
    paths.append(plot_algorithm_comparison(sweep, figures_dir / "algorithm_comparison.png"))
    return paths
