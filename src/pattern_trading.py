"""Pattern-based trading utilities using Dynamic Time Warping (DTW).

This module provides utilities to analyse the most recent price action
against historical patterns using DTW distances and k-nearest neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - dependency availability
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    pd = None  # type: ignore[assignment]


@dataclass
class PatternMatch:
    """Details about an individual historical pattern match."""

    start_index: int
    end_index: int
    distance: float
    entry_price: float
    final_return: float
    max_run_up: float
    max_drawdown: float
    time_to_tp: Optional[int]
    time_to_sl: Optional[int]


@dataclass
class TradeGuidance:
    """Aggregated probability-weighted trade guidance."""

    entry_confidence: float
    expected_return: float
    take_profit: float
    stop_loss: float
    exit_horizon: int
    tp_hit_probability: float
    sl_hit_probability: float
    matches: List[PatternMatch]

    def to_dict(self) -> Dict[str, object]:
        """Convert the guidance to a serialisable dictionary."""

        return {
            "entry_confidence": self.entry_confidence,
            "expected_return": self.expected_return,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "exit_horizon": self.exit_horizon,
            "tp_hit_probability": self.tp_hit_probability,
            "sl_hit_probability": self.sl_hit_probability,
            "matches": [match.__dict__ for match in self.matches],
        }


def _require_pandas():
    if pd is None:  # pragma: no cover - exercised when dependency missing
        raise ModuleNotFoundError(
            "pandas is required for pattern trading utilities but is not installed"
        )
    return pd


def load_ohlc_from_excel(
    path: str,
    *,
    sheet_name: int | str = 0,
    date_column: str = "Date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load OHLC data from an Excel spreadsheet.

    Args:
        path: Path to the Excel file.
        sheet_name: Sheet to read, passed directly to :func:`pandas.read_excel`.
        date_column: Name of the column containing timestamps.
        parse_dates: Whether to parse the date column into a datetime index.

    Returns:
        DataFrame sorted by the date column with a DatetimeIndex if requested.
    """

    pandas = _require_pandas()
    df = pandas.read_excel(path, sheet_name=sheet_name)
    if parse_dates and date_column in df.columns:
        df[date_column] = pandas.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        df = df.set_index(date_column)
    else:
        df = df.reset_index(drop=True)
    return df


def _validate_window_length(window: int) -> int:
    if not 10 <= window <= 20:
        raise ValueError("Window length must be between 10 and 20 candles")
    return window


def extract_recent_window(df: pd.DataFrame, column: str = "Close", window: int = 20) -> pd.Series:
    """Extract the most recent window of closing prices for analysis."""

    _require_pandas()
    _validate_window_length(window)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")
    if len(df) < window:
        raise ValueError("Dataframe has insufficient rows for the requested window")
    return df[column].iloc[-window:]


def _normalise_series(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return arr - mean
    return (arr - mean) / std


def dtw_distance(series_a: Sequence[float], series_b: Sequence[float]) -> float:
    """Compute the Dynamic Time Warping distance between two sequences."""

    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Sequences must be non-empty")

    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = abs(a[i - 1] - b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def _iter_candidate_windows(
    df: pd.DataFrame,
    column: str,
    window: int,
    *,
    lookback_years: float,
) -> Iterable[Tuple[int, int, pd.Series]]:
    """Yield rolling candidate windows within the lookback horizon."""

    pandas = _require_pandas()
    _validate_window_length(window)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")
    if (
        isinstance(df.index, pandas.DatetimeIndex)
        and lookback_years is not None
        and lookback_years > 0
    ):
        end_timestamp = df.index[-1]
        days_offset = max(1, int(lookback_years * 365))
        start_timestamp = end_timestamp - pandas.Timedelta(days=days_offset)
        if start_timestamp <= df.index[0]:
            start_idx = 0
        else:
            start_idx = int(df.index.get_indexer([start_timestamp], method="bfill")[0])
            if start_idx == -1:
                start_idx = 0
    else:
        start_idx = 0

    upper_bound = len(df) - window
    if upper_bound <= start_idx:
        return

    for idx in range(start_idx, upper_bound):
        yield idx, idx + window, df[column].iloc[idx : idx + window]


def compute_dtw_matrix(
    df: pd.DataFrame,
    column: str = "Close",
    window: int = 20,
    *,
    lookback_years: float = 10.0,
) -> pd.DataFrame:
    """Compute DTW distances between the latest pattern and historical windows."""

    _require_pandas()
    target_window = extract_recent_window(df, column=column, window=window)
    target_norm = _normalise_series(target_window)

    rows = []
    for start_idx, end_idx, window_values in _iter_candidate_windows(
        df, column, window, lookback_years=lookback_years
    ):
        candidate_norm = _normalise_series(window_values)
        distance = dtw_distance(target_norm, candidate_norm)
        rows.append((start_idx, end_idx, distance))

    return pd.DataFrame(rows, columns=["start_index", "end_index", "distance"]).sort_values(
        "distance"
    )


def select_neighbors(distance_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Select the k-nearest neighbours based on DTW distance."""

    if distance_df.empty:
        raise ValueError("Distance dataframe is empty; insufficient history")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    return distance_df.nsmallest(k, "distance").reset_index(drop=True)


def _time_to_threshold(
    highs: Sequence[float],
    lows: Sequence[float],
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
) -> Tuple[Optional[int], Optional[int]]:
    tp_level = entry_price * (1 + tp_pct)
    sl_level = entry_price * (1 - sl_pct)

    time_to_tp = None
    time_to_sl = None

    for idx, (high, low) in enumerate(zip(highs, lows), start=1):
        if time_to_tp is None and high >= tp_level:
            time_to_tp = idx
        if time_to_sl is None and low <= sl_level:
            time_to_sl = idx
        if time_to_tp is not None and time_to_sl is not None:
            break

    return time_to_tp, time_to_sl


def summarise_matches(
    df: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    window: int,
    forecast_horizon: int = 10,
    price_column: str = "Close",
    high_column: str = "High",
    low_column: str = "Low",
    tp_pct: float = 0.02,
    sl_pct: float = 0.02,
) -> List[PatternMatch]:
    """Create detailed statistics for each neighbour match."""

    _require_pandas()
    results: List[PatternMatch] = []
    for _, row in matches.iterrows():
        start_idx = int(row["start_index"])
        end_idx = int(row["end_index"])
        if end_idx - start_idx != window:
            continue
        entry_idx = end_idx - 1
        if entry_idx >= len(df) - 1:
            # Skip incomplete future data
            continue

        future = df.iloc[entry_idx + 1 : entry_idx + 1 + forecast_horizon]
        if future.empty:
            continue

        entry_price = float(df.iloc[entry_idx][price_column])
        closing_returns = (future[price_column] - entry_price) / entry_price
        final_return = float(closing_returns.iloc[-1])
        max_run_up = float((future[high_column].max() - entry_price) / entry_price)
        max_drawdown = float((future[low_column].min() - entry_price) / entry_price)
        time_to_tp, time_to_sl = _time_to_threshold(
            future[high_column], future[low_column], entry_price, tp_pct, sl_pct
        )

        results.append(
            PatternMatch(
                start_index=start_idx,
                end_index=end_idx,
                distance=float(row["distance"]),
                entry_price=entry_price,
                final_return=final_return,
                max_run_up=max_run_up,
                max_drawdown=max_drawdown,
                time_to_tp=time_to_tp,
                time_to_sl=time_to_sl,
            )
        )
    return results


def generate_trade_guidance(
    df: pd.DataFrame,
    *,
    column: str = "Close",
    window: int = 20,
    k: int = 5,
    forecast_horizon: int = 10,
    tp_pct: float = 0.02,
    sl_pct: float = 0.02,
    lookback_years: float = 10.0,
    visualisation_hook: Optional[Callable[[pd.DataFrame], None]] = None,
) -> TradeGuidance:
    """Generate trade guidance using DTW k-nearest neighbours."""

    _require_pandas()
    distances = compute_dtw_matrix(
        df, column=column, window=window, lookback_years=lookback_years
    )
    neighbours = select_neighbors(distances, k)
    detailed_matches = summarise_matches(
        df,
        neighbours,
        window=window,
        forecast_horizon=forecast_horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )

    if not detailed_matches:
        raise ValueError("No valid matches with sufficient future data")

    final_returns = np.array([match.final_return for match in detailed_matches])
    entry_confidence = float((final_returns > 0).mean())
    expected_return = float(final_returns.mean())

    tp_hits = [match.time_to_tp is not None for match in detailed_matches]
    sl_hits = [match.time_to_sl is not None for match in detailed_matches]
    avg_exit_horizon = int(
        np.ceil(
            np.mean(
                [
                    min(
                        t for t in [match.time_to_tp, match.time_to_sl, forecast_horizon]
                        if t is not None
                    )
                    for match in detailed_matches
                ]
            )
        )
    )

    guidance = TradeGuidance(
        entry_confidence=entry_confidence,
        expected_return=expected_return,
        take_profit=tp_pct,
        stop_loss=sl_pct,
        exit_horizon=avg_exit_horizon,
        tp_hit_probability=float(np.mean(tp_hits)),
        sl_hit_probability=float(np.mean(sl_hits)),
        matches=detailed_matches,
    )

    if visualisation_hook is not None:
        visualisation_hook(neighbours)

    return guidance


def default_visualisation_hook(neighbours: pd.DataFrame) -> None:
    """Basic example visualisation hook printing the top matches.

    Users can replace this with more elaborate reporting or plotting.
    """

    print("Top pattern matches (start_index, end_index, distance):")
    for _, row in neighbours.iterrows():
        print(int(row.start_index), int(row.end_index), float(row.distance))
