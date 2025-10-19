import pytest

pd = pytest.importorskip("pandas")

from src.pattern_trading import (
    compute_dtw_matrix,
    default_visualisation_hook,
    dtw_distance,
    generate_trade_guidance,
    select_neighbors,
    summarise_matches,
)


def test_dtw_distance_basic():
    assert dtw_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert dtw_distance([1, 2, 3], [2, 3, 4]) == pytest.approx(3.0)


def test_neighbor_selection_identifies_closest_patterns():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    base = pd.Series(range(100, 130), index=dates)
    df = pd.DataFrame({"Close": base, "High": base + 2, "Low": base - 0.5})

    distances = compute_dtw_matrix(df, column="Close", window=10, lookback_years=10)
    neighbours = select_neighbors(distances, k=3)

    assert len(neighbours) == 3
    assert (len(df) - 10 - 1) in neighbours["start_index"].values


def test_summarise_matches_produces_expected_metrics():
    dates = pd.date_range("2021-01-01", periods=15, freq="D")
    close = [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
    ]
    high = [
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        115,
        116,
        117,
        118,
    ]
    low = [
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        108,
        107,
        106,
        105,
    ]
    df = pd.DataFrame({"Close": close, "High": high, "Low": low}, index=dates)

    matches = pd.DataFrame(
        {
            "start_index": [0],
            "end_index": [10],
            "distance": [0.5],
        }
    )
    detailed = summarise_matches(
        df,
        matches,
        window=10,
        forecast_horizon=3,
        tp_pct=0.02,
        sl_pct=0.02,
    )

    assert len(detailed) == 1
    match = detailed[0]
    assert match.time_to_tp == 2  # High reaches TP on the second candle.
    assert match.time_to_sl == 3  # Low reaches SL later in the horizon.
    assert match.final_return == pytest.approx((112 - 109) / 109)
    assert match.max_run_up == pytest.approx((116 - 109) / 109)
    assert match.max_drawdown == pytest.approx((106 - 109) / 109)


def test_generate_trade_guidance_end_to_end():
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    close_values = [
        *range(100, 140),
    ]
    high_values = [value + 2 for value in close_values]
    low_values = [value - 0.5 for value in close_values]
    df = pd.DataFrame(
        {"Close": close_values, "High": high_values, "Low": low_values}, index=dates
    )

    guidance = generate_trade_guidance(
        df,
        column="Close",
        window=10,
        k=3,
        forecast_horizon=3,
        tp_pct=0.01,
        sl_pct=0.01,
        lookback_years=5,
        visualisation_hook=default_visualisation_hook,
    )

    assert guidance.entry_confidence > 0.0
    assert guidance.expected_return > 0.0
    assert guidance.tp_hit_probability == pytest.approx(1.0)
    assert guidance.sl_hit_probability == pytest.approx(0.0)
    assert len(guidance.matches) == 3
