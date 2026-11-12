import pandas as pd
from src.preprocessing import aggregate_hourly, split_data, pattern_imputer, eval_imputer, clean_data

# Tests using pytest

def test_aggregate_hourly_averages_values():
    index = pd.date_range("2026-01-01", periods=120, freq="min")

    df = pd.DataFrame({"Global_active_power": [1.0] * 60 + [-1.0] * 20 + [0] * 20 + [1.0] * 20 }, index=index)

    result = aggregate_hourly(df)

    expected_result = pd.DataFrame({"Global_active_power": [1.0, 0.0] }, index=pd.date_range("2026-01-01", periods=2, freq="h"))

    assert result.equals(expected_result)