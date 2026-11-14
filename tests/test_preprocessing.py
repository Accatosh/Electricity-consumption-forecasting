import pandas as pd
import numpy as np
from src.preprocessing import aggregate_hourly, split_data, pattern_imputer, eval_imputer, clean_data

# Tests using pytest

def test_aggregate_hourly_averages_values():
    index = pd.date_range("2026-01-01", periods=120, freq="min")

    df = pd.DataFrame({"Global_active_power": [1.0] * 60 + [-1.0] * 20 + [0] * 20 + [1.0] * 20 }, index=index)

    result = aggregate_hourly(df)

    expected_result = pd.DataFrame({"Global_active_power": [1.0, 0.0] }, index=pd.date_range("2026-01-01", periods=2, freq="h"))

    assert result.equals(expected_result)

def test_split_data_preserves_chronological_order():
    index = pd.date_range("2026-01-01", periods=100, freq="h")
    
    df = pd.DataFrame({"Global_active_power": [0]*80 + [1]*10 + [2]*10 }, index=index)

    result = split_data(df)

    train_expected_result = pd.DataFrame({"Global_active_power": [0]*80 }, index=pd.date_range("2026-01-01", periods=80, freq="h"))
    train_full_expected_result = pd.DataFrame({"Global_active_power": [0]*80 + [1]*10 }, index=pd.date_range("2026-01-01", periods=90, freq="h"))
    validation_expected_result = pd.DataFrame({"Global_active_power": [1]*10 }, index=pd.date_range("2026-01-04 08:00", periods=10, freq="h"))
    test_expected_result = pd.DataFrame({"Global_active_power":  [2]*10 }, index=pd.date_range("2026-01-04 18:00", periods=10, freq="h"))

    assert result["train"].equals( train_expected_result ), "Train set is incorrect"
    assert result["full_train"].equals( train_full_expected_result ), "Full train set is incorrect"
    assert result["validation"].equals( validation_expected_result ), "Validation set is incorrect"
    assert result["test"].equals( test_expected_result ), "Test set is incorrect"

def test_train_data_imputer():
    index = pd.date_range("2026-01-01", periods=337, freq="h")
        
    df = pd.DataFrame({"Global_active_power": [-2]*144 + [-1]*24 + [0] + [1]*24 + [2]*144 }, index=index)
    expected_result = df.copy()

    df.iloc[168] = np.nan
    result = pattern_imputer(df).astype("int64")

    assert result.equals(expected_result)

def test_eval_imputer_uses_only_past_values():
    index = pd.date_range("2026-01-01", periods=337, freq="h")
            
    df = pd.DataFrame({"Global_active_power": [-3]*144 + [-1]*24 + [0] + [1000]*24 + [2000]*144 }, index=index)
    expected_result = df.copy()

    df.iloc[168] = np.nan
    expected_result.iloc[168] = -2

    result = eval_imputer(df, df).astype("int64")

    assert result.equals(expected_result)

def test_clean_data_processes_raw_data():
    size = 180
    nr_decimals = 6
    
    df = pd.DataFrame({
        "Date": ["2026-01-01"] * 180,
        "Time": [f"00:{i}{j}" for i in range(6) for j in range(10)] + [f"01:{i}{j}" for i in range(6) for j in range(10)] + [f"02:{i}{j}" for i in range(6) for j in range(10)],
        "Global_active_power": [1] * 60 + ["?"] * 60 + [1] * 60,
        "Global_reactive_power": [0.0] * (size - 1) + ["?"],
        "Voltage": ["?"] + [0] * (size - 1),
        "Global_intensity": [-1] * 89 + ["?"] + [1] * 90,
        "Sub_metering_1": ["1"] * size,
        "Sub_metering_2": [0.0] * size,
        "Sub_metering_3": ["?"] * size
    })

    index = pd.date_range("2026-01-01", periods=3, freq="h")
    expected_result = pd.DataFrame({
        "Global_active_power": [1.0, np.nan, 1.0],
        "Global_reactive_power": [0.0] * 3,
        "Voltage": [0.0] * 3,
        "Global_intensity": [-1.0, 0.0169491525, 1.0],
        "Sub_metering_1": [1.0] * 3,
        "Sub_metering_2": [0.0] * 3,
        "Sub_metering_3": [np.nan] * 3
    }, index=index).astype(float).round(6)

    result = clean_data(df).round(6)

    assert result.equals(expected_result)

    