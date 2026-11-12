import pandas as pd
import numpy as np
from src.features import feature_engineer

# Tests using pytest

index = pd.date_range("2026-01-01", periods=25, freq="h")
test_df = pd.DataFrame({"Global_active_power": [i for i in range(25)],
                        "Global_reactive_power": [np.nan] * 25,
                        "Voltage": [np.nan] * 25,
                        "Global_intensity": [np.nan] * 25,
                        "Sub_metering_1": [i for i in range(-10, 15)], 
                        "Sub_metering_2": [i for i in range(-15, 10)], 
                        "Sub_metering_3": [i for i in range(-25, 0)]
                        }, index=index)

expected = pd.DataFrame({"Global_active_power": [i for i in range(25)],
                         "lag_1h": [np.nan] + [i for i in range(24)],
                         "hour": [i for i in range(24)] + [0],
                         "day_of_week": [3] * 24 + [4],
                         "Prev_sub_metering_1": [np.nan] + [i for i in range(-10, 14)], 
                         "Prev_sub_metering_2": [np.nan] + [i for i in range(-15, 9)], 
                         "Prev_sub_metering_3": [np.nan] + [i for i in range(-25, -1)]
                         }, index=index)


def test_feature_engineer_creates_lag_and_calendar_features():
    global test_df, expected
    
    df = test_df.copy()
    drop_features = ["Prev_sub_metering_1", "Prev_sub_metering_2", "Prev_sub_metering_3"]

    result = feature_engineer(df).drop(drop_features, axis=1)
    expected_result = expected.drop(drop_features, axis=1)
    result[result.columns] = result[result.columns].astype("float64")
    expected_result[expected_result.columns] = expected_result[expected_result.columns].astype("float64")

    assert result.equals(expected_result)


def test_feature_engineer_shifts_submetering_features():
    global test_df, expected
        
    df = test_df.copy()
    drop_features = ["Global_active_power", "lag_1h", "hour", "day_of_week"]

    result = feature_engineer(df).drop(drop_features, axis=1)
    expected_result = expected.drop(drop_features, axis=1)
    result[result.columns] = result[result.columns].astype("float64")
    expected_result[expected_result.columns] = expected_result[expected_result.columns].astype("float64")

    print(result)
    print("---")
    print(expected_result)

    assert result.equals(expected_result)