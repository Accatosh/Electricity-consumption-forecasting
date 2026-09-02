import pandas as pd

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates, deletes and shifts some features to improve forecasting.

    Note: requires that the index of df is in datetime format.
    """
    target_variable = "Global_active_power"
    delete_features = ["Global_reactive_power", "Voltage", "Global_intensity"]
    leaking_features = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

    df = df.copy()

    # Add some features
    df["lag_1h"] = df[target_variable].shift(1)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week

    # Delete some features
    df[delete_features] = df.drop(delete_features, axis=1)

    # Shift some features
    df[leaking_features] = df[leaking_features].shift(1)
    df = df.rename(columns={
        leaking_features[0]: "Prev_sub_metering_1",
        leaking_features[1]: "Prev_sub_metering_2",
        leaking_features[2]: "Prev_sub_metering_3"
    })

    return df
