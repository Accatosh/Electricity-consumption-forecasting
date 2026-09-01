from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import math
import warnings

def load_data() -> dict[str, pd.DataFrame]:
    """
    Loads and preprocesses the data from UCI Machine Learning Repository.
    Citation:
    Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. 
    UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.

    The following are the keys for the output dictionary :
    - "full" -> full dataset
    - "raw" -> raw dataset
    - "train" -> train set
    - "validation" -> validation set
    - "test" -> test set
    - "full_train" -> full train set

    Note: the full dataset's missing values has not been imputed.
    """
    # Download original data
    df = load_raw_data()
    raw_df = df.copy()

    dfs = clean_data(df)

    dfs["full"] = df
    dfs["raw"] = raw_df

    return dfs


def load_raw_data() -> pd.DataFrame:
    """
    Downloads and loads the data from UCI Machine Learning Repository.
    Citation:
    Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. 
    UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.
    """
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)
    
    return individual_household_electric_power_consumption.data.features


def clean_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Splits and cleans the data. 

    The following are the keys for the output dictionary :
    - "train" -> train set
    - "validation" -> validation set
    - "test" -> test set
    - "full_train" -> full train set
    """
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    # Make the missing data consistent
    df = df.replace("?", np.nan)
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Set the date and time as the index 
    df["datetime"] = pd.to_datetime( df["Date"] + " " + df["Time"])
    df = df.set_index("datetime")
    df = df.drop(columns=["Date", "Time"])

    # Aggregate to hourly data
    df = aggregate_hourly(df)

    # Split the data and impute missing values 
    return split_and_impute(df)


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts minutely data to hourly. This should be done before using the cleaning the data.
    """
    df = df.copy()
    return df.resample("1h").mean()


def split_and_impute(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Splits the data and fills missing values. 

    The following are the keys for the output dictionary :
    - "train" -> train set
    - "validation" -> validation set
    - "test" -> test set
    - "full_train" -> full train set
    """

    # Split data
    splitted_data = split_data(df)
    train_set = splitted_data["train"]
    validation_set = splitted_data["validation"]
    test_set = splitted_data["test"]
    full_train_set = splitted_data["full_train"]

    # Clean data
    clean_train_set = impute_data(train_set)
    clean_val_set = impute_data(validation_set, True, df)
    clean_test_set = impute_data(test_set, True, df)
    clean_full_train_set = impute_data(full_train_set)

    output = {
        "train": clean_train_set,
        "validation": clean_val_set,
        "test": clean_test_set,
        "full_train": clean_full_train_set
    }

    return output


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Splits the data into train, validation, test and full train set.

    The following are the keys for the output dictionary :
    - "train" -> train set
    - "validation" -> validation set
    - "test" -> test set
    - "full_train" -> full train set
    """
    n = df.shape[0]
    valid_cut_i = math.floor(n * 0.8)
    test_cut_i = math.floor(n * 0.9)

    train_set = df.iloc[:valid_cut_i]
    validation_set = df.iloc[valid_cut_i : test_cut_i]
    test_set = df.iloc[test_cut_i:]
    full_train_set = df.iloc[:test_cut_i]

    output = {
        "train": train_set,
        "validation": validation_set,
        "test": test_set,
        "full_train": full_train_set
    }

    return output


def impute_data(df: pd.DataFrame, is_eval_data=False, df_full: pd.DataFrame = None) -> pd.DataFrame:
    """
    Fills missing values in the data.

    Parameters :
        - df: the dataframe to clean.
        - is_eval_data: True if df is an evaluation dataset, i.e. either a validation or test set. If set to True, the next parameter needs a value.
        - df_full: the full dataset.
    """

    if (is_eval_data):

        if (df_full is not None):
            return eval_imputer(df, df_full)

        else:
            warnings.warn("Tried cleaning evaluation data without passing the full dataset.")
            return None

    else:
        return pattern_imputer( df )
            

def pattern_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    The imputer used for the training data. 
    Imputes values based on the average from the day before & after, and a week earlier and later.
    """
    df = df.copy()

    values = [
            df.shift(24 * 7),
            df.shift(24),
            df.shift(-24),
            df.shift(-24 * 7)
    ]

    imputed = pd.concat(values).groupby(level=0, sort=False).mean()

    return df.fillna(imputed)


def eval_imputer(val_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    The imputer used for the validation and test data.
    Imputes values based on the average from the previous day before and the day a week earlier.
    
    Parameters :
    - val_df: the dataframe to impute.
    - full_df: the full dataset so that the evaluation set has access to the full data history.
    """
    val_df = val_df.copy()

    values = [
            full_df.shift(24 * 7),
            full_df.shift(24)
    ]

    imputed = pd.concat(values).groupby(level=0, sort=False).mean()

    return val_df.fillna(imputed)