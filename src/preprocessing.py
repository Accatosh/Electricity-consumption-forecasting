from ucimlrepo import fetch_ucirepo
import pandas as pd
import math
import warnings

def load_data() -> pd.DataFrame:
    """
    Downloads and loads the data from UCI Machine Learning Repository.
    Citation:
    Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. 
    UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.
    """
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)

    return individual_household_electric_power_consumption.data.features

def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts minutely data to hourly. This should be done before using the cleaning the data.
    """
    df = df.copy()
    return df.resample("1h").mean()

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


def clean_data(df: pd.DataFrame, is_eval_data=False, df_full: pd.DataFrame = None) -> pd.DataFrame:
    """
    Cleans a dataset by filling in missing data.

    Parameters :
        - df: the dataframe to clean.
        - is_eval_data: True if df is an evaluation data set, i.e. either a validation or test set. If set to True, the next parameter needs a value.
    """

    if (is_eval_data):

        if (df_full is not None):
            return eval_imputer(df, df_full)

        else:
            warnings.warn("Tried cleaning evaluation data without passing the full dataset.")
            return None

    else:
        return pattern_impute( df )
            


def pattern_impute(df: pd.DataFrame) -> pd.DataFrame:
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