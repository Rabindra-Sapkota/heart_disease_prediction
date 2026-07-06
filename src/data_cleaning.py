import pandas as pd
import os
import config
import numpy as np


def load_data(data_path, sep=","):
    df = pd.read_csv(data_path, sep=sep)
    return df


def clean_data(df):
    if os.path.exists(config.CLEAN_DATA_PATH):
        print("Clean Data Already Exists. Loading from file")
        clean_df = load_data(config.CLEAN_DATA_PATH)
        return clean_df
    else:
        print("Cleanup required")
        df_copy = df.copy()
        print('Renaming column')
        df_copy.rename(
            columns={config.TARGET_COLUMN_RAW: config.TARGET_COLUMN_CLEANED},
            inplace=True,
        )
        df_copy.drop(columns=config.UNWANTED_COLUMNS, inplace=True)
        # print(df_copy.columns) # : To check column names
        # print(df_copy['Gender'].str.lower().value_counts())
        # print(df_copy['Exercise'].str.lower().value_counts())

        for col_name, mapping in config.CATEGORICAL_COLUMN_MAPPINGS:
            print(f"Cleaning data for column: {col_name}")
            col_series = df_copy[col_name]
            df_copy[col_name] = _clean_categorical(col_series, mapping)

        for numeric_col in config.NUMERIC_COLUMNS:
            print(f"Cleaning data for column: {numeric_col}")
            df_copy[numeric_col] = _clean_numeric(df_copy[numeric_col])

        print("Removing duplicates")
        df_copy.drop_duplicates(inplace=True)

        print("Saving files")
        df_copy.to_csv(config.CLEAN_DATA_PATH, index=False)

        print("Cleanup completed")
        return df_copy


def _clean_categorical(series, series_mapping):
    clean_series = series.str.lower()
    mode_value = series.mode()
    clean_series = clean_series.fillna(mode_value, inplace=True)
    clean_series = clean_series.map(series_mapping)
    return clean_series

def _clean_numeric(series):
    new_ser = _clean_non_numeric_in_digit(series)
    median = np.median(new_ser)
    new_ser.fillna(median)
    clean_series = _cleanup_outlier(new_ser)
    return clean_series


def _cleanup_outlier(series):
    q1 = np.quantile(series, 0.25)
    q3 = np.quantile(series, 0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return np.clip(series, lower, upper)


def _clean_non_numeric_in_digit(series):
    if series.dtypes in (int, float):
        return series
    non_digit_regex = r"[^0-9\.-]"
    clean_ser = series.str.replace(non_digit_regex, "", regex=True)
    return clean_ser.astype(float)
