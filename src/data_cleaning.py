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
        df_copy  = df.copy()
        df_copy.rename(columns={config.TARGET_COLUMN_RAW: config.TARGET_COLUMN_CLEANED}, inplace=True)
        # print(df_copy.columns) # : To check column names
        # print(df_copy['Gender'].value_counts())
        
        df['Gender'] = _clean_gender(df['Gender'])
        df['age'] = _clean_numeric(df['age'])

        # 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
        # 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
        # 'diaBP', 'BMI', 'heartRate', 'glucose', 'Exercise', 'heart_stroke']

def _clean_gender(gender_series):
    gender_mapping = {"Female": "F", "Male": "M", "F": "F",
                      "M":"M", "Woman": "F", "Man": "M"}
    clean_gender = gender_series.map(gender_mapping)
    return clean_gender.value_counts()

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
    non_digit_regex = r'[^0-9\.-]'
    clean_ser = series.str.replace(non_digit_regex, '', regex=True)
    return clean_ser.astype(float)