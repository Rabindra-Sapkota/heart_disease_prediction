from data_cleaning import load_data
from config import RAW_DATA_PATH
from data_cleaning import clean_data
from sklearn.model_selection import train_test_split
import config
from data_pipeline import build_pipeline

def split_data_for_training(df):
    df_copy = df.copy()
    X = df_copy[config.FEATURE_COLUMNS]
    y = df_copy[config.TARGET_COLUMN_CLEANED]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.SPLIT_SIZE, random_state=config.RANDOM_SEED)
    return X_train, X_test, y_train, y_test


def main():
    print('Staring Code')

if __name__ == '__main__':
    print("Loading Data")
    raw_df = load_data(RAW_DATA_PATH)

    print("Initiating Data Cleaning")
    clean_df = clean_data(raw_df)

    print("Data Cleaning Completed")

    X_train, X_test, y_train, y_test = split_data_for_training(clean_df)
    pipeline = build_pipeline()
