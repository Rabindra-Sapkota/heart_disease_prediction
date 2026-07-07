from data_cleaning import load_data
from config import RAW_DATA_PATH
from data_cleaning import clean_data
from data_pipeline import train_svc_model



def main():
    print('Staring Code')

if __name__ == '__main__':
    print("Loading Data")
    raw_df = load_data(RAW_DATA_PATH)

    print("Initiating Data Cleaning")
    clean_df = clean_data(raw_df)

    print("Data Cleaning Completed")

    train_svc_model(clean_df)
    # train_knn_model(clean_df)
    # train_logistic_regression_model(clean_df)
    # train_random_forest_model(clean_df)

