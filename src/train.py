from data_cleaning import load_data
from config import RAW_DATA_PATH
from data_cleaning import clean_data


def main():
    print('Staring Code')

if __name__ == '__main__':
    raw_df = load_data(RAW_DATA_PATH)
    clean_df = clean_data(raw_df)
    