from data_cleaning import load_data
from config import RAW_DATA_PATH
import os
from data_cleaning import clean_data
# from data_pipeline import train_svc_model

from data_pipeline import train_model 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import config
import json


def save_accuracy_metrics(accuracy_results):
    config_path = os.path.join(config.MODEL_FOLDER, "metrics.json")
    with open(config_path, "w") as f:
        json.dump(accuracy_results, f, indent=4)





def main():

    print("Loading Data...")
    raw_df = load_data(config.RAW_DATA_PATH)

    print("Initiating Data Cleaning...")
    clean_df = clean_data(raw_df)

    print("Data Cleaning Completed")
    accuracy_results = {}

    # Train SVC
    svc_acc = train_model(
        clean_df,
        SVC(),
        config.SVC_PARAM_GRID,
        "svc"
    )

    accuracy_results["SVC"] = round(svc_acc, 2)


    # Train KNN
    knn_acc = train_model(
        clean_df,
        KNeighborsClassifier(),
        config.KNN_PARAM_GRID,
        "knn"
    )

    accuracy_results["KNN"] = round(knn_acc, 2)

    # Train Logistic Regression
    lr_acc = train_model(
        clean_df,
        LogisticRegression(),
        config.LOGISTIC_PARAM_GRID,
        "logistic_regression"
    )

    accuracy_results["Logistic Regression"] = round(lr_acc, 2)

    # Train Random Forest
    rf_acc = train_model(
        clean_df,
        RandomForestClassifier(),
        config.RANDOM_FOREST_PARAM_GRID,
        "random_forest"
    )

    accuracy_results["Random Forest"] = round(rf_acc, 2)

    save_accuracy_metrics(accuracy_results)

    

if __name__ == "__main__":
    main()

