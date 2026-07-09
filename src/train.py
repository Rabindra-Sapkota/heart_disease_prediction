from data_cleaning import load_data
from config import RAW_DATA_PATH
from data_cleaning import clean_data
# from data_pipeline import train_svc_model

from data_pipeline import train_model 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import config



# def main():
#     print('Staring Code')

# if __name__ == '__main__':
#     print("Loading Data")
#     raw_df = load_data(RAW_DATA_PATH)

#     print("Initiating Data Cleaning")
#     clean_df = clean_data(raw_df)

#     print("Data Cleaning Completed")

#     train_svc_model(clean_df)
#     # train_knn_model(clean_df)
#     # train_logistic_regression_model(clean_df)
#     # train_random_forest_model(clean_df)
    





def main():

    print("Loading Data...")
    raw_df = load_data(config.RAW_DATA_PATH)

    print("Initiating Data Cleaning...")
    clean_df = clean_data(raw_df)

    print("Data Cleaning Completed")

    # Train SVC
    svc_model, svc_acc = train_model(
        clean_df,
        SVC(),
        config.SVC_PARAM_GRID,
    )

    # Train KNN
    knn_model, knn_acc = train_model(
        clean_df,
        KNeighborsClassifier(),
        config.KNN_PARAM_GRID,
    )

    # Train Logistic Regression
    lr_model, lr_acc = train_model(
        clean_df,
        LogisticRegression(),
        config.LOGISTIC_PARAM_GRID,
    )

    # Train Random Forest
    rf_model, rf_acc = train_model(
        clean_df,
        RandomForestClassifier(),
        config.RANDOM_FOREST_PARAM_GRID,
    )

    results = {
        "SVC": (svc_model, svc_acc),
        "KNN": (knn_model, knn_acc),
        "Logistic Regression": (lr_model, lr_acc),
        "Random Forest": (rf_model, rf_acc),
    }
    
    best_model_name = max(
    results,
    key=lambda x: results[x][1]
)

    best_model = results[best_model_name][0]
    best_accuracy = results[best_model_name][1]

    print(f"Best Model : {best_model_name}")
    print(f"Accuracy : {best_accuracy:.4f}")
    
    

if __name__ == "__main__":
    main()

