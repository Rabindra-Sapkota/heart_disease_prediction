from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import config
import os
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
import time
import joblib
from sklearn.model_selection import train_test_split


def split_data_for_training(df):
    df_copy = df.copy()
    X = df_copy[config.FEATURE_COLUMNS]
    y = df_copy[config.TARGET_COLUMN_CLEANED]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.SPLIT_SIZE, random_state=config.RANDOM_SEED, stratify=y)
    return X_train, X_test, y_train, y_test

# OLD CODE
# def train_svc_model(df):
#     print("Starting Training of SVC")
#     svc_pipeline = build_svc_pipeline()
#     best_model = find_model_with_optimal_hyperparameters(df, svc_pipeline, config.SVC_PARAM_GRID)
#     dump_best_model(best_model, "best_svc_model.pkl")
#     print("SVC training completed")


# Generic Or Dynamic pipeline 
def build_pipeline(model):

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


# OLD CODE
# def build_svc_pipeline():
#     preprocessor = build_preprocessor()
#     svc_pipeline = Pipeline(
#         steps= [
#             ("preprocessor", preprocessor),
#             ("classifier", SVC())
#         ]
#     )

#     return svc_pipeline

# Generic Dynamic training function
def train_model(df, model, param_grid):

    pipeline = build_pipeline(model)

    best_model, accuracy = find_model_with_optimal_hyperparameters(
        df,
        pipeline,
        param_grid,
    )

    return best_model, accuracy



def build_preprocessor():
    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    categories=[config.EDUCATION_ORDER, config.EXERCISE_ORDER]
                ),
            )
        ]
    )

    nominal_pipeline = Pipeline(steps=[("binary_encoder", BinaryEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, config.NUMERIC_COLUMNS),
            ("ord", ordinal_pipeline, config.ORDINAL_COLUMNS),
            ("nom", nominal_pipeline, config.NOMINAL_COLUMNS),
        ]
    )

    return preprocessor


# def find_model_with_optimal_hyperparameters(df, model_pipeline, param_grid):
#     X_train, X_test, y_train, y_test = split_data_for_training(df)

#     random_search = RandomizedSearchCV(
#         estimator=model_pipeline,
#         param_distributions=param_grid,
#         cv=10,
#         n_jobs=-1,
#         verbose=2
#     )

#     start_time = time.time()
#     print("Starting Randomized Search for Hyperparameter Tuning...")
#     print(start_time)
#     random_search.fit(X_train, y_train)
#     end_time = time.time()
#     print("Training Completed")
#     print(end_time)

#     best_params = random_search.best_params_
#     best_score = random_search.best_score_
#     best_model = random_search.best_estimator_

#     return best_model


def find_model_with_optimal_hyperparameters(df, model_pipeline, param_grid):
    """
    Finds the best hyperparameters for the given model pipeline using
    RandomizedSearchCV and returns the best trained model along with
    its test accuracy.
    """

    # Split Data
    X_train, X_test, y_train, y_test = split_data_for_training(df)

    # Hyperparameter Search
    random_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_grid,
        n_iter=config.N_ITER,
        cv=config.CV,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=2,
    )

    print("\nStarting Randomized Search...")
    start_time = time.time()

    random_search.fit(X_train, y_train)

    end_time = time.time()

    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

    # Best Results
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    # Evaluate on Test Set
    test_accuracy = best_model.score(X_test, y_test)

    print("\n========== Training Summary ==========")
    print(f"Best Parameters : {best_params}")
    print(f"Best CV Score   : {best_cv_score:.4f}")
    print(f"Test Accuracy   : {test_accuracy:.4f}")
    print("======================================\n")

    return best_model, test_accuracy

    # y_pred = best_model.predict(X_test)
    # test_accuracy = best_model.score(X_test, y_test)


# def dump_best_model(model, model_name):
#     full_path = os.path.join(config.MODEL_FOLDER, model_name)
#     joblib.dump(model, full_path)
#     print(f"Best model saved as {model_name}")

def dump_best_model(model, model_name):

    os.makedirs(config.MODEL_FOLDER, exist_ok=True)

    full_path = os.path.join(
        config.MODEL_FOLDER,
        model_name,
    )

    joblib.dump(model, full_path)

    print(f"Best model saved at {full_path}")