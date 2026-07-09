import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "heart_disease_raw.csv")
CLEAN_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "cleaned_heart_disease_data.csv"
)

# Model Paths
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# Data Columns
TARGET_COLUMN_RAW = "Heart_ stroke"
TARGET_COLUMN_CLEANED = "heart_stroke"
NUMERIC_COLUMNS = [
    "age",
    "cigsPerDay",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]


CATEGORICAL_COLUMNS = ['Gender', 'education', 'Exercise']
ORDINAL_COLUMNS = ['education', 'Exercise']
EDUCATION_ORDER = ["uneducated", "primary_school", "graduate", "postgraduate"]
EXERCISE_ORDER = ["daily", "weekly", "monthly"]

NOMINAL_COLUMNS = ['Gender']
BOOLEAN_COLUMNS = ["currentSmoker", "prevalentStroke", "prevalentHyp"]
UNWANTED_COLUMNS = ['BPMeds']
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS


GENDER_MAPPING = {
    "female": "F",
    "male": "M",
    "f": "F",
    "m": "M",
    "woman": "F",
    "man": "M",
}

EDUCATION_MAPPING = {
    "primaryschool": "primary_school",
    "uneducated": "uneducated",
    "graduate": "graduate",
    "postgraduate": "postgraduate",
    "uneducate": "uneducated",
    "primary school": "primary_school",
    "primary_school": "primary_school",
}

EXERCISE_MAPPING = {
    "monthly": "monthly",
    "daily": "daily",
    "weekly": "weekly",
    "day": "daily",
    "week": "weekly",
    "dialy": "daily",
}

TARGET_COLUMN_MAPPING = {
    "no": 0,
    "yes": 1,
    "No": 0,
    "YES": 1,
    0: 0,
    1: 1,
}

CATEGORICAL_COLUMN_MAPPINGS = (
    ("Gender", GENDER_MAPPING),
    ("education", EDUCATION_MAPPING),
    ("Exercise", EXERCISE_MAPPING),
)


## Training param
RANDOM_SEED = 101
SPLIT_SIZE = 0.2
N_ITER = 10
CV = 10

# Parameter Grid for RandomSearchCV # SVC Hyperparameters
SVC_PARAM_GRID = {
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["linear", "rbf"],
    }


# KNN Hyperparameters
KNN_PARAM_GRID = {
    "classifier__n_neighbors": [3, 5, 7, 9, 11],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}

# Logistic Regression Hyperparameters
LOGISTIC_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"],
    "classifier__max_iter": [100, 300, 500],
}

# Random Forest Hyperparameters
RANDOM_FOREST_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}
