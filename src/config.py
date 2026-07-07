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


CATEGORICAL_COLUMN_MAPPINGS = (
    ("Gender", GENDER_MAPPING),
    ("education", EDUCATION_MAPPING),
    ("Exercise", EXERCISE_MAPPING),
)


## Training param
RANDOM_SEED = 101
SPLIT_SIZE = 0.2

# Parameter Grid for RandomSearchCV
SVC_PARAM_GRID = {
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["linear", "rbf"],
    }