import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'heart_disease_raw.csv')
CLEAN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_heart_disease_data.csv')

# Model Paths
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_disease_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

# Data Columns
TARGET_COLUMN_RAW = 'Heart_ stroke'
TARGET_COLUMN_CLEANED = 'heart_stroke'
