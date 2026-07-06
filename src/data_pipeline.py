from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from src import config
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer


def build_pipeline():
    pass


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