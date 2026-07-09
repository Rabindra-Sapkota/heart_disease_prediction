import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import config
from data_cleaning import load_data

st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide")

MODEL_FILES = {
    "SVC": "best_model_svc.pkl",
    "KNN": "best_model_knn.pkl",
    "Logistic Regression": "best_model_logistic_regression.pkl",
    "Random Forest": "best_model_random_forest.pkl",
}


@st.cache_data(show_spinner=False)
def load_dataset():
    
    return load_data(str(config.CLEAN_DATA_PATH))


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics_path = ROOT_DIR / "models" / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_selected_model(model_name: str):
    model_path = ROOT_DIR / "models" / MODEL_FILES[model_name]
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def get_feature_template():
    return list(config.FEATURE_COLUMNS)


def build_prediction_frame(inputs: dict) -> pd.DataFrame:
    feature_names = get_feature_template()
    row = {name: [inputs[name]] for name in feature_names}
    return pd.DataFrame(row)


def get_prediction_probability(model, feature_frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)
        return float(probabilities[0][1])

    if hasattr(model, "decision_function"):
        decision = model.decision_function(feature_frame)[0]
        return float(1 / (1 + np.exp(-decision)))

    return 0.5


def render_accuracy_chart(metrics: dict):
    chart_df = pd.DataFrame({"Model": list(metrics.keys()), "Accuracy": list(metrics.values())})
    chart_df = chart_df.sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(chart_df["Model"], chart_df["Accuracy"], color="#4C78A8")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xticklabels(chart_df["Model"], rotation=25, ha="right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def render_eda(df: pd.DataFrame):
    st.subheader("Target Distribution")
    target_counts = df[config.TARGET_COLUMN_CLEANED].value_counts().reindex([0, 1])
    st.bar_chart(target_counts)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution by Risk")
        fig_age, ax_age = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="age", hue=config.TARGET_COLUMN_CLEANED, kde=True, ax=ax_age)
        ax_age.set_title("Age Distribution")
        st.pyplot(fig_age)

    with col2:
        st.subheader("Blood Pressure by Risk")
        fig_bp, ax_bp = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=config.TARGET_COLUMN_CLEANED, y="sysBP", ax=ax_bp)
        ax_bp.set_title("Systolic BP by Outcome")
        ax_bp.set_xticklabels(["No Risk", "Risk"])
        st.pyplot(fig_bp)

    st.subheader("Correlation Heatmap")
    numeric_cols = [col for col in config.NUMERIC_COLUMNS if col in df.columns]
    corr = df[numeric_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Numeric Feature Correlation")
    st.pyplot(fig_corr)


def render_dataset_description(df: pd.DataFrame):
    st.subheader("Dataset Description")
    st.write(
        "This cleaned dataset contains health and lifestyle attributes used to predict heart-stroke risk."
    )
    st.markdown(
        f"- Rows: {df.shape[0]}\n"
        f"- Columns: {df.shape[1]}\n"
        f"- Target column: {config.TARGET_COLUMN_CLEANED}\n"
        f"- Positive cases: {int(df[config.TARGET_COLUMN_CLEANED].sum())}"
    )
    st.markdown("### Column Types")
    dtype_summary = df.dtypes.rename("dtype").reset_index().rename(columns={"index": "Column"})
    st.dataframe(dtype_summary, use_container_width=True)


def render_data_explorer(df: pd.DataFrame):
    st.subheader("Dataset Preview")
    view_option = st.radio("Preview", ["Head", "Tail", "Sample"], horizontal=True, key="data_preview")
    row_count = st.slider("Number of rows", 5, 30, 10, key="data_rows")

    if view_option == "Head":
        st.dataframe(df.head(row_count), use_container_width=True)
    elif view_option == "Tail":
        st.dataframe(df.tail(row_count), use_container_width=True)
    else:
        st.dataframe(df.sample(row_count, random_state=42), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)


def render_prediction_panel(metrics: dict):
    st.subheader("Make a Prediction")
    st.caption("This uses your saved pre-trained model from the models folder.")

    prediction_col, performance_col = st.columns([2, 1])

    with prediction_col:
        selected_model = st.selectbox("Choose a pre-trained model", list(MODEL_FILES.keys()))

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=90, value=50)
                gender = st.selectbox("Gender", ["F", "M"], index=0)
                education = st.selectbox("Education", ["uneducated", "primary_school", "graduate", "postgraduate"], index=1)
                exercise = st.selectbox("Exercise", ["daily", "weekly", "monthly"], index=0)
            with col2:
                cigs_per_day = st.number_input("Cigarettes per day", min_value=0.0, max_value=60.0, value=0.0)
                tot_chol = st.number_input("Total Cholesterol", min_value=100.0, max_value=400.0, value=220.0)
                sys_bp = st.number_input("Systolic BP", min_value=80.0, max_value=220.0, value=130.0)
                dia_bp = st.number_input("Diastolic BP", min_value=50.0, max_value=140.0, value=80.0)
            with col3:
                bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=25.0)
                heart_rate = st.number_input("Heart Rate", min_value=40.0, max_value=140.0, value=75.0)
                glucose = st.number_input("Glucose", min_value=40.0, max_value=220.0, value=80.0)
                diabetes = st.selectbox("Diabetes", [0, 1], index=0)

            current_smoker = st.selectbox("Current Smoker", [0, 1], index=0)
            prevalent_stroke = st.selectbox("Prevalent Stroke", [0, 1], index=0)
            prevalent_hyp = st.selectbox("Prevalent Hypertension", [0, 1], index=0)

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            model = load_selected_model(selected_model)
            inputs = {
                "age": age,
                "Gender": gender,
                "education": education,
                "Exercise": exercise,
                "cigsPerDay": cigs_per_day,
                "currentSmoker": current_smoker,
                "prevalentStroke": prevalent_stroke,
                "prevalentHyp": prevalent_hyp,
                "diabetes": diabetes,
                "totChol": tot_chol,
                "sysBP": sys_bp,
                "diaBP": dia_bp,
                "BMI": bmi,
                "heartRate": heart_rate,
                "glucose": glucose,
            }

            feature_frame = build_prediction_frame(inputs)
            prediction = int(model.predict(feature_frame)[0])
            probability = get_prediction_probability(model, feature_frame)

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("Predicted: High Risk of Heart Stroke")
            else:
                st.success("Predicted: Low Risk of Heart Stroke")

            st.metric("Model", selected_model)
            st.metric("Risk Probability", f"{probability * 100:.1f}%")
            st.progress(probability)

    with performance_col:
        st.markdown("### Model Performance")
        st.pyplot(render_accuracy_chart(metrics))
        st.markdown("**Accuracy Summary**")
        for model_name, accuracy in sorted(metrics.items(), key=lambda item: item[1], reverse=True):
            st.write(f"- {model_name}: {accuracy:.2f}")


def main():
    df = load_dataset()
    metrics = load_metrics()

    st.title("Heart Disease Prediction Dashboard")
    st.write("Explore the cleaned heart disease dataset, review model performance, and run predictions with pre-trained models.")

    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Description", "Data Explorer", "EDA & Visuals", "Prediction"])

    with tab1:
        render_dataset_description(df)

    with tab2:
        render_data_explorer(df)

    with tab3:
        render_eda(df)

    with tab4:
        render_prediction_panel(metrics)


if __name__ == "__main__":
    main()
