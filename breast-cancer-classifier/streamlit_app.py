from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Local project paths
PROJECT_DIR = Path(__file__).resolve().parent
    
)
MODEL_DIR = PROJECT_DIR / "model"
SAMPLE_CSV = PROJECT_DIR / "test_data.csv"

st.set_page_config(
    page_title="Tumor Classifier Lab",
    page_icon="🧬",
    layout="wide",
)

# Animated medical background and custom interface styling
st.markdown(
    """
    <style>
    .stApp {
        background-image:
            linear-gradient(rgba(5, 18, 34, 0.78), rgba(5, 18, 34, 0.87)),
            url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmidvY-R-OuE_D-SdaNCSOrhvPSD1riTvOfX8SWCPiAw&s=10");
        background-size: cover, cover;
        background-position: center, 48% 50%;
        background-attachment: fixed;
        animation: medical-pan 28s ease-in-out infinite alternate;
    }

    @keyframes medical-pan {
        from { background-position: center, 44% 50%; }
        to   { background-position: center, 56% 50%; }
    }

    [data-testid="stSidebar"] {
        background: rgba(7, 25, 43, 0.94);
        border-right: 1px solid rgba(110, 220, 230, 0.25);
    }

    .block-container {
        background: rgba(8, 25, 43, 0.78);
        padding: 2rem 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 18px;
        backdrop-filter: blur(8px);
        animation: fade-in 0.7s ease-out;
    }

    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: #f3f8ff;
    }

    [data-testid="stMetric"] {
        background: rgba(20, 172, 181, 0.14);
        border: 1px solid rgba(110, 220, 230, 0.28);
        padding: 14px;
        border-radius: 14px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 0.8rem;
    }

    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #62d5d3;
        background: linear-gradient(100deg, #087f8c, #13a9a2);
        color: white;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 18px rgba(19, 169, 162, 0.35);
        color: white;
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp, .block-container {
            animation: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the saved model bundles
model_paths = sorted(
    list(MODEL_DIR.glob("*.joblib")) +
    list(MODEL_DIR.glob("*.pkl"))
)

if not model_paths:
    st.error(f"No saved model files found in: {MODEL_DIR}")
    st.stop()

loaded_models = {}
load_errors = []

for path in model_paths:
    try:
        bundle = joblib.load(path)

        if not isinstance(bundle, dict) or "model" not in bundle:
            load_errors.append(
                f"{path.name} does not contain the model metadata from Colab."
            )
            continue

        display_name = path.stem.replace("_", " ").title()

        loaded_models[display_name] = {
            "path": path,
            "model": bundle["model"],
            "feature_columns": bundle["feature_columns"],
            "target_column": bundle["target_column"],
            "label_encoder": bundle["label_encoder"],
        }

    except Exception as error:
        load_errors.append(f"{path.name}: {error}")

if not loaded_models:
    st.error("No model bundles could be loaded.")
    for message in load_errors:
        st.write(f"- {message}")
    st.stop()

# Sidebar navigation and upload controls
st.sidebar.title("🩺 Classifier Lab")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data & Upload", "Compare Models", "Model Details"],
)

st.sidebar.divider()
st.sidebar.subheader("Test dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload labeled CSV",
    type=["csv"],
    help="Include all model input features and the diagnosis target column.",
)

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name
    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
elif SAMPLE_CSV.exists():
    try:
        test_data = pd.read_csv(SAMPLE_CSV)
        data_source = "Project test_data.csv"
        st.sidebar.success("Using the project's test_data.csv")
    except Exception as error:
        st.error(f"Could not read test_data.csv: {error}")
        st.stop()
else:
    test_data = None
    data_source = None
    st.sidebar.info("Upload a labeled test CSV to evaluate the models.")

st.sidebar.caption(f"Saved models found: {len(loaded_models)}")

# Evaluate all loaded models against the same supplied test CSV
results = []
evaluations = {}
evaluation_errors = []

if test_data is not None:
    for name, info in loaded_models.items():
        feature_columns = info["feature_columns"]
        target_column = info["target_column"]
        label_encoder = info["label_encoder"]

        required_columns = feature_columns + [target_column]
        missing_columns = [
            column for column in required_columns
            if column not in test_data.columns
        ]

        if missing_columns:
            evaluation_errors.append(
                f"{name}: CSV is missing {', '.join(missing_columns)}"
            )
            continue

        try:
            X_test = test_data[feature_columns].copy()
            y_test = label_encoder.transform(
                test_data[target_column].astype(str)
            )
            predictions = info["model"].predict(X_test)

            try:
                probabilities = info["model"].predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probabilities)
            except (AttributeError, ValueError, IndexError):
                auc = float("nan")

            # Binary metrics use encoded class 1 as the positive class.
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": auc,
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "MCC": matthews_corrcoef(y_test, predictions),
            })

            evaluations[name] = {
                "y_test": y_test,
                "predictions": predictions,
                "label_encoder": label_encoder,
            }

        except Exception as error:
            evaluation_errors.append(f"{name}: {error}")

st.title("🧬 Tumor Classifier Lab")
st.caption(
    "Breast Cancer Wisconsin (Diagnostic) · Model evaluation workspace"
)

if evaluation_errors:
    with st.expander("Data or model messages"):
        for message in evaluation_errors:
            st.write(f"- {message}")

# Page 1: Overview
if page == "Overview":
    st.subheader("Welcome")
    st.write(
        "Use the navigation menu to inspect your test data, compare all saved "
        "classifiers, or view detailed results for an individual model."
    )

    if test_data is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Test rows", f"{len(test_data):,}")
        col2.metric("Saved models", str(len(loaded_models)))
        col3.metric("Data source", "Uploaded" if uploaded_file else "Local CSV")

        st.info(f"Current dataset: {data_source}")

        if results:
            best_row = max(results, key=lambda row: row["F1"])
            st.success(
                f"Highest F1 on this test file: **{best_row['Model']}** "
                f"({best_row['F1']:.4f})"
            )
        else:
            st.warning(
                "No models could be evaluated. Check the CSV columns and target labels."
            )
    else:
        st.info("Upload your labeled test CSV in the sidebar to begin.")

    with st.expander("What each page contains"):
        st.markdown(
            """
            - **Data & Upload:** inspect the current CSV and its columns.
            - **Compare Models:** view Accuracy, AUC, Precision, Recall, F1, and MCC
              for every model that could be evaluated.
            - **Model Details:** choose a model and view its confusion matrix and
              classification report.
            """
        )

# Page 2: Data and upload
elif page == "Data & Upload":
    st.subheader("Test dataset")
    if test_data is None:
        st.info("Upload a labeled CSV using the sidebar uploader.")
    else:
        st.write(f"Data source: **{data_source}**")
        st.write(f"Rows: **{len(test_data):,}** · Columns: **{len(test_data.columns)}**")
        st.dataframe(test_data.head(20), use_container_width=True)

        st.subheader("Column check")
        expected_features = loaded_models[next(iter(loaded_models))]["feature_columns"]
        expected_target = loaded_models[next(iter(loaded_models))]["target_column"]
        missing = [
            column for column in expected_features + [expected_target]
            if column not in test_data.columns
        ]

        if missing:
            st.error("Required columns missing:")
            st.write(missing)
        else:
            st.success(
                f"All {len(expected_features)} input features and "
                f"`{expected_target}` target column are present."
            )

# Page 3: Compare all models
elif page == "Compare Models":
    st.subheader("Model comparison")
    if not results:
        st.info("Upload a CSV with the required features and target column.")
    else:
        comparison = pd.DataFrame(results).set_index("Model")
        st.write(
            "All available saved models are evaluated on the same uploaded "
            "test data. No training occurs in this app."
        )
        st.dataframe(
            comparison.style.format("{:.4f}"),
            use_container_width=True,
        )

        st.bar_chart(comparison[["Accuracy", "AUC", "F1", "MCC"]])

        st.download_button(
            "Download comparison table",
            data=comparison.to_csv().encode("utf-8"),
            file_name="model_comparison_on_uploaded_data.csv",
            mime="text/csv",
        )

# Page 4: Detailed results for selected model
elif page == "Model Details":
    st.subheader("Individual model results")

    if not evaluations:
        st.info("Upload a valid labeled CSV to show model details.")
    else:
        selected_name = st.selectbox(
            "Choose a model",
            list(evaluations.keys()),
        )
        selected = evaluations[selected_name]
        y_test = selected["y_test"]
        predictions = selected["predictions"]
        class_names = list(selected["label_encoder"].classes_)

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=range(len(class_names)),
        )
        matrix_df = pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in class_names],
            columns=[f"Predicted {label}" for label in class_names],
        )

        matrix_tab, report_tab = st.tabs(
            ["Confusion Matrix", "Classification Report"]
        )

        with matrix_tab:
            st.write(f"Results for **{selected_name}**")
            st.dataframe(matrix_df, use_container_width=True)

        with report_tab:
            report = classification_report(
                y_test,
                predictions,
                labels=range(len(class_names)),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).transpose().round(4),
                use_container_width=True,
            )

        prediction_column = (
            f"predicted_{loaded_models[selected_name]['target_column']}"
        )
        output_data = test_data.copy()
        output_data[prediction_column] = (
            selected["label_encoder"].inverse_transform(predictions)
        )

        st.download_button(
            "Download this model's predictions",
            data=output_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_name.lower().replace(' ', '_')}_predictions.csv",
            mime="text/csv",
        )

st.markdown(
    """
    <style>
    .sparkle-layer {
        position: fixed;
        inset: 0;
        overflow: hidden;
        pointer-events: none;
        z-index: 10;
    }

    .sparkle {
        position: absolute;
        color: #b8fbff;
        text-shadow: 0 0 8px #55eaff, 0 0 18px #55eaff;
        opacity: 0;
        animation: sparkle-float 6s ease-in-out infinite;
    }

    .sparkle:nth-child(1) { left: 8%;  top: 82%; font-size: 18px; animation-delay: 0s; }
    .sparkle:nth-child(2) { left: 22%; top: 65%; font-size: 13px; animation-delay: 1s; }
    .sparkle:nth-child(3) { left: 39%; top: 90%; font-size: 20px; animation-delay: 2s; }
    .sparkle:nth-child(4) { left: 61%; top: 72%; font-size: 14px; animation-delay: 3s; }
    .sparkle:nth-child(5) { left: 78%; top: 88%; font-size: 19px; animation-delay: 1.5s; }
    .sparkle:nth-child(6) { left: 91%; top: 63%; font-size: 12px; animation-delay: 2.5s; }

    @keyframes sparkle-float {
        0%   { opacity: 0; transform: translateY(25px) scale(0.5) rotate(0deg); }
        25%  { opacity: 0.9; }
        70%  { opacity: 0.7; }
        100% { opacity: 0; transform: translateY(-100px) scale(1.3) rotate(90deg); }
    }

    @media (prefers-reduced-motion: reduce) {
        .sparkle { animation: none; opacity: 0.45; }
    }
    </style>

    <div class="sparkle-layer">
        <span class="sparkle">✦</span>
        <span class="sparkle">✧</span>
        <span class="sparkle">✦</span>
        <span class="sparkle">✧</span>
        <span class="sparkle">✦</span>
        <span class="sparkle">✧</span>
    </div>
    """,
    unsafe_allow_html=True,
)
