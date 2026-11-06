# Breast Cancer Classification

## a. Problem Statement

This project compares five machine learning classification models using the Breast Cancer Wisconsin (Diagnostic) dataset. Each model predicts whether a tumor is benign or malignant based on measurements of cell nuclei.

The models are evaluated with Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The Streamlit app loads saved models and evaluates labeled test CSV data without retraining the models.

This project is for educational purposes only and is not a medical diagnostic tool.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances:** 569
- **Input features:** 30 numeric features
- **Target column:** `diagnosis`
- **Classes:** Benign and malignant
- **Task:** Binary classification

The features describe cell nuclei measurements extracted from digitized fine needle aspirate images. The diagnosis column is the target and is excluded from the model input features.

## c. GitHub Repository Link

[Open the GitHub repository]([PASTE YOUR GITHUB REPOSITORY LINK HERE])

## d. Models Used and Evaluation Metrics

All five classifiers were evaluated on the same held-out test data. For binary Precision, Recall, F1, and AUC, the positive class is **malignant**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.965035 | 0.996226 | 0.980000 | 0.924528 | 0.951456 | 0.925128 |
| Decision Tree | 0.944056 | 0.954088 | 0.978723 | 0.867925 | 0.920000 | 0.880970 |
| K-Nearest Neighbors | 0.958042 | 0.985954 | 0.979592 | 0.905660 | 0.941176 | 0.910336 |
| Naive Bayes | 0.944056 | 0.995178 | 1.000000 | 0.849057 | 0.918367 | 0.883032 |
| Support Vector Machine | 0.972028 | 0.996646 | 1.000000 | 0.924528 | 0.960784 | 0.940844 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed strongly, with 0.965035 accuracy and 0.996226 AUC. Its 0.924528 recall indicates it identified most positive cases in this test set. |
| Decision Tree | Had 0.944056 accuracy and 0.954088 AUC, the lowest AUC of the five models. Its 0.867925 recall was also lower than Logistic Regression and SVM. |
| K-Nearest Neighbors | Achieved 0.958042 accuracy and 0.985954 AUC. Its scores were generally strong, though below SVM on accuracy, F1, and MCC. |
| Naive Bayes | Achieved perfect precision (1.000000) but the lowest recall (0.849057). In this test set, its positive predictions were precise, but it identified fewer of the actual positive cases. Its AUC remained high at 0.995178. |
| Support Vector Machine | Had the highest accuracy (0.972028), AUC (0.996646), F1 (0.960784), and MCC (0.940844). Its precision was 1.000000 and its recall was 0.924528. |

**Overall winner for this dataset:** Support Vector Machine (SVM).

**Reason:** SVM achieved the highest Accuracy, AUC, F1, and MCC among the five models, while tying Naive Bayes for highest Precision. Logistic Regression tied its Recall. This makes SVM the strongest overall performer on the reported test split, based on these metrics.

## Streamlit App

- **Live app:** [PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE]
- Upload a labeled test CSV containing the required input features and target column.
- Compare the evaluation metrics for the available saved models.
- View a confusion matrix and classification report for a selected model.
- The app uses saved models and does not retrain on uploaded data.

## Project Files

```text
streamlit_app.py                  Streamlit application
requirements.txt                  Python dependencies
your_colab_notebook.ipynb         Colab source notebook
test_data.csv                     Labeled test data
model_comparison.csv              Model evaluation results
breast_cancer_awareness.png       Awareness image used in the app
model/                            Saved trained model files
```

## Run Locally

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run streamlit_app.py
```
