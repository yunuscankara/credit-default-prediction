# Credit Default Prediction and Model Explainability

A machine learning project for predicting credit card default using historical customer payment information from the UCI Credit Card dataset.

## Project Overview

This project builds a binary classification model to predict whether a customer is likely to default on their credit card payment in the following month.

A Random Forest classifier is trained and evaluated using multiple classification metrics. The project also includes a baseline comparison, feature importance analysis, SHAP-based model explainability, and a basic fairness audit across `SEX` groups.

## Project Structure

```text
.
├── credit_default_prediction.ipynb
├── data/
│   └── UCI_Credit_Card.csv
├── README.md
└── requirements.txt
```

## Technologies

- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- Jupyter Notebook

## Modeling Process

1. Load and inspect the dataset
2. Check missing values and duplicate rows
3. Analyze the target class distribution
4. Remove the `ID` identifier from the feature set
5. Split the data using a stratified train/test split
6. Build a majority-class baseline
7. Train a Random Forest classifier
8. Evaluate the model using:
   - Accuracy
   - Balanced Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC
   - PR-AUC
9. Visualize the Confusion Matrix
10. Plot ROC and Precision-Recall curves
11. Analyze Random Forest feature importance
12. Use SHAP for model explainability
13. Compare basic performance metrics across `SEX` groups

## Methodological Notes

- The `ID` column is excluded because it is only an identifier and does not represent a meaningful predictive feature.
- Random Forest does not require feature scaling, so `StandardScaler` is not used.
- The train/test split is performed before model training to reduce the risk of data leakage.
- The target variable is imbalanced, with default cases representing a minority of the observations. Therefore, accuracy is not used as the only evaluation metric.
- The fairness analysis is exploratory and reports group-level performance differences. It should not be interpreted as a legal or ethical certification of the model.

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients Dataset**

The dataset contains 30,000 records and 25 variables. The target variable is:

```text
default.payment.next.month
```

where:

- `0` = no default
- `1` = default

## Installation

```bash
pip install -r requirements.txt
```

Then open the notebook with Jupyter:

```bash
jupyter notebook
```

and run `credit_default_prediction.ipynb`.

## Purpose

This project was originally developed as an academic machine learning project and has been further cleaned and structured for portfolio use.

It is intended for educational and portfolio purposes. A real-world credit decision system should not rely solely on model predictions and would require additional legal, ethical, domain, and organizational considerations.
