# Personal Loan Default Risk Prediction

## Overview

This project develops a machine learning model to predict the probability of personal loan default using applicant-level financial, credit, employment, and loan-related information.

The project focuses on how credit risk can be approached using data analysis and machine learning, with an emphasis on model evaluation and practical risk segmentation.

## Business Problem

For lenders, identifying applicants with higher default risk is important for improving credit decisions and managing potential losses.

The objective of this project is to:

- Identify factors associated with loan default
- Build a classification model to predict default risk
- Evaluate model performance using appropriate classification metrics
- Select a decision threshold using a validation dataset
- Segment applicants into different risk categories

## Dataset

The dataset contains 25,000 personal loan applications and 22 variables covering:

- Applicant demographics
- Employment information
- Income
- Existing loans and EMIs
- Credit utilization
- Credit inquiries
- Late payments
- Loan amount and tenure
- Loan purpose
- CIBIL score
- Interest rate
- Default risk score

The target variable is:

`default_flag`

where:

- `0` = No default
- `1` = Default

The dataset has a default rate of approximately 6.14%.

## Project Workflow

The analysis follows these stages:

1. Data understanding
2. Exploratory data analysis
3. Missing-value analysis
4. Outlier analysis
5. Train-test splitting
6. Data preprocessing
7. Logistic Regression baseline
8. Random Forest modelling
9. Random Forest hyperparameter tuning
10. Validation-based threshold selection
11. Final model evaluation
12. Risk segmentation
13. Model export

## Models

### Logistic Regression

Logistic Regression was used as the baseline classification model.

Class weights were adjusted to account for the imbalance between default and non-default applicants.

### Random Forest

A Random Forest classifier was developed as the main non-linear model.

Hyperparameters were tuned using GridSearchCV with ROC-AUC as the evaluation metric.

The final model was trained on the complete training dataset after the decision threshold had been selected using a separate validation split.

## Evaluation

The project evaluates model performance using:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

Because loan default is an imbalanced classification problem, accuracy is not treated as the only measure of model performance.

## Threshold Selection

Instead of selecting the classification threshold using the test set, the training data was further divided into:

- Model-fitting data
- Validation data

The classification threshold was selected using the validation set based on F1-score.

The test set was kept untouched until the final evaluation.

## Risk Segmentation

Predicted default probabilities are also used to create three risk categories:

- Low Risk: 0–20%
- Medium Risk: 20–50%
- High Risk: 50–100%

These categories are intended for relative risk segmentation rather than calibrated probability-of-default estimates.

## Project Structure

```text
credit-risk-project/
├── data/
│   └── raw/
├── notebooks/
│   └── 01_data_understanding.ipynb
├── models/
│   └── credit_risk_random_forest.pkl
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
├── .gitignore
├── README.md
└── requirements.txt

```
