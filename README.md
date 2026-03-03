# Credit-Card-Fraud-Detection
📌 Problem Statement

Credit card fraud detection is a critical classification problem where the objective is to identify fraudulent transactions from highly imbalanced transactional data. Since fraudulent transactions are rare, traditional accuracy metrics can be misleading.

This project builds a machine learning pipeline to detect fraudulent transactions using Logistic Regression with SMOTE for imbalance handling and GridSearchCV for hyperparameter tuning.

📊 Dataset Description

Source: Public credit card transaction dataset (Kaggle)

Total transactions: 284,807

Fraud cases: 492 (~0.17%)

Non-fraud cases: 284,315 (~99.83%)

Highly imbalanced dataset

Features V1–V28 are PCA-transformed.
‘Amount’ was scaled using StandardScaler.

⚙️ Project Workflow

Data Exploration & Cleaning

Class Imbalance Analysis

Train-Test Split (Stratified)

SMOTE applied only on training data

Logistic Regression Model

Hyperparameter tuning using GridSearchCV

Model evaluation using:

Confusion Matrix

Classification Report

ROC-AUC Score

🔧 Hyperparameter Tuning

GridSearchCV was used with 3-fold cross-validation.

Best Parameters:

C = 10

Penalty = L2

Solver = lbfgs

Scoring metric used: ROC-AUC

📈 Model Performance

Confusion Matrix:

[[55355 1509]
[8 90]]

Classification Report (Fraud Class):

Precision: 0.06

Recall: 0.92

F1-score: 0.11

ROC-AUC Score: 0.97

🧠 Key Insights

The dataset is extremely imbalanced.

SMOTE significantly improved fraud detection recall.

The model achieves high recall (92%), meaning most fraudulent transactions are detected.

Precision is low, indicating false positives are high.

In fraud detection systems, high recall is often prioritized since missing a fraud case is costlier than incorrectly flagging a genuine transaction.

🚀 Future Improvements

Threshold tuning to improve precision

Compare with Random Forest / XGBoost

Deploy using Streamlit or Flask API

Implement cost-sensitive learning

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Jupyter Notebook
