# 🩺 Breast Cancer Classification using Logistic Regression

This project demonstrates how to build and evaluate a **Logistic Regression** model to classify breast cancer cases as **malignant** or **benign** using the **Breast Cancer dataset** from `scikit-learn`.

---

## 📌 Project Overview

- **Dataset**: Built-in `load_breast_cancer` from `sklearn.datasets`
- **Goal**: Predict whether a tumor is **malignant (0)** or **benign (1)**
- **Algorithm**: Logistic Regression
- **Evaluation Metrics**: Confusion Matrix, Precision, Recall, ROC-AUC
- **Additional**: Threshold tuning & ROC curve visualization

---

## 📂 Project Structure

🚀 Steps Performed
Load Dataset

Used load_breast_cancer() from sklearn.datasets.

Data Preparation

Converted features into a Pandas DataFrame.

Split dataset into train (80%) and test (20%) sets.

Standardized features using StandardScaler.

Model Training

Applied Logistic Regression (sklearn.linear_model.LogisticRegression).

Predictions & Evaluation

Generated predictions and probabilities.

Evaluated model using:

Confusion Matrix

Precision

Recall

ROC-AUC Score

Threshold Tuning

Tested thresholds: 0.3, 0.5, 0.7 to observe precision-recall trade-off.

Visualization

Plotted ROC Curve.

Plotted Sigmoid Function for understanding logistic regression output mapping.

