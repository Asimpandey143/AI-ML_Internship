# 🧠 Breast Cancer Classification with SVM

This project implements a **Support Vector Machine (SVM)** classifier to detect whether a tumor is **Benign (B)** or **Malignant (M)** using the **Breast Cancer Wisconsin Diagnostic Dataset**.  
It includes **data preprocessing**, **hyperparameter tuning (GridSearchCV)**, **model evaluation**, and **decision boundary visualization**.

---

## 📌 Features

- **Data Preprocessing**  
  - Standardization using `StandardScaler`  
  - Encoding target labels (`M` → 1, `B` → 0)

- **Model Training**  
  - Linear and RBF kernel SVMs  
  - Hyperparameter tuning with `GridSearchCV`  
  - Cross-validation for robust performance estimation

- **Model Evaluation**  
  - Classification report (precision, recall, f1-score)  
  - Confusion matrix visualization  
  - Decision boundary plots (PCA-reduced 2D space)

- **Reproducibility**  
  - Random seed fixed (`np.random.seed(42)`)

---
