# 🌸 K-Nearest Neighbors (KNN) on Iris Dataset

This project implements the **K-Nearest Neighbors (KNN)** algorithm on the **Iris dataset**, using **petal length** and **petal width** as features for classification.  
It evaluates model performance for different values of **K**, visualizes **confusion matrices**, **decision boundaries**, and plots **accuracy vs K**.

---

## 📌 Features

- **Data Preprocessing**
  - Loads the Iris dataset from `scikit-learn`
  - Selects **petal length** and **petal width**
  - Standardizes features with `StandardScaler`
- **Model Training**
  - Trains KNN models with different `K` values
  - Evaluates accuracy on test data
- **Visualization**
  - Confusion matrix heatmaps for each `K`
  - Decision boundaries for selected `K` values
  - Accuracy vs. K value plot
- **Performance Metrics**
  - Accuracy scores
  - Detailed classification report for the best K

---

## 📊 Example Outputs

### Accuracy vs. K Value
![Accuracy vs K](accuracy_vs_k.png)

### Confusion Matrix (K=3)
![Confusion Matrix K=3](confusion_matrix_k1.png)

### Decision Boundary (K=3)
![Decision Boundary K=3](confusion_matrix_k1.png)

---

