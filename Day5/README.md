# 🌳 Task 5: Decision Trees & Random Forests

This project demonstrates the use of **Decision Trees** and **Random Forests** for classification tasks using the **Heart Disease Dataset**.  
We explore tree-based models, control overfitting, interpret feature importance, and evaluate model performance.

---

## 📌 Objective
- Learn how tree-based models work for **classification** and **regression**.
- Understand **overfitting** in decision trees and ways to control it.
- Compare **Decision Trees** vs **Random Forests**.
- Interpret **feature importances** to understand model decisions.
- Use **cross-validation** for robust evaluation.

---

## 🛠 Tools & Libraries
- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Graphviz** (for tree visualization)

---

## 📂 Dataset
We use the **Heart Disease Dataset** from Kaggle:  
[📥 Click here to download](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Dataset Features Example**:
- `age` – Age of the patient
- `sex` – Gender (1 = male, 0 = female)
- `cp` – Chest pain type
- `trestbps` – Resting blood pressure
- `chol` – Serum cholesterol (mg/dl)
- `thalach` – Maximum heart rate achieved
- `target` – Heart disease presence (1 = yes, 0 = no)

---

## 🚀 Steps Implemented

### 1️⃣ Train a Decision Tree Classifier
- Loaded dataset and performed **data preprocessing**.
- Trained a **Decision Tree Classifier** using Scikit-learn.
- Visualized the tree using **Graphviz**.
  
### 2️⃣ Analyze Overfitting & Control Depth
- Compared performance of deep trees vs shallow trees.
- Controlled tree depth using:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`

### 3️⃣ Train a Random Forest & Compare Accuracy
- Trained a **Random Forest Classifier**.
- Compared accuracy with the Decision Tree model.
- Observed that Random Forest reduced overfitting.

### 4️⃣ Interpret Feature Importances
- Extracted **feature importance scores** from models.
- Visualized top contributing features.

### 5️⃣ Evaluate Using Cross-Validation
- Performed **K-Fold Cross-Validation** for better performance estimation.

---

## 📊 Results
| Model                  | Accuracy | Cross-Validation Score |
|------------------------|----------|------------------------|
| Decision Tree (Default) | 78.05%   | 70.73% (+/- 9.26%)      |
| Random Forest           | 98.54%   | 83.90% (+/- 9.05%)      |

---

## 📷 Visualization Examples
- **Decision Tree Structure**
- **Feature Importance Bar Chart**
- **Confusion Matrix Heatmap**

---

## 📝 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Amanjha112113/AI-ML_Internship_ElevateLabs/tree/main/Day-5
   cd task5-decision-trees
