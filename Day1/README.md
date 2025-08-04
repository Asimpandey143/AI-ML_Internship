# 🚢 Day 1: Titanic Dataset – Data Preprocessing & Cleaning

This marks my **first day** of the AI/ML Internship journey.  
The focus of the task was to clean and preprocess the Titanic dataset to prepare it for machine learning models.

---

## ✅ What I Accomplished

- Imported the dataset using **Pandas**
- Explored and handled missing data:  
  - Replaced missing values in `Age` with the **median**  
  - Filled missing entries in `Embarked` with the **most frequent value**
- Transformed categorical features:  
  - Converted `Sex` to numeric (male → 0, female → 1)  
  - Applied **one-hot encoding** to the `Embarked` column  
- Applied **StandardScaler** to normalize `Age` and `Fare`
- Identified and removed **outliers** in numeric data using the **Interquartile Range (IQR)** method
- Visualized the data through:  
  - Count plots, boxplots, histograms, pie charts, and heatmaps

---

## 🛠 Technologies & Libraries

- **Python**  
- **Pandas**  
- **NumPy**  
- **Matplotlib**  
- **Seaborn**  
- **Scikit-learn**

---

## 📁 Project Files

- `main.py` – Script for data preprocessing and plotting  
- `Figure_1.png` – Visualization image combining multiple plots  
- `Titanic-Dataset/` – Folder containing the CSV dataset  
- `README.md` – Project overview and documentation (this file)

---

## 📥 Dataset Reference

- Source: [Kaggle - Titanic Dataset by Yasser H.](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

---

## 📚 Lessons Learned

- Practical workflow of data cleaning  
- Techniques for handling missing values in datasets  
- Converting categorical to numerical data  
- Normalizing features using standard scaling  
- Outlier detection and visualization using Python tools
