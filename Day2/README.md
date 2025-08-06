# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

This repository contains a comprehensive Exploratory Data Analysis (EDA) of the Titanic dataset using Python, Pandas, Matplotlib, and Seaborn.

---

## 📁 Files in This Repo

- `Titanic-Dataset.csv` – The dataset used for analysis (make sure to add this).
- `eda_summary.png` – Final combined visual summary of the EDA.
- `summary_statistics.csv` – Summary statistics including mean, median, std, etc.
- `eda_hist_box_combined.png` – Combined histograms and boxplots of numerical features.
- `eda_correlation.png` – Heatmap of correlation between numeric features.
- `eda_pairplot.png` – Pairplot for key features colored by survival status.
- `eda_script.py` – Python script used to perform the entire analysis.

---

## ✅ EDA Tasks Performed

### 📊 Summary Statistics
- Computed basic descriptive statistics for all columns (mean, std, min, max, etc.)
- Saved output to `summary_statistics.csv`

### 📈 Visualization of Numeric Features
- Histograms and Boxplots for each numerical column
- Saved as `eda_hist_box_combined.png`

### 🔥 Correlation Matrix
- Computed and visualized correlation between numeric features using a heatmap
- Saved as `eda_correlation.png`

### 🔍 Pairplot Analysis
- Created pairplots of selected features (`Age`, `Fare`, `Pclass`, `Survived`)
- Colored by survival to see class separability
- Saved as `eda_pairplot.png`

### 🖼️ Combined Output
- Merged all visualizations into a single image file: `eda_summary.png` for easy reference

---

## 🛠️ Tools & Libraries Used

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- PIL (Pillow)

---

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-eda.git
   cd titanic-eda
