# 📈 Day 3: Housing Price Prediction - Linear Regression

This is my **Day 3 task** for the **AI/ML Internship Program**.  
In this project, I implemented **simple** and **multiple linear regression** models on a **Housing Price Prediction** dataset. I also created both static and interactive visualizations using Python libraries to analyze model performance and feature relationships.

---

## 📌 What I Did

- **Loaded the dataset** using `Pandas`.
- **Encoded categorical variables** (e.g., `mainroad`, `furnishingstatus`) using `LabelEncoder`.

### 🔹 Performed Linear Regression
- Built a **simple linear regression** model using `area` as the sole feature.
- Built a **multiple linear regression** model using all available features.

### 🔹 Evaluated Models
- Calculated:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R² Score**
- Exported evaluation metrics and coefficients to `model_results.txt`.

### 🔹 Created Visualizations
- **Correlation heatmap** showing feature relationships.
- **Scatter plot with regression line** (Area vs. Price) for simple linear regression.
- **Bar chart** of feature importance for multiple regression.
- **Actual vs. Predicted** price plot to assess model performance.
- Combined all plots into a single image: `housing_price_analysis.png`.

### 🔹 Created Interactive Visualizations (Using Plotly)
- Interactive **2D Scatter Plot**: Area vs. Price → `scatter_plot.html`
- Interactive **3D Scatter Plot**: Area, Bedrooms vs. Price → `3d_scatter_plot.html`

---

## 📊 Tools Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Plotly  

---

## 📂 Files in this Repository

| File Name                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `main.py`                | Main Python script: data preprocessing, training, evaluation, visualizations |
| `model_results.txt`      | Model performance metrics and regression coefficients                       |
| `housing_price_analysis.png` | Combined static plots showing various insights                             |
| `scatter_plot.html`      | Interactive 2D scatter plot (Area vs. Price)                                 |
| `3d_scatter_plot.html`   | Interactive 3D scatter plot (Area, Bedrooms vs. Price)                       |
| `Housing.csv`            | Dataset file (see link below if not present)                                 |
| `README.md`              | This file                                                                     |

---

## 📥 Dataset Source

You can get the Housing Price Prediction dataset from:  
🔗 [Kaggle: Housing Price Prediction Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

---

## ✅ What I Learned

- How to implement **simple** and **multiple linear regression** models.
- How to **preprocess categorical variables** for machine learning.
- How to evaluate regression models using **MAE**, **MSE**, and **R²** score.
- How to create **static visualizations** using Matplotlib and Seaborn.
- How to create **interactive visualizations** using Plotly.
- How to **interpret regression coefficients** and feature importance.

---

## 🎯 Summary

This project builds a strong foundation in:
- Understanding and applying **Linear Regression**
- **Data preprocessing**
- **Model evaluation**
- **Data visualization** — both static and interactive

It serves as an essential stepping stone for more advanced machine learning tasks.

---
