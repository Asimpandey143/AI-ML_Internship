# 🛍️ Mall Customers Segmentation with K-Means Clustering

This project applies **K-Means clustering** to segment mall customers based on **Annual Income** and **Spending Score**.  
It uses **Elbow Method** and **Silhouette Scores** to find the optimal number of clusters and visualizes the customer segments.

---

## 📌 Features

- **Data Preprocessing**  
  - Selects features: `Annual Income (k$)` and `Spending Score (1-100)`  
  - Standardizes data using `StandardScaler`

- **Optimal Cluster Selection**  
  - **Elbow Method** (Inertia vs. K)  
  - **Silhouette Score** evaluation

- **Clustering**  
  - Fits **K-Means** model with the optimal K  
  - Assigns each customer to a cluster  
  - Saves results in `mall_customers_clustered.csv`

- **Visualization**  
  - Elbow Method Plot (`elbow_method.png`)  
  - Silhouette Scores Plot (`silhouette_scores.png`)  
  - Customer Clusters Plot (`kmeans_clusters.png`)

- **Interactive Chart Configs** (Chart.js format) printed in console

---
