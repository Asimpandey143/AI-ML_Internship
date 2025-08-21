import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import opendatasets as od

# Set random seed for reproducibility
np.random.seed(42)

# Download dataset from Kaggle or load locally
def load_dataset(local_path='Mall_Customers.csv'):
    if os.path.exists(local_path):
        print(f"Loading dataset from {local_path}")
        return pd.read_csv(local_path)
    else:
        dataset_url = 'https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python'
        print("Downloading dataset from Kaggle...")
        od.download(dataset_url)
        return pd.read_csv('./customer-segmentation-tutorial-in-python/Mall_Customers.csv')

# Preprocess data
def preprocess_data(df):
    # Select relevant features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Elbow Method to find optimal K
def plot_elbow_method(X, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()

# Compute Silhouette Scores and plot
def compute_silhouette_scores(X, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    # Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Different K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('silhouette_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return silhouette_scores

# Visualize clusters
def plot_clusters(X, labels, centers, scaler):
    # Inverse transform centers for original scale
    centers_orig = scaler.inverse_transform(centers)
    X_orig = scaler.inverse_transform(X)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_orig[:, 0], X_orig[:, 1], c=labels, cmap='viridis', edgecolors='k')
    plt.scatter(centers_orig[:, 0], centers_orig[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title('K-Means Clustering (Annual Income vs Spending Score)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
def main():
    # Load and preprocess data
    df = load_dataset()
    X_scaled, scaler = preprocess_data(df)
    
    # Plot Elbow Method
    plot_elbow_method(X_scaled)
    
    # Compute and plot Silhouette Scores
    silhouette_scores = compute_silhouette_scores(X_scaled)
    optimal_k = 2 + np.argmax(silhouette_scores)  # Silhouette starts at k=2
    print(f"Optimal number of clusters (based on Silhouette Score): {optimal_k}")
    
    # Fit K-Means with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_
    
    # Evaluate clustering
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score for K={optimal_k}: {silhouette_avg:.4f}")
    
    # Visualize clusters
    plot_clusters(X_scaled, labels, centers, scaler)
    
    # Add cluster labels to original dataset and save
    df['Cluster'] = labels
    df.to_csv('mall_customers_clustered.csv', index=False)
    print("Clustered dataset saved as 'mall_customers_clustered.csv'")

if __name__ == "__main__":
    main()