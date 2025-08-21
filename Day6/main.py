import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from matplotlib.colors import ListedColormap

# Load and prepare the Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]  # Using petal length and width for visualization
y = iris.target
feature_names = iris.feature_names[2:4]
class_names = iris.target_names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to train and evaluate KNN for different K values
def evaluate_knn(k_values):
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"\nK={k} Accuracy: {accuracy:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (K={k})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_k{k}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    return accuracies

# Function to plot decision boundaries
def plot_decision_boundary(X, y, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # Create mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on mesh points
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and points
    plt.figure(figsize=(10, 8))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#00FF00', '#0000FF']
    
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold),
                        edgecolor='k', s=100)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f'KNN (k={k}) Decision Boundary')
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.savefig(f'decision_boundary_k{k}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Evaluate different K values
k_values = [1, 3, 5, 7, 9]
accuracies = evaluate_knn(k_values)

# Plot accuracy vs K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('accuracy_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot decision boundaries for selected K values
for k in [1, 3, 9]:
    plot_decision_boundary(X_train_scaled, y_train, k)

# Detailed classification report for the best K (based on accuracy)
best_k = k_values[np.argmax(accuracies)]
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)
print(f"\nDetailed Classification Report for Best K={best_k}:")
print(classification_report(y_test, y_pred_best, target_names=class_names))