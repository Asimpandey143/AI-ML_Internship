import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess data
def load_and_preprocess_data(file_path='breast-cancer.csv'):
    # Load dataset from local CSV
    df = pd.read_csv(file_path)
    
    # Features and target
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Visualize decision boundary for 2D data
def plot_decision_boundary(X, y, model, kernel_name, ax):
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Create mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict on mesh grid
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_title(f'SVM Decision Boundary ({kernel_name} Kernel)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(*scatter.legend_elements(), title="Classes", labels=['Benign', 'Malignant'])

# Train and evaluate SVM
def train_and_evaluate_svm(X, y, kernel, param_grid):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with GridSearchCV
    svm = SVC(kernel=kernel, probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    print(f"\n{kernel.upper()} Kernel Results:")
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix ({kernel} Kernel)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{kernel}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_model, X_train, y_train

# Main execution
def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Define parameter grids
    param_grid_linear = {'C': [0.1, 1, 10, 100]}
    param_grid_rbf = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1]}
    
    # Train and evaluate Linear SVM
    linear_model, X_train_linear, y_train_linear = train_and_evaluate_svm(X, y, 'linear', param_grid_linear)
    
    # Train and evaluate RBF SVM
    rbf_model, X_train_rbf, y_train_rbf = train_and_evaluate_svm(X, y, 'rbf', param_grid_rbf)
    
    # Visualize decision boundaries
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_decision_boundary(X_train_linear, y_train_linear, linear_model, 'Linear', ax1)
    plot_decision_boundary(X_train_rbf, y_train_rbf, rbf_model, 'RBF', ax2)
    plt.tight_layout()
    plt.savefig('svm_decision_boundaries.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()