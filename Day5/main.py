import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from sklearn.tree import export_graphviz
import os

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the dataset
def load_and_preprocess_data(file_path='heart.csv'):
    """Load and preprocess the heart disease dataset"""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: 'heart.csv' not found. Please ensure the file is in the same directory.")
        exit(1)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns

# 2. Train and visualize Decision Tree
def train_and_visualize_dt(X_train, y_train, feature_names, max_depth=3):
    """Train Decision Tree and create visualization"""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # Export tree visualization
    dot_data = export_graphviz(dt, 
                             out_file=None,
                             feature_names=feature_names,
                             class_names=['No Disease', 'Disease'],
                             filled=True,
                             rounded=True)
    
    # Save tree visualization
    try:
        graph = Source(dot_data, format='png')
        graph.render(filename="decision_tree", directory='.', cleanup=True)
        print("Decision Tree visualization saved as 'decision_tree.png'")
    except Exception as e:
        print(f"Error rendering Decision Tree visualization: {e}")
        print("Ensure Graphviz system binary is installed and in PATH (run 'dot -V' to check).")
    
    return dt

# 3. Analyze overfitting with different depths
def analyze_overfitting(X_train, X_test, y_train, y_test):
    """Analyze Decision Tree performance with varying depths"""
    depths = range(1, 15)
    train_scores = []
    test_scores = []
    
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
        test_scores.append(accuracy_score(y_test, dt.predict(X_test)))
    
    # Plot overfitting analysis
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, label='Training Accuracy')
    plt.plot(depths, test_scores, label='Testing Accuracy')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree: Training vs Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('overfitting_analysis.png')
    plt.close()

# 4. Train Random Forest and get feature importances
def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train Random Forest and analyze feature importances"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances in Random Forest')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    return rf

# 5. Evaluate models
def evaluate_models(dt, rf, X_test, y_test):
    """Evaluate both models using accuracy and cross-validation"""
    # Decision Tree evaluation
    dt_pred = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_cv_scores = cross_val_score(dt, X_test, y_test, cv=5)
    
    # Random Forest evaluation
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_cv_scores = cross_val_score(rf, X_test, y_test, cv=5)
    
    # Print results
    print("Decision Tree Results:")
    print(f"Accuracy: {dt_accuracy:.4f}")
    print(f"Cross-validation scores: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std() * 2:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, dt_pred))
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"Cross-validation scores: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))

# Main execution
def main():
    # Check if Graphviz binary is installed
    if os.system('dot -V') != 0:
        print("Error: Graphviz system binary not found. Install it using 'brew install graphviz' or manually.")
        print("See https://graphviz.org/download/ for manual installation.")
        exit(1)
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and visualize Decision Tree
    dt = train_and_visualize_dt(X_train, y_train, feature_names)
    
    # Analyze overfitting
    analyze_overfitting(X_train, X_test, y_train, y_test)
    
    # Train Random Forest
    rf = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    
    # Evaluate models
    evaluate_models(dt, rf, X_test, y_test)
    
    print("\nVisualizations generated:")
    print("- decision_tree.png: Decision Tree visualization")
    print("- overfitting_analysis.png: Training vs Testing accuracy plot")
    print("- feature_importances.png: Random Forest feature importances")

if __name__ == "__main__":
    main()