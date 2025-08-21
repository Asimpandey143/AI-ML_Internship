import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Import and preprocess the dataset
df = pd.read_csv('Housing.csv')

# Convert categorical variables to numeric
le = LabelEncoder()
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea', 'furnishingstatus']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# 2. Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Linear Regression models
# Simple Linear Regression (using area as the only feature)
simple_lr = LinearRegression()
simple_lr.fit(X_train[['area']], y_train)

# Multiple Linear Regression (using all features)
multiple_lr = LinearRegression()
multiple_lr.fit(X_train, y_train)

# 4. Evaluate models
# Simple Linear Regression predictions
y_pred_simple = simple_lr.predict(X_test[['area']])
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

# Multiple Linear Regression predictions
y_pred_multiple = multiple_lr.predict(X_test)
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

# Print evaluation metrics
print("Simple Linear Regression Metrics:")
print(f"MAE: {mae_simple:.2f}")
print(f"MSE: {mse_simple:.2f}")
print(f"R²: {r2_simple:.2f}")
print("\nMultiple Linear Regression Metrics:")
print(f"MAE: {mae_multiple:.2f}")
print(f"MSE: {mse_multiple:.2f}")
print(f"R²: {r2_multiple:.2f}")

# 5. Data Visualizations
sns.set_style('darkgrid')  # Replaced plt.style.use('seaborn')
sns.set_palette("husl")

# Create figure for multiple visualizations
fig = plt.figure(figsize=(20, 15))

# Visualization 1: Correlation Heatmap
plt.subplot(2, 2, 1)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Features')

# Visualization 2: Simple Linear Regression Scatter Plot
plt.subplot(2, 2, 2)
plt.scatter(X_test['area'], y_test, color='blue', alpha=0.5, label='Actual')
plt.plot(X_test['area'], y_pred_simple, color='red', label='Regression Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Area vs Price')
plt.legend()

# Visualization 3: Feature Importance for Multiple Linear Regression
plt.subplot(2, 2, 3)
feature_importance = pd.Series(multiple_lr.coef_, index=X.columns)
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance in Multiple Linear Regression')
plt.xlabel('Coefficient Value')

# Visualization 4: Actual vs Predicted Prices
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_multiple, color='purple', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices (Multiple Linear Regression)')

plt.tight_layout()
plt.savefig('housing_price_analysis.png')
plt.close()

# Plotly Interactive Visualizations
# Interactive Scatter Plot
fig1 = px.scatter(df, x='area', y='price', color='bedrooms', size='bathrooms',
                 hover_data=['stories', 'furnishingstatus'],
                 title='Interactive Scatter Plot: Area vs Price')
fig1.update_layout(template='plotly_dark')

# 3D Scatter Plot
fig2 = px.scatter_3d(df, x='area', y='bedrooms', z='price', 
                    color='bathrooms', size='stories',
                    hover_data=['furnishingstatus'],
                    title='3D Scatter Plot: Area, Bedrooms vs Price')
fig2.update_layout(template='plotly_dark')

# Save Plotly figures
fig1.write_html('scatter_plot.html')
fig2.write_html('3d_scatter_plot.html')

# Print interpretation of coefficients
print("\nMultiple Linear Regression Coefficients:")
for feature, coef in zip(X.columns, multiple_lr.coef_):
    print(f"{feature}: {coef:.2f}")

# Save results to a text file
with open('model_results.txt', 'w') as f:
    f.write("Simple Linear Regression Metrics:\n")
    f.write(f"MAE: {mae_simple:.2f}\n")
    f.write(f"MSE: {mse_simple:.2f}\n")
    f.write(f"R²: {r2_simple:.2f}\n")
    f.write("\nMultiple Linear Regression Metrics:\n")
    f.write(f"MAE: {mae_multiple:.2f}\n")
    f.write(f"MSE: {mse_multiple:.2f}\n")
    f.write(f"R²: {r2_multiple:.2f}\n")
    f.write("\nMultiple Linear Regression Coefficients:\n")
    for feature, coef in zip(X.columns, multiple_lr.coef_):
        f.write(f"{feature}: {coef:.2f}\n")
